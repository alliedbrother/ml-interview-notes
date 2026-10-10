"""Stateful environments, group advantages and observation masking for agentic RL.

Purpose
    Lab L15.2 of module 15 (Agentic and Tool-Use Fine-Tuning). L15.3 uses the TRL
    examples/async_grpo_harbor example; this script does not cover it.
    --smoke builds the whole multi-turn RL loop in miniature on CPU: a stateful
    number-guessing environment with reset()/guess()/get_reward() (the same
    contract TRL's environment_factory expects), a scripted policy, and a tiny
    tabular softmax policy trained with GRPO-style group-relative advantages.
    It asserts that the success rate rises above a threshold and that
    environment (observation) tokens carry zero loss weight.
    --full trains Qwen/Qwen3-1.7B with TRL GRPOTrainer(environment_factory=...)
    against the OpenEnv Wordle environment (TextArena).

Module
    content/courses/finetuning/15-agentic-and-tool-use-fine-tuning.md

Hardware tier for --full
    T1 (one 24 GB GPU, vLLM colocate, Qwen3-1.7B) or T2 (80 GB, Qwen3-4B).
    Duplicate the openenv/wordle Space or run its Docker image locally: the
    hosted Space has limited concurrency and the trainer opens one session per
    generation.

Pinned versions for --full (as of October 2026)
    trl[vllm]>=1.15, transformers>=5.2 (required by environment_factory),
    openenv-textarena from https://huggingface.co/spaces/openenv/wordle

Run (from the repository root)
    python content/courses/finetuning/code/env_rollout.py --smoke
    python content/courses/finetuning/code/env_rollout.py --full --env-url https://<your-copy>.hf.space
"""

from __future__ import annotations

import argparse
import random

import torch


class GuessEnv:
    """Guess a hidden integer in [1, n]. Feedback: 'higher', 'lower' or 'correct'.

    reset() returns the prompt; guess() is the only tool; get_reward() scores the
    episode from internal state, exactly the shape TRL's environment_factory uses.
    """

    def __init__(self, n=8, max_turns=3):
        self.n, self.max_turns = n, max_turns

    def reset(self, seed=None, **kwargs):
        self.rng = random.Random(seed)
        self.target = self.rng.randint(1, self.n)
        self.turns, self.solved, self.lo, self.hi = 0, False, 1, self.n
        return f"Guess a number from 1 to {self.n}. You have {self.max_turns} guesses."

    def guess(self, value: int) -> str:
        """Submit a guess.

        Args:
            value: The guessed integer.

        Returns:
            Feedback: higher, lower or correct.
        """
        if self.solved or self.turns >= self.max_turns:
            raise ValueError("Game over.")
        self.turns += 1
        if value == self.target:
            self.solved = True
            return "correct"
        if value < self.target:
            self.lo = max(self.lo, value + 1)
            return "higher"
        self.hi = min(self.hi, value - 1)
        return "lower"

    def get_reward(self) -> float:
        return 1.0 if self.solved else 0.0


# ---------------- tabular policy over the (lo, hi) belief state ----------------


class TabularPolicy:
    def __init__(self, n):
        self.n = n
        self.logits = torch.zeros(n + 2, n + 2, n, requires_grad=True)  # (lo, hi, action)

    def dist(self, lo, hi):
        return torch.distributions.Categorical(logits=self.logits[lo, hi])


def rollout(env, policy, seed, gen):
    """One episode. Returns reward, the summed log-prob of policy actions, and a token trace."""
    prompt = env.reset(seed=seed)
    trace = [(prompt, 0)]  # (text, loss_weight): prompt is context
    logp = torch.zeros(())
    while True:
        d = policy.dist(env.lo, env.hi)
        a = d.sample(generator=None) if gen is None else torch.multinomial(d.probs, 1, generator=gen)[0]
        logp = logp + d.log_prob(a)
        trace.append((f"<tool_call>guess({int(a) + 1})</tool_call>", 1))  # model-generated: trained
        try:
            obs = env.guess(int(a) + 1)
        except ValueError as e:
            obs = str(e)
        trace.append((f"<tool_response>{obs}</tool_response>", 0))  # environment: masked
        if env.solved or env.turns >= env.max_turns:
            break
    return env.get_reward(), logp, trace


def success_rate(policy, episodes, seed0, gen):
    env = GuessEnv()
    with torch.no_grad():
        return sum(rollout(env, policy, seed0 + i, gen)[0] for i in range(episodes)) / episodes


def train(iterations=100, group=8, lr=5.0, seed=0):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    policy = TabularPolicy(8)
    opt = torch.optim.SGD([policy.logits], lr=lr)
    before = success_rate(policy, 400, 10_000, gen)
    for it in range(iterations):
        rewards, logps = [], []
        for g in range(group):  # one prompt (seed), G rollouts: a GRPO group
            env = GuessEnv()
            r, lp, trace = rollout(env, policy, seed=it, gen=gen)
            rewards.append(r)
            logps.append(lp)
            weights = [w for t, w in trace]
            assert all(w == 0 for t, w in trace if t.startswith("<tool_response>")), "env tokens must be masked"
            assert sum(weights) == env.turns
        r = torch.tensor(rewards)
        adv = (r - r.mean()) / (r.std(unbiased=False) + 1e-4)  # group-relative advantage
        loss = -(adv * torch.stack(logps)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = success_rate(policy, 400, 10_000, gen)
    return before, after


def smoke():
    # Scripted policy: binary search solves 7 of 8 targets within 3 guesses.
    solved = 0
    for target_seed in range(200):
        env = GuessEnv()
        env.reset(seed=target_seed)
        while not (env.solved or env.turns >= env.max_turns):
            env.guess((env.lo + env.hi) // 2)
        solved += env.solved
    assert 0.8 < solved / 200 < 0.95, solved
    # Determinism: same seed, same hidden state.
    e1, e2 = GuessEnv(), GuessEnv()
    assert e1.reset(seed=3) == e2.reset(seed=3) and e1.target == e2.target
    before, after = train()
    print(f"success rate before={before:.3f} after={after:.3f} (uniform-random baseline ~{1 - (7 / 8) ** 3:.3f})")
    assert before < 0.45, before
    assert after > 0.6, after
    print("smoke ok")


def full(args):
    from datasets import Dataset
    from textarena_env import TextArenaAction, TextArenaEnv
    from trl import GRPOConfig, GRPOTrainer

    url = args.env_url

    class WordleEnv:
        def __init__(self):
            self.client = None

        def reset(self, **kwargs):
            if self.client is not None:
                self.client.close()
            self.client = TextArenaEnv(base_url=url)
            result = self.client.reset()
            self._last = result.observation.messages[0].content
            self.reward, self.done = 0.0, False
            return self._last

        def guess(self, guess: str) -> str:
            """
            Make a guess in the Wordle environment.

            Args:
                guess: The guessed word, formatted as '[abcde]'

            Returns:
                The feedback message from the environment.
            """
            if self.done:
                raise ValueError("Game over.")
            result = self.client.step(TextArenaAction(message=guess))
            full_fb = result.observation.messages[0].content
            fb, self._last = full_fb[len(self._last):], full_fb
            self.reward = 0.0 if "You attempted an invalid move" in fb else result.reward
            self.done = result.done
            return fb

        def get_reward(self) -> float:
            return float(self.reward)

    prompt = ("You are an expert Wordle solver. Guess the secret 5-letter word in 6 tries. "
              "Use the tool `guess` with a word formatted as [abcde].")
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": prompt}]] * args.episodes})
    cfg = GRPOConfig(output_dir=args.output_dir, use_vllm=True, vllm_mode="colocate",
                     vllm_gpu_memory_utilization=0.3, chat_template_kwargs={"enable_thinking": False},
                     max_completion_length=1024, num_generations=4, per_device_train_batch_size=1,
                     gradient_accumulation_steps=32, learning_rate=1e-6, logging_steps=1,
                     max_tool_calling_iterations=8, log_completions=True)
    trainer = GRPOTrainer(model=args.model, args=cfg, train_dataset=dataset, environment_factory=WordleEnv)
    trainer.train()
    trainer.save_model(args.output_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--env-url", default="https://openenv-wordle.hf.space")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--output-dir", default="wordle-grpo-qwen3-1.7b")
    args = ap.parse_args()
    if args.full:
        full(args)
    else:
        smoke()


if __name__ == "__main__":
    main()
