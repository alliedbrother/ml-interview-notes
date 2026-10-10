"""Supervised fine-tuning data for the capstone GPT (module 36).

Thirty fixed facts about *The Adventures of Sherlock Holmes* (12 story numbers
-> titles, 18 characters), rendered through three question templates. A and B
are for training; C is a held-out paraphrase used only for evaluation.

Rendered example (prompt | answer):
    "Q: Which story is number IX?\\nA:" | " The Adventure of the Engineer’s Thumb<|endoftext|>"

The loss sees only answer tokens: prompt positions and right padding carry the
label -100 (cross_entropy's ignore_index), so the mean is over answer tokens.

Interface:
    FACTS, TEMPLATES, IGNORE_INDEX
    build_examples(split) -> list[(prompt, answer)]       split in {"train", "heldout"}
    encode_example(tokenizer, prompt, answer, max_len) -> (input_ids, labels)
    collate(batch, pad_id) -> (x, y)                        right padding
    greedy_answers(model, tokenizer, prompts, max_new_tokens=24) -> list[str]
    answer_exact_match(model, tokenizer, examples, max_new_tokens=24) -> float
    generation_backend() -> "llm.generate" | "local-fallback"
"""

from __future__ import annotations

from collections import defaultdict

import torch

from llm.model import IGNORE_INDEX

EOT = "<|endoftext|>"

# (kind, slot, answer). The slot fills the question; the answer is what the model must produce.
FACTS = [
    ("story", "I", "A Scandal in Bohemia"),
    ("story", "II", "The Red-Headed League"),
    ("story", "III", "A Case of Identity"),
    ("story", "IV", "The Boscombe Valley Mystery"),
    ("story", "V", "The Five Orange Pips"),
    ("story", "VI", "The Man with the Twisted Lip"),
    ("story", "VII", "The Adventure of the Blue Carbuncle"),
    ("story", "VIII", "The Adventure of the Speckled Band"),
    ("story", "IX", "The Adventure of the Engineer’s Thumb"),
    ("story", "X", "The Adventure of the Noble Bachelor"),
    ("story", "XI", "The Adventure of the Beryl Coronet"),
    ("story", "XII", "The Adventure of the Copper Beeches"),
    ("char", "Holmes's friend and chronicler", "Dr. Watson"),
    ("char", "the woman who outwitted Holmes", "Irene Adler"),
    ("char", "the landlady at Baker Street", "Mrs. Hudson"),
    ("char", "the Scotland Yard inspector", "Lestrade"),
    ("char", "the King of Bohemia in disguise", "Count Von Kramm"),
    ("char", "the red-headed pawnbroker", "Jabez Wilson"),
    ("char", "the pawnbroker's clever assistant", "John Clay"),
    ("char", "the typist who lost her fiancé", "Mary Sutherland"),
    ("char", "the vanished fiancé", "Hosmer Angel"),
    ("char", "the son accused at Boscombe Valley", "James McCarthy"),
    ("char", "the client sent five orange pips", "John Openshaw"),
    ("char", "the man behind the beggar Hugh Boone", "Neville St. Clair"),
    ("char", "the commissionaire who found the goose", "Peterson"),
    ("char", "the hotel attendant who stole the jewel", "James Ryder"),
    ("char", "the violent stepfather at Stoke Moran", "Dr. Grimesby Roylott"),
    ("char", "the sister who feared the speckled band", "Helen Stoner"),
    ("char", "the engineer who lost his thumb", "Victor Hatherley"),
    ("char", "the nobleman whose bride vanished", "Lord Robert St. Simon"),
]

TEMPLATES = {
    "A": {"story": "Q: Which story is number {slot}?\nA:", "char": "Q: Who is {slot}?\nA:"},
    "B": {"story": "Q: What is the title of story {slot}?\nA:", "char": "Q: Name {slot}.\nA:"},
    "C": {"story": "Q: Story {slot} is called what?\nA:", "char": "Q: Tell me who {slot} is.\nA:"},
}
SPLIT_TEMPLATES = {"train": ("A", "B"), "heldout": ("C",)}


def build_examples(split: str = "train") -> list[tuple[str, str]]:
    """Deterministic (prompt, answer) pairs; the answer starts with a space and has no EOT."""
    if split not in SPLIT_TEMPLATES:
        raise ValueError(f"split must be one of {tuple(SPLIT_TEMPLATES)}, got {split!r}")
    return [(TEMPLATES[t][kind].format(slot=slot), " " + answer)
            for t in SPLIT_TEMPLATES[split] for kind, slot, answer in FACTS]


def encode_example(tokenizer, prompt: str, answer: str, max_len: int = 48):
    """-> (input_ids, labels), both 1-D int64 of the same length n - 1.

    full = prompt ids + answer ids + EOT. input_ids = full[:-1], labels = full[1:]
    with every position whose target is still a prompt token set to -100.
    Encoding prompt and answer separately keeps the boundary exact (no BPE merge
    across it); the answer starts with a space, as in pretraining text.
    """
    p = tokenizer.encode(prompt)
    a = tokenizer.encode(answer) + [tokenizer.eot_id]
    full = p + a
    if len(full) - 1 > max_len:
        raise ValueError(f"example has {len(full) - 1} positions > max_len={max_len}: {prompt!r}")
    ids = torch.tensor(full, dtype=torch.long)
    x, y = ids[:-1].clone(), ids[1:].clone()
    y[: len(p) - 1] = IGNORE_INDEX            # targets 1..len(p)-1 are prompt tokens
    return x, y


def collate(batch, pad_id: int):
    """Right-pad a list of (input_ids, labels) to the longest row: x with pad_id, y with -100."""
    T = max(x.numel() for x, _ in batch)
    xs = torch.full((len(batch), T), pad_id, dtype=torch.long)
    ys = torch.full((len(batch), T), IGNORE_INDEX, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        xs[i, : x.numel()] = x
        ys[i, : y.numel()] = y
    return xs, ys


# ------------------------------------------------------------------ greedy evaluation

def generation_backend() -> str:
    try:
        from llm.generate import generate  # noqa: F401  (module 35)
        return "llm.generate"
    except ImportError:
        return "local-fallback"


@torch.no_grad()
def _greedy_local(model, idx, max_new_tokens: int, eos_id: int):
    """LOCAL FALLBACK used only when llm/generate.py (module 35) is absent:
    uncached greedy decoding, stops once every row has produced eos_id."""
    done = torch.zeros(idx.size(0), dtype=torch.bool)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.config.block_size:])
        nxt = logits[:, -1].argmax(dim=-1)                       # (B,)
        nxt = torch.where(done, torch.full_like(nxt, eos_id), nxt)
        idx = torch.cat([idx, nxt[:, None]], dim=1)
        done |= nxt == eos_id
        if bool(done.all()):
            break
    return idx


def _greedy_padded(model, idx, mask, max_new_tokens: int, eos_id: int):
    """One left-padded batch through module 35's generate (per-row key mask and positions)."""
    from llm.generate import generate
    out = generate(model, idx, max_new_tokens, temperature=0.0, eos_id=eos_id, attention_mask=mask)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def greedy_answers(model, tokenizer, prompts, max_new_tokens: int = 24) -> list[str]:
    """Greedy continuation of each prompt up to the first EOT, decoded.

    With llm.generate (module 35): one batch, prompts left-padded with EOT and an
    attention mask. Fallback: prompts grouped by token length so each batch is
    rectangular and needs no padding. Restores the model's train/eval mode.
    """
    was_training = model.training
    model.eval()
    eot = tokenizer.eot_id
    try:
        encoded = [tokenizer.encode(p) for p in prompts]
        if generation_backend() == "llm.generate":
            T = max(len(e) for e in encoded)
            idx = torch.full((len(encoded), T), eot, dtype=torch.long)
            mask = torch.zeros(len(encoded), T, dtype=torch.long)
            for i, e in enumerate(encoded):
                idx[i, T - len(e):] = torch.tensor(e)
                mask[i, T - len(e):] = 1
            batches = [(list(range(len(encoded))), T, _greedy_padded(model, idx, mask, max_new_tokens, eot))]
        else:
            groups = defaultdict(list)
            for i, e in enumerate(encoded):
                groups[len(e)].append(i)
            batches = [(rows, T, _greedy_local(model, torch.tensor([encoded[i] for i in rows]),
                                               max_new_tokens, eot))
                       for T, rows in groups.items()]
        answers = [None] * len(prompts)
        for rows, T, out in batches:
            for i, new in zip(rows, out[:, T:].tolist()):
                answers[i] = tokenizer.decode(new[: new.index(eot)] if eot in new else new)
        return answers
    finally:
        model.train(was_training)


def answer_exact_match(model, tokenizer, examples, max_new_tokens: int = 24) -> float:
    """Fraction of (prompt, answer) pairs whose greedy answer equals the reference (whitespace-stripped)."""
    got = greedy_answers(model, tokenizer, [p for p, _ in examples], max_new_tokens)
    return sum(g.strip() == a.strip() for g, (_, a) in zip(got, examples)) / len(examples)
