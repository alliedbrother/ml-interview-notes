"""Scaling contracts for PyTorch course module 37 (llm/scale.py, 37_scale_recipe.py).

In process (``--small`` config): activation checkpointing keeps gradients and
cuts saved bytes, counted FLOPs match the formula, aot_eager compile matches
eager with zero graph breaks, and the GPT-2 budget arithmetic is self-consistent.
The DDP and FSDP2 runs spawn processes, so they run as two subprocesses of the
lab (syllabus D.4); both start in setUpModule and run while the in-process
checks do. Never runs Inductor, never retrains the committed checkpoint.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
SCRIPT = CODE / "37_scale_recipe.py"
LAB_NAME = "pytorch_37_scale_recipe"

lab = scale = None
_STATE = {}
_PROCS = {}


def setUpModule():
    global lab, scale
    _STATE.update(threads=torch.get_num_threads(), dtype=torch.get_default_dtype(),
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  rng=torch.get_rng_state(), path=list(sys.path))
    _STATE["tmp"] = tempfile.TemporaryDirectory()
    for command in ("ddp", "fsdp"):
        out = Path(_STATE["tmp"].name) / f"{command}.json"
        proc = subprocess.Popen(
            [sys.executable, str(SCRIPT), command, "--small", "--json-out", str(out)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        _PROCS[command] = (proc, out)
    spec = importlib.util.spec_from_file_location(LAB_NAME, SCRIPT)
    lab = importlib.util.module_from_spec(spec)
    sys.modules[LAB_NAME] = lab  # Dynamo may re-import the module that defines compiled code
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* resolves
    from llm import scale as scale_mod
    scale = scale_mod


def tearDownModule():
    for proc, _ in _PROCS.values():
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    _STATE["tmp"].cleanup()
    torch._dynamo.reset()
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    torch.set_rng_state(_STATE["rng"])
    sys.modules.pop(LAB_NAME, None)
    sys.path[:] = _STATE["path"]


def small_model_and_batch(seed=1337):
    from llm.model import GPT, GPTConfig
    cfg = GPTConfig(**lab.SMALL)
    torch.manual_seed(seed)
    return GPT(cfg), scale.make_batch(cfg, 8, 64, seed)


def spawned_result(command):
    proc, out = _PROCS[command]
    stdout, stderr = proc.communicate(timeout=120)
    if proc.returncode != 0:
        raise AssertionError(f"{command} failed:\n{stderr[-3000:]}")
    result = json.loads(out.read_text())
    assert json.loads(stdout) == result
    return result[command]


class InProcess(unittest.TestCase):
    def test_checkpointing_keeps_grads_and_cuts_saved_bytes(self):
        r = lab.main(["checkpoint", "--small"])["checkpoint"]
        self.assertLessEqual(r["max_grad_diff"], 1e-6)
        self.assertLess(r["bytes_ckpt"], 0.6 * r["bytes_plain"])
        self.assertTrue(r["state_dict_keys_equal"])
        # recompute adds at most one forward of the blocks (a third of their fwd+bwd)
        self.assertGreater(r["flops_ckpt"], r["flops_plain"])
        self.assertLessEqual(r["flops_ckpt"], r["flops_ckpt_no_early_stop"])
        self.assertLess(r["flops_ckpt_no_early_stop"], 4 / 3 * r["flops_plain"] + 1)

    def test_wrapped_model_loads_committed_checkpoint(self):
        from llm.model import GPT
        from llm.train import load_pretrained
        plain, _, _ = load_pretrained()
        wrapped = scale.enable_activation_checkpointing(GPT(plain.config))
        wrapped.load_state_dict(plain.state_dict(), strict=True)
        self.assertIs(wrapped.lm_head.weight, wrapped.wte.weight)
        x = torch.randint(0, 512, (2, 32), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            torch.testing.assert_close(wrapped(x)[0], plain(x)[0], rtol=0, atol=0)

    def test_flops_match_formula(self):
        r = lab.main(["flops", "--small"])["flops"]
        self.assertGreaterEqual(r["ratio"], 0.8)
        self.assertLessEqual(r["ratio"], 1.25)
        # the CPU flash SDPA op has no FlopCounterMode formula in 2.8: attention goes missing
        self.assertLessEqual(r["measured_with_sdpa"], r["measured"])

    def test_aot_eager_compile_matches_eager(self):
        model, batch = small_model_and_batch()
        r = scale.compile_parity(model, batch, backend="aot_eager")
        self.assertLessEqual(r["max_loss_diff"], 1e-5)
        self.assertLessEqual(r["max_grad_diff"], 1e-5)
        self.assertEqual(r["graph_breaks"], 0)

    def test_profile_sees_matmuls(self):
        model, batch = small_model_and_batch()
        r = scale.profile_step(model, batch, steps=2, warmup=1)
        names = {op["name"] for op in r["top_ops"]}
        self.assertTrue(names & {"aten::mm", "aten::addmm", "aten::bmm"}, names)
        self.assertEqual(set(r["phase_ms_per_step"]), {"forward", "backward", "optimizer"})

    def test_gpt2_budget_is_self_consistent(self):
        b = scale.gpt2_budget()
        self.assertLess(abs(b["params"] - 124.4e6) / 124.4e6, 0.01)
        L, T, d = 12, 1024, 768
        expected = (6 * b["params"] + 12 * L * T * d) * b["tokens"]
        self.assertLess(abs(b["total_flops"] - expected) / expected, 0.01)
        self.assertEqual(b["train_state_bytes"], 16 * b["params"])
        self.assertAlmostEqual(b["hours"] * 3600 * 8 * 312e12 * 0.4, b["total_flops"], delta=1e9)


class Distributed(unittest.TestCase):
    def test_ddp_matches_single_process(self):
        r = spawned_result("ddp")
        self.assertEqual(r["world_size"], 2)
        self.assertLess(r["max_grad_diff_vs_ref"], 1e-5)
        self.assertLess(r["max_grad_diff_no_sync_accum"], 1e-5)
        self.assertLess(r["max_param_diff_vs_ref"], 1e-5)
        self.assertTrue(r["ranks_equal"])
        self.assertTrue(r["tied_after_wrap"])

    def test_fsdp_matches_single_process_and_shards(self):
        r = spawned_result("fsdp")
        self.assertTrue(r["all_dtensor"])
        self.assertEqual(r["placements"], ["(Shard(dim=0),)"])
        self.assertTrue(r["tied_after_shard"])
        self.assertEqual(r["full_bytes"], r["plain_model_bytes"])
        self.assertLessEqual(max(r["shard_bytes_per_rank"]), 0.6 * r["full_bytes"])
        self.assertLess(r["max_grad_diff_vs_ref"], 1e-5)
        self.assertLess(r["max_param_diff_vs_ref"], 1e-5)
        self.assertLess(abs(r["grad_ratio_force_sum"] - 1.0), 1e-4)
        # 2.0 on torch 2.8 (gloo reduce_scatter ignores AVG); 1.0 once fixed upstream
        ratio = r["grad_ratio_default_avg"]
        self.assertTrue(abs(ratio - 2.0) < 1e-4 or abs(ratio - 1.0) < 1e-4, ratio)


if __name__ == "__main__":
    unittest.main()
