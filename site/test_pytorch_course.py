"""Executable checks for the PyTorch from Tensors to LLMs course.

Every ```python pytorch-check``` fence in content/courses/pytorch/NN-*.md is an
independent, self-contained CPU program that asserts its own worked numbers.
Each fence runs in a fresh namespace with stdout captured, torch seeded to 0,
outbound sockets refused, and a 10 s ceiling. A fence must leave torch's
global state (default dtype, thread count, grad mode, deterministic
algorithms, quantized engine) as it found it.

Counts: every module file that exists needs at least MIN_PER_MODULE fences;
the course-wide minimum MIN_TOTAL applies once all EXPECTED_MODULES files exist.
"""

import contextlib
import io
import os
from pathlib import Path
import re
import socket
import sys
import time
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "content/courses/pytorch"
FENCE = re.compile(r"^```python pytorch-check\n(.*?)^```\s*$", re.M | re.S)
EXPECTED_MODULES = 38
MIN_PER_MODULE = 2
MIN_TOTAL = 100
MAX_SECONDS = 10.0


class _NoNetworkSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise OSError("pytorch-check fences must not use the network")


def global_state():
    return {
        "default dtype": torch.get_default_dtype(),
        "num threads": torch.get_num_threads(),
        "grad mode": torch.is_grad_enabled(),
        "deterministic algorithms": torch.are_deterministic_algorithms_enabled(),
        "quantized engine": torch.backends.quantized.engine,
    }


def restore(state):
    torch.set_default_dtype(state["default dtype"])
    torch.set_num_threads(state["num threads"])
    torch.set_grad_enabled(state["grad mode"])
    torch.use_deterministic_algorithms(state["deterministic algorithms"])
    torch.backends.quantized.engine = state["quantized engine"]


def modules():
    return sorted(p for p in COURSE.glob("[0-9][0-9]-*.md") if not p.name.startswith("00-"))


def collect(path):
    """Return (first line number, code) per fence; line numbers match the .md file."""
    text = path.read_text(encoding="utf-8")
    return [(text.count("\n", 0, m.start(1)) + 1, m.group(1)) for m in FENCE.finditer(text)]


class PublishedPyTorchChecks(unittest.TestCase):
    def test_pytorch_check_fences(self):
        rows = []
        for path in modules():
            fences = collect(path)
            started = time.perf_counter()
            for index, (line, code) in enumerate(fences, 1):
                with self.subTest(page=path.name, fence=index, line=line):
                    before = global_state()
                    torch.manual_seed(0)
                    t0 = time.perf_counter()
                    try:
                        with mock.patch.object(socket, "socket", _NoNetworkSocket), \
                                contextlib.redirect_stdout(io.StringIO()):
                            # Leading newlines make traceback line numbers point into the .md file.
                            exec(compile("\n" * (line - 1) + code, str(path), "exec"), {"__name__": "__main__"})
                    finally:
                        elapsed = time.perf_counter() - t0
                        after = global_state()
                        restore(before)
                    leaked = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
                    self.assertEqual(leaked, {}, f"fence at line {line} did not restore global state")
                    self.assertLessEqual(elapsed, MAX_SECONDS,
                                         f"fence at line {line} took {elapsed:.2f} s (> {MAX_SECONDS:.0f} s)")
            rows.append((path.name, len(fences), time.perf_counter() - started))

        width = max([len(name) for name, _, _ in rows] + [6])
        print(f"\n{'module':<{width}}  fences  seconds", file=sys.stderr)
        for name, count, seconds in rows:
            print(f"{name:<{width}}  {count:>6}  {seconds:>7.2f}", file=sys.stderr)
        total = sum(count for _, count, _ in rows)
        print(f"{'total':<{width}}  {total:>6}  {sum(s for _, _, s in rows):>7.2f}", file=sys.stderr)

        sparse = [f"{name}: {count}" for name, count, _ in rows if count < MIN_PER_MODULE]
        self.assertEqual(sparse, [], f"each module needs at least {MIN_PER_MODULE} pytorch-check fences")
        if len(rows) < EXPECTED_MODULES:
            print(f"WARNING: {len(rows)}/{EXPECTED_MODULES} module files present; "
                  f"the {MIN_TOTAL}-fence course minimum is not enforced yet", file=sys.stderr)
        else:
            self.assertEqual(len(rows), EXPECTED_MODULES, "unexpected extra module files")
            self.assertGreaterEqual(total, MIN_TOTAL, f"expected at least {MIN_TOTAL} pytorch-check fences")


if __name__ == "__main__":
    unittest.main(verbosity=2)
