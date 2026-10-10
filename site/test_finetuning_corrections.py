"""CPU numerical contracts for the LLM Fine-Tuning course.

Every ```python finetuning-check``` fence in content/courses/finetuning/*.md is
an independent, self-contained program that asserts its own worked numbers.
Each runs in a fresh namespace, on CPU, with outbound network connections
refused, so a check can never depend on another check or on a download.
"""

import contextlib
import io
import os
from pathlib import Path
import re
import socket
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch


# Raise this as the course gains checks; it guards against fences being lost.
MIN_CHECKS = 10

ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / "content/courses/finetuning"
FENCE = re.compile(r"^```python finetuning-check\n(.*?)^```\s*$", re.M | re.S)
torch.set_num_threads(1)


def _no_network(*args, **kwargs):
    raise OSError("finetuning-check fences must not use the network")


def collect():
    return [(path, code) for path in sorted(COURSE.glob("*.md"))
            for code in FENCE.findall(path.read_text(encoding="utf-8"))]


class PublishedFinetuningChecks(unittest.TestCase):
    def test_independent_numerical_fences(self):
        checks = collect()
        for number, (path, code) in enumerate(checks, 1):
            with self.subTest(page=path.name, check=number):
                torch.manual_seed(0)
                with mock.patch.object(socket.socket, "connect", _no_network), \
                        mock.patch.object(socket, "create_connection", _no_network), \
                        contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, str(path), "exec"), {"__name__": "__main__"})
        per_page = {}
        for path, _ in checks:
            per_page[path.name] = per_page.get(path.name, 0) + 1
        found = ", ".join(f"{name}: {n}" for name, n in per_page.items()) or "none"
        self.assertGreaterEqual(
            len(checks), MIN_CHECKS,
            f"expected at least {MIN_CHECKS} ```python finetuning-check fences in "
            f"{COURSE.relative_to(ROOT)}/*.md, found {len(checks)} "
            f"({len(list(COURSE.glob('*.md')))} Markdown files; per page: {found})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
