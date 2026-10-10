"""Contracts for the module 15 lab: scratch SGD/AdamW/schedule match torch.optim."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"


def load(name):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


lab = load("15_optimizers_from_scratch")
STEPS = 60
_STATE = {}
_RESULT = {}


def setUpModule():
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["det"] = torch.are_deterministic_algorithms_enabled()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "out.json"
        result = lab.main(["--seed", "0", "--steps", str(STEPS), "--json-out", str(path)])
        _RESULT.update(result)
        _RESULT["_json_roundtrip"] = json.loads(path.read_text()) == json.loads(json.dumps(result))


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["det"])


def train_steps(model, opt, x, y, n):
    for _ in range(n):
        opt.zero_grad(set_to_none=True)
        lab.mse(model, x, y).backward()
        opt.step()


class LabResult(unittest.TestCase):
    def test_contract_and_json(self):
        for key in ("sgd_max_param_diff", "adamw_max_param_diff", "foreach_max_diff",
                    "final_loss", "ema_eval_loss", "lr_trace"):
            self.assertIn(key, _RESULT)
        self.assertTrue(_RESULT["_json_roundtrip"])

    def test_scratch_matches_torch_in_float64(self):
        self.assertLess(_RESULT["sgd_max_param_diff"], 1e-12)
        self.assertLess(_RESULT["adamw_max_param_diff"], 1e-12)

    def test_foreach_matches_for_loop_within_tolerance(self):
        self.assertLess(_RESULT["foreach_max_diff"], 1e-5)
        self.assertLess(_RESULT["fused_cpu_max_diff"], 1e-5)

    def test_training_loss_halves(self):
        init = _RESULT["initial_loss"]
        for name, loss in _RESULT["final_loss"].items():
            self.assertLessEqual(loss, 0.5 * init, name)

    def test_ema_eval_loss_is_finite_and_comparable(self):
        self.assertTrue(0 < _RESULT["ema_eval_loss"] < _RESULT["initial_loss"])

    def test_lr_trace_matches_closed_form(self):
        warmup, peak = _RESULT["warmup"], _RESULT["peak_lr"]
        expected = [peak * lab.warmup_cosine(s, warmup=warmup, total=STEPS, floor=0.1)
                    for s in range(STEPS)]
        self.assertEqual(len(_RESULT["lr_trace"]), STEPS)
        for got, want in zip(_RESULT["lr_trace"], expected):
            self.assertAlmostEqual(got, want, delta=1e-15)
        self.assertAlmostEqual(_RESULT["lr_trace"][warmup - 1], peak, delta=1e-15)


class ScratchOptimizerContracts(unittest.TestCase):
    def setUp(self):
        self.x, self.y, _, _ = lab.make_data(1, torch.float64, n_train=32, n_eval=1)

    def check_round_trip(self, make_opt):
        model = lab.make_model(1, torch.float64)
        opt = make_opt(model)
        train_steps(model, opt, self.x, self.y, 3)
        clone = copy.deepcopy(model)
        fresh = make_opt(clone)
        fresh.load_state_dict(copy.deepcopy(opt.state_dict()))
        train_steps(model, opt, self.x, self.y, 1)
        train_steps(clone, fresh, self.x, self.y, 1)
        for a, b in zip(model.parameters(), clone.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_state_dict_round_trips(self):
        self.check_round_trip(lambda m: lab.ScratchSGD(m.parameters(), lr=0.05, momentum=0.9,
                                                       nesterov=True))
        self.check_round_trip(lambda m: lab.ScratchAdamW(lab.build_param_groups(m, 0.1), lr=1e-2))
        self.check_round_trip(lambda m: lab.SignMomentum(m.parameters(), lr=1e-3))

    def test_no_decay_group_excludes_biases_and_norm_gains(self):
        lr, wd = 0.1, 0.5
        for cls in (lab.ScratchAdamW, torch.optim.AdamW):
            model = lab.make_model(2, torch.float64)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt = cls(lab.build_param_groups(model, wd), lr=lr)
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            opt.step()
            for name, p in model.named_parameters():
                if p.ndim >= 2:
                    torch.testing.assert_close(p, before[name] * (1 - lr * wd), rtol=0, atol=1e-15)
                else:
                    self.assertTrue(torch.equal(p, before[name]), name)

    def test_tied_weights_listed_once(self):
        emb = nn.Embedding(11, 4)
        head = nn.Linear(4, 11, bias=False)
        head.weight = emb.weight
        model = nn.ModuleDict({"emb": emb, "head": head, "norm": nn.LayerNorm(4)})
        groups = lab.build_param_groups(model, 0.1)
        ids = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(p.numel() for g in groups for p in g["params"]), 11 * 4 + 4 + 4)
        torch.optim.AdamW(groups)  # would raise if a tensor sat in two groups


if __name__ == "__main__":
    unittest.main()
