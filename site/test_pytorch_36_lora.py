"""LoRA and SFT contracts for PyTorch course module 36 (llm/lora.py, llm/sft.py, 36_lora_finetune.py).

Loads the committed checkpoint through llm.train.load_pretrained (never
pretrains). Runs the lab once: LoRA r=8 for 150 steps and a 60-step full
fine-tune, both with seed 1337. Asserts contracts and relative quality bars,
not the exact numbers quoted in the module.
"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"

lab = lora = sft = None
RESULT = None
TMP = None
_STATE = {}


def setUpModule():
    global lab, lora, sft, RESULT, TMP
    _STATE.update(threads=torch.get_num_threads(), dtype=torch.get_default_dtype(),
                  deterministic=torch.are_deterministic_algorithms_enabled(),
                  rng=torch.get_rng_state(), path=list(sys.path))
    spec = importlib.util.spec_from_file_location("pytorch_36_lora_finetune", CODE / "36_lora_finetune.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* imports resolve
    import llm.lora as lora_mod
    import llm.sft as sft_mod
    lora, sft = lora_mod, sft_mod
    TMP = tempfile.TemporaryDirectory()
    RESULT = lab.main(["--method", "both", "--steps", "150", "--full-steps", "60", "--seed", "1337",
                       "--output-dir", str(Path(TMP.name) / "run"),
                       "--json-out", str(Path(TMP.name) / "result.json")])


def tearDownModule():
    TMP.cleanup()
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    torch.set_rng_state(_STATE["rng"])
    sys.path[:] = _STATE["path"]


def pretrained():
    from llm.train import load_pretrained
    return load_pretrained()


class Surgery(unittest.TestCase):
    def test_apply_lora_replaces_twelve_modules_and_counts_45056(self):
        model, _, _ = pretrained()
        names = lora.apply_lora(model, ("c_attn", "c_proj"), r=8, alpha=16)
        self.assertEqual(len(names), 12)
        self.assertEqual(sorted({n.split(".", 2)[2] for n in names}),
                         ["attn.c_attn", "attn.c_proj", "mlp.c_proj"])
        self.assertTrue(all(isinstance(model.get_submodule(n), lora.LoRALinear) for n in names))
        lora.mark_only_lora_trainable(model)
        per_layer = sum(8 * (m.in_features + m.out_features) for _, m in lora.lora_modules(model))
        self.assertEqual(per_layer, 4 * 8 * (512 + 256 + 640))
        self.assertEqual(lora.count_trainable(model), 45_056)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(len(trainable), 24)
        self.assertTrue(all(n.endswith(("lora_A", "lora_B")) for n in trainable))
        self.assertIn("blocks.0.attn.c_attn.base.weight", model.state_dict())
        self.assertIs(model.lm_head.weight, model.wte.weight)  # tying survives surgery

    def test_output_equals_base_exactly_at_init(self):
        model, tok, _ = pretrained()
        x = torch.tensor([tok.encode("Q: Who is the Scotland Yard inspector?\nA: Lestrade")])
        with torch.no_grad():
            ref = model(x)[0]
            lora.apply_lora(model, r=8, alpha=16)
            got = model(x)[0]
        self.assertTrue(torch.equal(got, ref))
        self.assertEqual(RESULT["lora"]["init_max_diff"], 0.0)

    def test_merge_is_exact_in_float64_and_unwrap_restores_keys(self):
        torch.manual_seed(0)
        model, tok, _ = pretrained()
        keys = list(model.state_dict())
        lora.apply_lora(model, r=4, alpha=8)
        for _, m in lora.lora_modules(model):
            torch.nn.init.normal_(m.lora_B, std=0.05)
        model = model.double()
        x = torch.tensor([tok.encode("Q: Which story is number IX?\nA:")])
        with torch.no_grad():
            unmerged = model(x)[0]
            lora.merge_all(model)
            self.assertTrue(all(m.merged for _, m in lora.lora_modules(model)))
            lora.merge_all(model)  # idempotent
            merged = model(x)[0]
            torch.testing.assert_close(merged, unmerged, rtol=0, atol=1e-10)
            lora.unmerge_all(model)
            lora.merge_all(model)
            self.assertEqual(len(lora.unwrap_lora(model)), 12)
            torch.testing.assert_close(model(x)[0], unmerged, rtol=0, atol=1e-10)
        self.assertEqual(list(model.state_dict()), keys)

    def test_load_lora_state_dict_rejects_mismatched_adapters(self):
        model, _, _ = pretrained()
        lora.apply_lora(model, r=8, alpha=16)
        sd = lora.lora_state_dict(model)
        self.assertEqual(len(sd), 24)
        self.assertEqual(sum(v.numel() for v in sd.values()), 45_056)
        with self.assertRaises(KeyError):
            lora.load_lora_state_dict(model, {**sd, "blocks.9.attn.c_attn.lora_A": sd[next(iter(sd))]})
        partial = dict(list(sd.items())[2:])
        with self.assertRaises(KeyError):
            lora.load_lora_state_dict(model, partial)


class SFTData(unittest.TestCase):
    def test_examples_and_templates(self):
        self.assertEqual(len(sft.FACTS), 30)
        train, held = sft.build_examples("train"), sft.build_examples("heldout")
        self.assertEqual((len(train), len(held)), (60, 30))
        self.assertFalse({p for p, _ in train} & {p for p, _ in held})
        with self.assertRaises(ValueError):
            sft.build_examples("test")

    def test_prompt_tokens_never_contribute_to_loss(self):
        _, tok, _ = pretrained()
        prompt, answer = "Q: Who is the Scotland Yard inspector?\nA:", " Lestrade"
        x, y = sft.encode_example(tok, prompt, answer, max_len=48)
        n_prompt, n_answer = len(tok.encode(prompt)), len(tok.encode(answer)) + 1  # + EOT
        self.assertEqual(x.numel(), n_prompt + n_answer - 1)
        self.assertTrue(torch.all(y[: n_prompt - 1] == -100))
        self.assertEqual(int((y != -100).sum()), n_answer)
        self.assertEqual(int(y[-1]), tok.eot_id)
        self.assertEqual(tok.decode(y[n_prompt - 1:-1].tolist()), answer)
        # perturbing logits at prompt positions leaves the loss unchanged
        torch.manual_seed(0)
        logits = torch.randn(x.numel(), tok.vocab_size)
        loss = F.cross_entropy(logits, y, ignore_index=-100)
        bumped = logits.clone()
        bumped[: n_prompt - 1] += 5 * torch.randn(n_prompt - 1, tok.vocab_size)
        self.assertEqual(F.cross_entropy(bumped, y, ignore_index=-100).item(), loss.item())
        # right padding: x padded with pad_id, y with -100
        x2, y2 = sft.encode_example(tok, "Q: Which story is number IX?\nA:",
                                    " The Adventure of the Engineer’s Thumb", 48)
        xb, yb = sft.collate([(x, y), (x2, y2)], tok.eot_id)
        self.assertEqual(xb.shape, (2, max(x.numel(), x2.numel())))
        self.assertTrue(torch.all(yb[0, x.numel():] == -100))
        self.assertTrue(torch.all(xb[0, x.numel():] == tok.eot_id))
        with self.assertRaises(ValueError):
            sft.encode_example(tok, prompt, answer, max_len=10)

    def test_greedy_backends_agree(self):
        if sft.generation_backend() != "llm.generate":
            self.skipTest("llm/generate.py absent: only the local fallback exists")
        model, tok, _ = pretrained()
        prompts = [p for p, _ in sft.build_examples("heldout")[::5]]
        batched = sft.greedy_answers(model, tok, prompts, max_new_tokens=8)
        original = sft.generation_backend
        try:
            sft.generation_backend = lambda: "local-fallback"
            grouped = sft.greedy_answers(model, tok, prompts, max_new_tokens=8)
        finally:
            sft.generation_backend = original
        self.assertEqual(batched, grouped)


class LabRun(unittest.TestCase):
    def test_lora_learns_the_task(self):
        base, lr = RESULT["base"], RESULT["lora"]
        self.assertEqual(lr["trainable"], 45_056)
        self.assertGreaterEqual(lr["em"], base["em"] + 0.3)
        self.assertGreaterEqual(lr["train_em"], 0.8)
        self.assertLess(lr["last_loss"], 0.5 * lr["first_loss"])

    def test_base_weights_untouched(self):
        self.assertEqual(RESULT["lora"]["base_max_change"], 0.0)

    def test_forgetting_lora_not_worse_than_full_at_default_lrs(self):
        base = RESULT["base"]["val_nll"]
        d_lora = RESULT["lora"]["val_nll"] - base
        d_full = RESULT["full"]["val_nll"] - base
        self.assertGreater(d_full, 0.0)
        self.assertLessEqual(d_lora, d_full + 0.02)

    def test_merge_reload_and_adapter_size(self):
        lr = RESULT["lora"]
        bound = 1e-5 + 1e-5 * lr["logit_absmax"]  # assert_close(rtol=1e-5, atol=1e-5) on fp32 logits
        self.assertLessEqual(lr["merge_max_diff"], bound)
        self.assertLessEqual(lr["reload_max_diff"], 1e-6)
        self.assertEqual(lr["reload_unexpected_keys"], 0)
        self.assertEqual(lr["reload_missing_keys"], 53)  # every base tensor, no adapter tensor
        self.assertEqual(lr["adapter_tensors"], 24)
        self.assertLess(lr["adapter_bytes"], 300 * 1024)

    def test_optimizer_state_accounting(self):
        self.assertGreaterEqual(RESULT["lora"]["opt_state_bytes"], 2 * 4 * 45_056)
        self.assertLess(RESULT["lora"]["opt_state_bytes"], 2 * 4 * 45_056 + 1024)
        self.assertGreaterEqual(RESULT["full"]["opt_state_bytes"], 2 * 4 * 875_264)
        self.assertEqual(RESULT["full"]["trainable"], 875_264)

    def test_output_dir_and_json(self):
        run = Path(TMP.name)
        self.assertEqual((run / "run" / "adapter.pt").stat().st_size, RESULT["lora"]["adapter_bytes"])
        saved = json.loads((run / "result.json").read_text())
        self.assertEqual(saved["lora"]["trainable"], 45_056)
        self.assertIn(saved["generation_backend"], ("llm.generate", "local-fallback"))


if __name__ == "__main__":
    unittest.main()
