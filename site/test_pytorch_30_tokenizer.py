"""Corpus provenance, story splits and byte-level BPE contracts for PyTorch course module 30."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "content/courses/pytorch/code"
CORPUS_SHA256 = "3552d466d95a92fb58e96bbfabbfc02370d359ac95933b5feafe4ebaf3f243b3"

lab = corpus = tokenizer = None
RESULT = TOK = SPLITS = None
_STATE = {}


def setUpModule():
    global lab, corpus, tokenizer, RESULT, TOK, SPLITS
    _STATE["threads"] = torch.get_num_threads()
    _STATE["dtype"] = torch.get_default_dtype()
    _STATE["deterministic"] = torch.are_deterministic_algorithms_enabled()
    _STATE["path"] = list(sys.path)
    spec = importlib.util.spec_from_file_location("pytorch_30_build_tokenizer",
                                                  CODE / "30_build_tokenizer.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)  # adds CODE to sys.path so llm.* imports resolve
    import llm.corpus as corpus_mod
    import llm.tokenizer as tokenizer_mod
    corpus, tokenizer = corpus_mod, tokenizer_mod
    # The one full retrain of this test file: `check` retrains on the train split
    # and raises unless the merges equal the committed artifact exactly.
    RESULT = lab.main(["check"])
    TOK = tokenizer.ByteBPE.load(CODE / "data/bpe_sherlock_512.json")
    SPLITS = {k: corpus.join_with_eot(v) for k, v in corpus.load_splits().items()}


def tearDownModule():
    torch.set_num_threads(_STATE["threads"])
    torch.set_default_dtype(_STATE["dtype"])
    torch.use_deterministic_algorithms(_STATE["deterministic"])
    sys.path[:] = _STATE["path"]


class CorpusContracts(unittest.TestCase):
    def test_committed_corpus_hash_and_provenance(self):
        data = (CODE / "data/sherlock.txt").read_bytes()
        self.assertEqual(corpus.sha256_bytes(data), CORPUS_SHA256)
        self.assertEqual(corpus.CORPUS_SHA256, CORPUS_SHA256)
        self.assertNotIn(b"\r", data)
        self.assertNotIn(b"PROJECT GUTENBERG", data.upper())
        prov = json.loads((CODE / "data/sherlock_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(prov["output_sha256"], CORPUS_SHA256)
        self.assertEqual(prov["output_bytes"], len(data))
        self.assertEqual(prov["output_chars"], len(data.decode("utf-8")))
        self.assertEqual(prov["gutenberg_ebook"], 1661)
        self.assertEqual(prov["heading_regex"], corpus.STORY_RE.pattern)
        self.assertEqual(RESULT["corpus_sha256"], CORPUS_SHA256)

    def test_twelve_stories_in_order_and_split_titles(self):
        stories = corpus.split_stories(corpus.load_corpus())
        self.assertEqual([n for n, _ in stories],
                         ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"])
        self.assertIn("ENGINEER’S THUMB", stories[8][1].splitlines()[0])
        splits = corpus.load_splits()
        self.assertEqual([len(splits[k]) for k in ("train", "val", "test")], [10, 1, 1])
        self.assertIn("BERYL CORONET", splits["val"][0].splitlines()[0])
        self.assertIn("COPPER BEECHES", splits["test"][0].splitlines()[0])
        # Stories partition the text after the front matter: nothing is lost or duplicated.
        text = corpus.load_corpus()
        self.assertTrue(text.endswith("".join(s for _, s in stories)))
        self.assertEqual(SPLITS["train"].count(corpus.EOT), 9)

    def test_strip_rule_on_synthetic_gutenberg_file(self):
        raw = ("header\r\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\r\n\r\nbody “1”\r\n"
               "*** END OF THE PROJECT GUTENBERG EBOOK X ***\r\nfooter\r\n").encode()
        self.assertEqual(lab.strip_gutenberg(raw), "\nbody “1”\n")
        with self.assertRaises(ValueError):
            lab.strip_gutenberg(b"a\rb")


class TokenizerContracts(unittest.TestCase):
    def test_retrain_matches_artifact_and_schema(self):
        self.assertTrue(RESULT["merges_match"])
        self.assertEqual(RESULT["merges"], 255)
        self.assertEqual((TOK.vocab_size, TOK.eot_id, len(TOK.merges)), (512, 511, 255))
        art = json.loads((CODE / "data/bpe_sherlock_512.json").read_text(encoding="utf-8"))
        self.assertEqual(art["version"], 1)
        self.assertEqual(art["pattern"], tokenizer.PATTERN)
        self.assertEqual(art["special_tokens"], {"<|endoftext|>": 511})
        self.assertEqual(art["train_sha256"],
                         corpus.sha256_bytes(SPLITS["train"].encode("utf-8")))

    def test_round_trips_on_splits_and_adversarial_strings(self):
        for name, text in SPLITS.items():
            ids = TOK.encode(text, allowed_special={"<|endoftext|>"})
            self.assertEqual(TOK.decode(ids), text, name)
            self.assertTrue(all(0 <= i < 512 for i in ids))
        for s in ["", " ", "\x00", "é", "́", "🕵️‍♂️ £5 — café",
                  "  trailing\n\n", "Holmes's 1892\tpipe", "<|endoftext|>"]:
            ids = TOK.encode(s)
            self.assertEqual(TOK.decode(ids), s, repr(s))
            self.assertNotIn(511, ids)

    def test_special_token_only_with_permission(self):
        self.assertNotIn(511, TOK.encode("a<|endoftext|>b"))
        self.assertEqual(TOK.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}), [511])
        ids = TOK.encode("a<|endoftext|>b", allowed_special={"<|endoftext|>"})
        self.assertEqual(ids.count(511), 1)
        self.assertEqual(TOK.decode(ids), "a<|endoftext|>b")
        with self.assertRaises(ValueError):
            TOK.encode("x", allowed_special={"<|pad|>"})

    def test_merges_never_produce_eot_and_ids_are_well_formed(self):
        for rank, (a, b) in enumerate(TOK.merges):
            self.assertLess(a, 256 + rank)
            self.assertLess(b, 256 + rank)
        self.assertEqual(TOK.token_bytes(511), b"<|endoftext|>")
        produced = {256 + r for r in range(len(TOK.merges))}
        self.assertNotIn(511, produced)

    def test_compression_quality_bars(self):
        bpt = RESULT["bytes_per_token"]
        self.assertGreaterEqual(bpt["val"], 1.9)
        self.assertLessEqual(bpt["val"], 2.5)
        self.assertGreaterEqual(bpt["train"], bpt["val"] - 0.05)
        self.assertGreater(RESULT["n_tokens"]["train"], 8 * RESULT["n_tokens"]["val"])

    def test_determinism_and_overlap_rule(self):
        story = corpus.load_splits()["train"][2]
        first = tokenizer.ByteBPE.train(story, vocab_size=320)
        second = tokenizer.ByteBPE.train(story, vocab_size=320)
        self.assertEqual(first.merges, second.merges)
        self.assertEqual(tokenizer.merge_pair([97, 97, 97, 97], (97, 97), 256), [256, 256])
        self.assertEqual(tokenizer.merge_pair([97, 97, 97], (97, 97), 256), [256, 97])
        tiny, trace = tokenizer.ByteBPE.train_with_trace("aaaa", vocab_size=258)
        self.assertEqual(trace[0]["count"], 3)
        self.assertEqual(tiny.encode("aaaa"), [256, 256])

    def test_save_load_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bpe.json"
            TOK.save(path)
            again = tokenizer.ByteBPE.load(path)
            self.assertEqual(again, TOK)
            self.assertEqual(again.train_sha256, TOK.train_sha256)
            bad = json.loads(path.read_text(encoding="utf-8"))
            bad["merges"][0] = [600, 1]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                tokenizer.ByteBPE.load(path)


if __name__ == "__main__":
    unittest.main()
