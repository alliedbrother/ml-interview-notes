"""Build the capstone corpus and train/verify its byte-level BPE tokenizer (module 30).

Run from the repository root:
    python content/courses/pytorch/code/30_build_tokenizer.py check
    python content/courses/pytorch/code/30_build_tokenizer.py train --out /tmp/bpe.json
    # one-time, offline after a manual download; never run in CI:
    python content/courses/pytorch/code/30_build_tokenizer.py prepare --source /tmp/pg1661.txt

Subcommands:
  prepare  strip the Project Gutenberg header/footer from pg1661.txt, convert CRLF
           to LF, verify hashes, write data/sherlock.txt and data/sherlock_provenance.json
  train    train the 512-id BPE on the train split and write the artifact
  check    (default) retrain in memory, assert the merges equal the committed
           artifact, verify every sha256, and report compression statistics
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from llm.corpus import (CORPUS_PATH, CORPUS_SHA256, DATA_DIR, EOT, STORY_RE,  # noqa: E402
                        SPLIT_NUMERALS, join_with_eot, load_corpus, load_splits,
                        sha256_bytes, sha256_file, split_stories)
from llm.tokenizer import PATTERN, ByteBPE  # noqa: E402

SOURCE_URL = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
SOURCE_SHA256 = "922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0"
SOURCE_BYTES = 607606
ARTIFACT_PATH = DATA_DIR / "bpe_sherlock_512.json"
PROVENANCE_PATH = DATA_DIR / "sherlock_provenance.json"
START_MARK = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"


def strip_gutenberg(raw: bytes) -> str:
    """Keep the lines strictly between the START and END marker lines, LF endings."""
    text = raw.decode("utf-8")
    if "\r" in text.replace("\r\n", ""):
        raise ValueError("source contains a bare carriage return")
    lines = text.replace("\r\n", "\n").split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith(START_MARK))
    end = next(i for i, line in enumerate(lines) if line.startswith(END_MARK))
    if not start < end:
        raise ValueError("START marker must precede END marker")
    return "\n".join(lines[start + 1:end]) + "\n"


def prepare(source, out, force=False):
    raw = Path(source).read_bytes()
    source_sha = sha256_bytes(raw)
    if source_sha != SOURCE_SHA256:
        print(f"warning: {source} sha256 {source_sha} differs from the 2026-10-10 retrieval "
              f"{SOURCE_SHA256}; Project Gutenberg has changed the file", file=sys.stderr)
        if not force:
            raise SystemExit("refusing to overwrite the corpus without --force")
    text = strip_gutenberg(raw)
    data = text.encode("utf-8")
    if sha256_bytes(data) != CORPUS_SHA256 and not force:
        raise SystemExit(f"stripped sha256 {sha256_bytes(data)} != {CORPUS_SHA256}")
    out = Path(out)
    out.write_bytes(data)
    provenance = {
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "gutenberg_ebook": 1661,
        "source_url": SOURCE_URL,
        "landing_page": "https://www.gutenberg.org/ebooks/1661",
        "retrieved": "2026-10-10",
        "source_bytes": len(raw),
        "source_sha256": source_sha,
        "stripping_rule": (
            "Decode UTF-8, convert CRLF to LF, keep only the lines strictly between the line "
            f"starting '{START_MARK}' and the line starting '{END_MARK}', join with LF and "
            "append one final LF. Equivalent: sed -n '28,11959p' pg1661.txt | tr -d '\\r'"),
        "line_endings": "LF",
        "output_bytes": len(data),
        "output_chars": len(text),
        "output_sha256": sha256_bytes(data),
        "license": ("Public domain in the USA; author died 1930; Project Gutenberg "
                    "header/footer and trademark removed"),
        "split": {k: list(v) for k, v in SPLIT_NUMERALS.items()},
        "heading_regex": STORY_RE.pattern,
    }
    prov_path = out.parent / PROVENANCE_PATH.name
    prov_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    return {"command": "prepare", "corpus": str(out), "provenance": str(prov_path),
            "output_bytes": len(data), "output_chars": len(text),
            "output_sha256": sha256_bytes(data), "source_sha256": source_sha}


def split_texts(text):
    return {name: join_with_eot(stories) for name, stories in load_splits(text).items()}


def tokenizer_stats(tok, texts):
    """Compression and id-usage statistics; torch is used for the id histogram."""
    import torch

    n_tokens, bpt, unused = {}, {}, {}
    for name, txt in texts.items():
        ids = tok.encode(txt, allowed_special={EOT})
        if tok.decode(ids) != txt:
            raise AssertionError(f"round trip failed on {name}")
        counts = torch.bincount(torch.tensor(ids, dtype=torch.int64), minlength=tok.vocab_size)
        n_tokens[name] = len(ids)
        bpt[name] = round(len(txt.encode("utf-8")) / len(ids), 4)
        unused[name] = int((counts == 0).sum())
    longest = sorted(range(256, tok.eot_id), key=lambda i: (-len(tok.token_bytes(i)), i))[:20]
    return {
        "n_tokens": n_tokens,
        "bytes_per_token": bpt,
        "unused_ids": unused,
        "longest_tokens": [tok.token_bytes(i).decode("utf-8", errors="replace") for i in longest],
    }


def compare_hf(train_text, val_text, vocab_size):
    try:
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError:
        return {"skipped": "tokenizers is not installed"}
    hf = Tokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    t0 = time.perf_counter()
    hf.train_from_iterator(train_text.split(EOT), trainer=trainer)
    seconds = time.perf_counter() - t0
    out = {"vocab_size": hf.get_vocab_size(), "train_seconds": round(seconds, 3)}
    for name, txt in (("train", train_text), ("val", val_text)):
        n = sum(len(hf.encode(part).ids) for part in txt.split(EOT)) + txt.count(EOT)
        out[f"bytes_per_token_{name}"] = round(len(txt.encode("utf-8")) / n, 4)
    return out


def train_and_report(vocab_size, want_hf):
    text = load_corpus(verify=True)
    texts = split_texts(text)
    t0 = time.perf_counter()
    tok, trace = ByteBPE.train_with_trace(texts["train"], vocab_size)
    seconds = time.perf_counter() - t0
    result = {
        "merges": len(tok.merges),
        "vocab_size": tok.vocab_size,
        "eot_id": tok.eot_id,
        "train_seconds": round(seconds, 3),
        "unique_train_chunks": None,
        "first_merges": [
            {"pair": [tok.token_bytes(a).decode("utf-8", "replace"),
                      tok.token_bytes(b).decode("utf-8", "replace")],
             "ids": list(e["pair"]), "count": e["count"], "new_id": e["new_id"]}
            for e, (a, b) in zip(trace[:10], tok.merges[:10])],
        "corpus_sha256": sha256_file(CORPUS_PATH),
        "train_sha256": tok.train_sha256,
        "split_bytes": {k: len(v.encode("utf-8")) for k, v in texts.items()},
        "story_bytes": {n: len(s.encode("utf-8")) for n, s in split_stories(text)},
    }
    result["unique_train_chunks"] = len({c for seg in texts["train"].split(EOT)
                                         for c in re.findall(PATTERN, seg)})
    t0 = time.perf_counter()
    result.update(tokenizer_stats(tok, texts))
    result["encode_seconds"] = round(time.perf_counter() - t0, 3)
    if want_hf:
        result["hf_comparison"] = compare_hf(texts["train"], texts["val"], vocab_size)
    return tok, result


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", nargs="?", default="check", choices=["prepare", "train", "check"])
    parser.add_argument("--source", help="path to the downloaded pg1661.txt (prepare)")
    parser.add_argument("--out", help="output path (prepare: corpus; train: artifact)")
    parser.add_argument("--force", action="store_true", help="prepare: accept changed hashes")
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--json-out", help="also write the result dict to this path")
    parser.add_argument("--compare-hf", action="store_true",
                        help="compare with a Hugging Face tokenizers ByteLevel BPE")
    parser.add_argument("--seed", type=int, default=0,
                        help="accepted for CLI uniformity; BPE training is deterministic")
    args = parser.parse_args(argv)

    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    if args.command == "prepare":
        if not args.source:
            parser.error("prepare needs --source")
        result = prepare(args.source, args.out or CORPUS_PATH, force=args.force)
    elif args.command == "train":
        tok, result = train_and_report(args.vocab_size, args.compare_hf)
        out = Path(args.out) if args.out else ARTIFACT_PATH
        tok.save(out)
        result.update(command="train", artifact=str(out), artifact_sha256=sha256_file(out))
    else:
        tok, result = train_and_report(args.vocab_size, args.compare_hf)
        committed = ByteBPE.load(ARTIFACT_PATH)
        if committed.merges != tok.merges:
            first = next(i for i, (a, b) in enumerate(zip(committed.merges, tok.merges)) if a != b)
            raise AssertionError(f"retrained merges differ from the artifact at rank {first}")
        if committed.train_sha256 != tok.train_sha256:
            raise AssertionError("artifact train_sha256 does not match the current train split")
        prov = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
        if prov["output_sha256"] != CORPUS_SHA256 or result["corpus_sha256"] != CORPUS_SHA256:
            raise AssertionError("corpus sha256 disagrees with provenance or llm.corpus")
        result.update(command="check", merges_match=True,
                      artifact_sha256=sha256_file(ARTIFACT_PATH))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, ensure_ascii=False))
