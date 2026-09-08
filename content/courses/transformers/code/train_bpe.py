"""Inspectable raw-byte BPE training, not a production tokenizer implementation.

Run: python content/courses/transformers/code/train_bpe.py
Only the standard library is required. No normalization or special tokens.
"""

import argparse
from collections import Counter
import json
from pathlib import Path


def replace_pair(tokens, pair, new_id):
    result, i = [], 0
    while i < len(tokens):
        if tuple(tokens[i:i + 2]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def train_library_tokenizer(texts):
    """Production-library counterpart; its pretokenization differs from raw bytes."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts, trainer=trainers.BpeTrainer(
        vocab_size=280, min_frequency=2, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return tokenizer


class ByteBPE:
    def __init__(self, merges=()):
        self.merges = [tuple(pair) for pair in merges]
        self.vocab = [bytes([i]) for i in range(256)]
        for left, right in self.merges:
            if not (type(left) is int and type(right) is int
                    and 0 <= left < len(self.vocab) and 0 <= right < len(self.vocab)):
                raise ValueError("Merge operands must reference earlier token IDs")
            self.vocab.append(self.vocab[left] + self.vocab[right])

    @classmethod
    def train(cls, texts, num_merges=12, min_frequency=2):
        if type(num_merges) is not int or num_merges < 0:
            raise ValueError("num_merges must be a nonnegative integer")
        if type(min_frequency) is not int or min_frequency < 1:
            raise ValueError("min_frequency must be a positive integer")
        sequences = [list(text.encode("utf-8")) for text in texts]
        merges, trace = [], []
        for rank in range(num_merges):
            # Count each adjacent window, including overlapping occurrences.
            counts = Counter(pair for seq in sequences for pair in zip(seq, seq[1:]))
            if not counts:
                break
            pair = min(counts, key=lambda candidate: (-counts[candidate], candidate))
            if counts[pair] < min_frequency:
                break
            before = sum(map(len, sequences))
            new_id = 256 + rank
            sequences = [replace_pair(seq, pair, new_id) for seq in sequences]
            merges.append(pair)
            trace.append({"rank": rank, "pair": list(pair), "new_id": new_id,
                          "frequency": counts[pair], "tokens_before": before,
                          "tokens_after": sum(map(len, sequences)),
                          "pair_counts": [{"pair": list(p), "count": n}
                                          for p, n in sorted(counts.items())]})
        return cls(merges), trace

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        for rank, pair in enumerate(self.merges):
            tokens = replace_pair(tokens, pair, 256 + rank)
        return tokens

    def decode(self, ids):
        if any(type(i) is not int or not 0 <= i < len(self.vocab) for i in ids):
            raise ValueError("Unknown token ID")
        # Decode after concatenation: one token need not be a complete UTF-8 unit.
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="strict")

    def save(self, path):
        Path(path).write_text(json.dumps({"format": "course-raw-byte-bpe-v1",
                                        "merges": self.merges}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if data.get("format") != "course-raw-byte-bpe-v1":
            raise ValueError("Unsupported tokenizer format")
        return cls(data["merges"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--library", action="store_true", help="Also train Hugging Face Tokenizers BPE")
    args = parser.parse_args()
    corpus = ["low", "low", "low", "lower", "lowest", "newer", "newest"]
    tokenizer, trace = ByteBPE.train(corpus)
    for event in trace:
        left, right = event["pair"]
        print(f"rank={event['rank']:2d} pair={tokenizer.vocab[left]!r}+"
              f"{tokenizer.vocab[right]!r} count={event['frequency']} "
              f"tokens={event['tokens_before']}->{event['tokens_after']}")
    for text in ("lower", "slower", "  low\n", "caf\u00e9 \U0001f30d", "<BOS>", ""):
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text
        print(repr(text), "->", ids)
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save(args.output_dir / "tokenizer.json")
        (args.output_dir / "merge-trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        restored = ByteBPE.load(args.output_dir / "tokenizer.json")
        assert restored.encode("slower") == tokenizer.encode("slower")
    if args.library:
        library = train_library_tokenizer(corpus)
        for text in ("slower", "  low\n", "caf\u00e9 \U0001f30d", "<BOS>", ""):
            assert library.decode(library.encode(text).ids) == text
        if args.output_dir:
            library.save(str(args.output_dir / "tokenizer-hf.json"))
        print("Hugging Face byte-level BPE round trips verified; merge IDs may differ.")
    print("Raw-byte round trips verified; IDs are not compatible with pretrained checkpoints.")


if __name__ == "__main__":
    main()
