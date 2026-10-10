"""Deterministic byte-level BPE tokenizer for the capstone (module 30).

Ids 0-255 are raw bytes, ids 256.. are merges in training order, and the last
id is ``<|endoftext|>``. With the default 512-id vocabulary that is 255 merges
(ids 256-510) and EOT = 511. Pre-tokenization uses an ASCII GPT-2-like pattern
with the standard-library ``re`` module, so merges never cross chunk boundaries.
Standard library only; not a production tokenizer (no normalization, English-
oriented pattern).
"""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]{1,3}| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
EOT = "<|endoftext|>"
FORMAT_VERSION = 1


def merge_pair(ids, pair, new_id):
    """Replace non-overlapping occurrences of ``pair`` left to right."""
    out, i, n = [], 0, len(ids)
    a, b = pair
    while i < n:
        if i + 1 < n and ids[i] == a and ids[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class ByteBPE:
    """Byte-level BPE with one special token, ``<|endoftext|>``, as the last id."""

    def __init__(self, merges, pattern: str = PATTERN, train_sha256: str | None = None):
        self._merges = [(int(a), int(b)) for a, b in merges]
        self.pattern = pattern
        self._re = re.compile(pattern)
        self.train_sha256 = train_sha256
        self._vocab = [bytes([i]) for i in range(256)]
        self._ranks = {}
        for rank, (a, b) in enumerate(self._merges):
            new_id = 256 + rank
            if not (0 <= a < new_id and 0 <= b < new_id):
                raise ValueError(f"merge {rank} {(a, b)} references an id not yet defined")
            if (a, b) in self._ranks:
                raise ValueError(f"duplicate merge {(a, b)}")
            self._ranks[(a, b)] = new_id
            self._vocab.append(self._vocab[a] + self._vocab[b])
        self.special_tokens = {EOT: 256 + len(self._merges)}
        self._cache = {}

    # ------------------------------------------------------------------ props
    @property
    def merges(self) -> list[tuple[int, int]]:
        return list(self._merges)

    @property
    def eot_id(self) -> int:
        return self.special_tokens[EOT]

    @property
    def vocab_size(self) -> int:
        return 256 + len(self._merges) + len(self.special_tokens)

    def token_bytes(self, token_id: int) -> bytes:
        if token_id == self.eot_id:
            return EOT.encode("utf-8")
        if not 0 <= token_id < len(self._vocab):
            raise ValueError(f"token id {token_id} out of range [0, {self.vocab_size})")
        return self._vocab[token_id]

    def __eq__(self, other):
        if not isinstance(other, ByteBPE):
            return NotImplemented
        return (self._merges, self.pattern, self.special_tokens) == (
            other._merges, other.pattern, other.special_tokens)

    def __repr__(self):
        return f"ByteBPE(vocab_size={self.vocab_size}, merges={len(self._merges)})"

    # --------------------------------------------------------------- training
    @classmethod
    def train(cls, text: str, vocab_size: int = 512) -> "ByteBPE":
        """Train ``vocab_size - 257`` merges on ``text`` (EOT markers are split out)."""
        return cls.train_with_trace(text, vocab_size)[0]

    @classmethod
    def train_with_trace(cls, text: str, vocab_size: int = 512):
        """Return ``(tokenizer, trace)``; trace[k] = {"pair", "count", "new_id"}.

        Pre-token frequencies are counted once; each merge step counts adjacent
        pairs over the unique-chunk table weighted by chunk frequency, picks the
        pair with the highest count (ties: smallest ``(a, b)``), and rewrites
        every chunk. Training stops early if no adjacent pair is left.
        """
        if type(vocab_size) is not int or vocab_size < 257:
            raise ValueError("vocab_size must be an int >= 257 (256 bytes + EOT)")
        num_merges = vocab_size - 257
        pat = re.compile(PATTERN)
        freq = Counter()
        for segment in text.split(EOT):
            freq.update(pat.findall(segment))
        # Distinct strings stay distinct after merges, so the table never needs re-aggregation.
        table = [(list(chunk.encode("utf-8")), count) for chunk, count in freq.items()]
        merges, trace = [], []
        for rank in range(num_merges):
            pairs = Counter()
            for ids, count in table:
                for pair in zip(ids, ids[1:]):
                    pairs[pair] += count
            if not pairs:
                break
            best = min(pairs, key=lambda p: (-pairs[p], p))
            new_id = 256 + rank
            a = best[0]
            # `a in ids` is a C-speed scan that skips chunks the merge cannot touch.
            table = [(merge_pair(ids, best, new_id) if a in ids else ids, count)
                     for ids, count in table]
            merges.append(best)
            trace.append({"pair": best, "count": pairs[best], "new_id": new_id})
        sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(merges, PATTERN, train_sha256=sha), trace

    # --------------------------------------------------------------- encoding
    def _encode_chunk(self, chunk: str) -> list[int]:
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        ranks = self._ranks
        while len(ids) > 1:
            # Lowest new id = earliest merge; applying it everywhere, then the next
            # earliest, reproduces training's merge order exactly.
            best = min(zip(ids, ids[1:]), key=lambda p: ranks.get(p, 1 << 30))
            if best not in ranks:
                break
            ids = merge_pair(ids, best, ranks[best])
        self._cache[chunk] = ids
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        out = []
        for chunk in self._re.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text: str, allowed_special=frozenset()) -> list[int]:
        """Encode text to ids. ``<|endoftext|>`` becomes ``eot_id`` only if allowed;
        otherwise the literal string is encoded as ordinary bytes."""
        allowed = set(allowed_special)
        unknown = allowed - set(self.special_tokens)
        if unknown:
            raise ValueError(f"unknown special tokens: {sorted(unknown)}")
        if EOT not in allowed:
            return self._encode_ordinary(text)
        out = []
        for i, segment in enumerate(text.split(EOT)):
            if i:
                out.append(self.eot_id)
            out.extend(self._encode_ordinary(segment))
        return out

    def decode(self, ids) -> str:
        """Concatenate token bytes, then UTF-8 decode once (errors="replace")."""
        return b"".join(self.token_bytes(int(i)) for i in ids).decode("utf-8", errors="replace")

    # ---------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        return {
            "version": FORMAT_VERSION,
            "pattern": self.pattern,
            "vocab_size": self.vocab_size,
            "merges": [list(p) for p in self._merges],
            "special_tokens": dict(self.special_tokens),
            "train_sha256": self.train_sha256,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        d = self.to_dict()
        merges = ",\n  ".join(json.dumps(m) for m in d.pop("merges"))
        head = json.dumps(d, indent=1, ensure_ascii=False)[:-2]  # drop the closing "\n}"
        tmp.write_text(f'{head},\n "merges": [\n  {merges}\n ]\n}}\n', encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path) -> "ByteBPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer format version {d.get('version')}")
        if d["pattern"] != PATTERN:
            raise ValueError("tokenizer pattern differs from this code's PATTERN")
        tok = cls(d["merges"], d["pattern"], train_sha256=d.get("train_sha256"))
        if tok.vocab_size != d["vocab_size"] or tok.special_tokens != d["special_tokens"]:
            raise ValueError("vocab_size/special_tokens inconsistent with merges")
        return tok
