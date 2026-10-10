"""Corpus loading and story-level splits for the capstone (module 30).

The corpus is *The Adventures of Sherlock Holmes* (Project Gutenberg #1661) with
the Gutenberg header and footer stripped and CRLF converted to LF. Provenance is
recorded in ``data/sherlock_provenance.json``. Standard library only.
"""

import hashlib
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CORPUS_PATH = DATA_DIR / "sherlock.txt"
CORPUS_SHA256 = "3552d466d95a92fb58e96bbfabbfc02370d359ac95933b5feafe4ebaf3f243b3"

EOT = "<|endoftext|>"
NUMERALS = ("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII")
# Story headings are upper-case lines such as "IX. THE ADVENTURE OF THE ENGINEER’S THUMB"
# (curly apostrophe). The indented table of contents does not match because of "^".
STORY_RE = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)\. [A-Z][A-Z’' -]+$", re.M)
SPLIT_NUMERALS = {
    "train": NUMERALS[:10],
    "val": ("XI",),
    "test": ("XII",),
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def load_corpus(verify: bool = True) -> str:
    """Return the committed corpus as text; with ``verify`` check its sha256 first."""
    raw = CORPUS_PATH.read_bytes()
    if verify and sha256_bytes(raw) != CORPUS_SHA256:
        raise ValueError(
            f"{CORPUS_PATH} sha256 {sha256_bytes(raw)} != expected {CORPUS_SHA256}")
    return raw.decode("utf-8")


def split_stories(text: str) -> list[tuple[str, str]]:
    """Return ``[(numeral, story_text), ...]`` for the 12 stories in order.

    Each story runs from its heading line up to the next heading (or end of
    text) and includes the heading. Front matter before story I is dropped.
    """
    matches = list(STORY_RE.finditer(text))
    numerals = tuple(m.group(1) for m in matches)
    if numerals != NUMERALS:
        raise ValueError(f"expected story headings {NUMERALS}, found {numerals}")
    ends = [m.start() for m in matches[1:]] + [len(text)]
    return [(m.group(1), text[m.start():end]) for m, end in zip(matches, ends)]


def load_splits(text: str | None = None) -> dict[str, list[str]]:
    """Return ``{"train": [...10 stories], "val": [XI], "test": [XII]}``."""
    if text is None:
        text = load_corpus()
    stories = dict(split_stories(text))
    return {name: [stories[n] for n in nums] for name, nums in SPLIT_NUMERALS.items()}


def join_with_eot(stories) -> str:
    """Join stories with the end-of-text marker between (not after) them."""
    return EOT.join(stories)
