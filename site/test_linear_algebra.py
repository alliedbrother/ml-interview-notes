"""Execute the chapter's trusted NumPy examples. Optional dependency: NumPy."""

from pathlib import Path
import re

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "content/notes/math/linear-algebra.md"


def main():
    article = SOURCE.read_text(encoding="utf-8")
    tokens = MarkdownIt().parse(article)
    examples = [token for token in tokens if token.type == "fence" and token.info == "python"]
    assert len(examples) >= 8, "Expected the chapter's executable worked examples"
    for number, example in enumerate(examples, 1):
        # Only run repository-authored examples, never reader-provided input.
        exec(compile(example.content, f"{SOURCE}:example-{number}", "exec"), {})
        print(f"OK: NumPy example {number}")
    headings = [tokens[i + 1].content for i, token in enumerate(tokens) if token.type == "heading_open" and token.tag == "h2"]
    for topic in ("vector spaces", "rank", "projections", "eigenvalues", "quadratic forms",
                  "singular value", "principal component", "least squares", "pseudoinverses",
                  "conditioning", "positive-definite", "matrix calculus", "kernels"):
        assert any(topic in title.lower() for title in headings), f"Missing topic: {topic}"
    assert not re.search(r"\[\[[A-Z_]+\]\]", article), "Unresolved draft marker"
    print(f"OK: {len(examples)} examples and {len(headings)} major sections")


if __name__ == "__main__":
    main()
