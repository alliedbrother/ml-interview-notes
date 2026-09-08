"""One source of truth for executable lesson fences and their downloads."""

from dataclasses import dataclass

from markdown_it import MarkdownIt


@dataclass(frozen=True)
class Example:
    number: int
    code: str
    extras: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return f"example-{self.number:02d}.py"

    def download(self, source: str) -> str:
        dependency = " plus XGBoost/CatBoost" if "boosters" in self.extras else ""
        return (f"# Source: {source}\n"
                f"# Independent CPU example; use the curriculum environment{dependency}.\n"
                "# See /notes/ml/#example-environment or /notes/deep-learning/#example-environment.\n\n"
                + self.code)


def example_from_fence(info: str, code: str, number: int) -> Example | None:
    fields = info.lower().split()
    if fields[:2] != ["python", "runnable"]:
        return None
    extras = tuple(fields[2:])
    if any(extra != "boosters" for extra in extras) or len(set(extras)) != len(extras):
        raise ValueError(f"Unknown or duplicate runnable-example options: {info}")
    return Example(number, code.rstrip() + "\n", extras)


def collect_examples(markdown: str) -> list[Example]:
    examples = []
    for token in MarkdownIt().parse(markdown):
        if token.type == "fence":
            example = example_from_fence(token.info, token.content, len(examples) + 1)
            if example is not None:
                examples.append(example)
    return examples
