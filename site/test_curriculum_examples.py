"""Validate and independently execute trusted ML/DL lesson examples.

Run --check-only in the lightweight build environment. Full execution requires
requirements-examples.txt; --include-boosters also requires requirements-boosters.txt.
Only repository-authored, explicitly marked fences are executed, never user input.
"""

import argparse
import ast
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from curriculum_examples import collect_examples, example_from_fence


ROOT = Path(__file__).resolve().parent.parent
NOTES = ROOT / "content/notes"
REQUIRED_ALGORITHMS = {
    "linear-models": {"LinearRegression", "LogisticRegression"},
    "trees-and-ensembles": {"DecisionTreeClassifier", "RandomForestClassifier",
                            "AdaBoostClassifier", "XGBClassifier", "CatBoostClassifier"},
    "probabilistic-and-instance-models": {"KNeighborsClassifier"},
    "unsupervised-learning": {"KMeans", "AgglomerativeClustering"},
}


def has_assertion(tree: ast.AST) -> bool:
    assertion_calls = {"torch.testing.assert_close", "np.testing.assert_allclose",
                       "numpy.testing.assert_allclose", "np.testing.assert_array_equal",
                       "numpy.testing.assert_array_equal", "assert_allclose", "assert_array_equal"}
    return any(isinstance(node, ast.Assert) or
               (isinstance(node, ast.Call) and ast.unparse(node.func) in assertion_calls)
               for node in ast.walk(tree))


def called_names(tree: ast.AST) -> set[str]:
    return {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
            for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Name, ast.Attribute))}


def algorithm_coverage_errors(calls_by_topic):
    return [f"ml/{topic}: missing required runnable estimator calls: {', '.join(sorted(missing))}"
            for topic, required in REQUIRED_ALGORITHMS.items()
            if (missing := required - calls_by_topic.get(topic, set()))]


def verify_published_html(document, examples, page_url):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(document, "html.parser")
    main = soup.find("main")
    if main is None:
        raise ValueError("Published page has no main content")
    blocks = []
    for block in main.select(".code-file"):
        header = block.select_one(".code-file__hd")
        label = header.get_text(" ", strip=True) if header else ""
        # Keep a labeled block even if its download control was accidentally lost.
        if label.startswith("Python / CPU example") or block.select_one("a.code-download"):
            blocks.append(block)
    if len(blocks) != len(examples):
        raise ValueError(f"Published runnable-block count {len(blocks)} != source count {len(examples)}")
    for example, block in zip(examples, blocks):
        links = block.select("a.code-download")
        expected_href = f"{page_url}examples/{example.filename}"
        if len(links) != 1 or links[0].get("href") != expected_href or not links[0].has_attr("download"):
            raise ValueError(f"{example.filename}: missing, duplicated or out-of-order download link; "
                             f"expected {expected_href}")
        code = block.select("pre code")
        if len(code) != 1 or code[0].get_text().rstrip("\n") + "\n" != example.code.rstrip("\n") + "\n":
            raise ValueError(f"{example.filename}: displayed code differs from source")


class FenceTests(unittest.TestCase):
    def test_only_explicit_examples(self):
        self.assertEqual(collect_examples("```python\nundefined_fragment()\n```"), [])
        self.assertIsNone(example_from_fence("python runnable-ish", "pass", 1))

    def test_numbering_and_optional_dependencies(self):
        examples = collect_examples("```python runnable\nassert True\n```\n"
                                    "```text\nignore me\n```\n"
                                    "```python runnable boosters\nassert 1 == 1\n```")
        self.assertEqual([e.filename for e in examples], ["example-01.py", "example-02.py"])
        self.assertEqual(examples[1].extras, ("boosters",))
        self.assertTrue(examples[0].download("test.md").endswith("assert True\n"))

    def test_bad_options_fail(self):
        for info in ("python runnable typo", "python runnable boosters boosters"):
            with self.assertRaises(ValueError):
                example_from_fence(info, "pass", 1)

    def test_numerical_assertions_count(self):
        self.assertTrue(has_assertion(ast.parse('torch.testing.assert_close(actual, expected)')))
        self.assertTrue(has_assertion(ast.parse('np.testing.assert_allclose(actual, expected)')))
        self.assertTrue(has_assertion(ast.parse('assert result > 0')))
        self.assertFalse(has_assertion(ast.parse('print("assert True")')))


class AlgorithmCoverageTests(unittest.TestCase):
    def test_imports_and_strings_do_not_count(self):
        tree = ast.parse('from sklearn.linear_model import LinearRegression\nname = "KMeans"')
        self.assertEqual(called_names(tree), set())
        self.assertEqual(called_names(ast.parse('model = LinearRegression()\nxgb.XGBClassifier()')),
                         {"LinearRegression", "XGBClassifier"})

    def test_required_calls_are_page_specific(self):
        calls = {topic: set(required) for topic, required in REQUIRED_ALGORITHMS.items()}
        self.assertEqual(algorithm_coverage_errors(calls), [])
        calls["unsupervised-learning"].remove("KMeans")
        calls["linear-models"].add("KMeans")
        errors = algorithm_coverage_errors(calls)
        self.assertEqual(len(errors), 1)
        self.assertIn("unsupervised-learning", errors[0])
        self.assertIn("KMeans", errors[0])

    def test_booster_calls_are_required(self):
        calls = {topic: set(required) for topic, required in REQUIRED_ALGORITHMS.items()}
        calls["trees-and-ensembles"].remove("CatBoostClassifier")
        self.assertIn("CatBoostClassifier", algorithm_coverage_errors(calls)[0])


class PublishedHtmlTests(unittest.TestCase):
    def setUp(self):
        self.examples = collect_examples("```python runnable\nassert 1 < 2\n```\n"
                                         "```python runnable boosters\nassert True\n```")
        self.url = "/notes/ml/example-topic/"
        blocks = []
        for example in self.examples:
            blocks.append(
                '<div class="code-file"><div class="code-file__hd">Python / CPU example'
                f'<a class="code-download" href="{self.url}examples/{example.filename}" download>'
                'Download</a></div><pre><code>' + html.escape(example.code) + '</code></pre></div>')
        ordinary = '<div class="code-file"><div class="code-file__hd">python</div><pre><code>fragment()</code></pre></div>'
        self.document = "<main>" + ordinary + "".join(blocks) + "</main>"

    def test_correct_html_and_trailing_newlines(self):
        verify_published_html(self.document, self.examples, self.url)
        verify_published_html(self.document.replace("assert True\n", "assert True\n\n"),
                              self.examples, self.url)

    def test_swapped_downloads_fail(self):
        swapped = (self.document.replace("example-01.py", "temporary.py")
                   .replace("example-02.py", "example-01.py")
                   .replace("temporary.py", "example-02.py"))
        with self.assertRaisesRegex(ValueError, "download link"):
            verify_published_html(swapped, self.examples, self.url)

    def test_changed_displayed_code_fails(self):
        with self.assertRaisesRegex(ValueError, "displayed code"):
            verify_published_html(self.document.replace("assert True", "assert False"),
                                  self.examples, self.url)

    def test_missing_download_fails(self):
        missing = self.document.replace(
            f'<a class="code-download" href="{self.url}examples/example-01.py" download>Download</a>', "")
        with self.assertRaisesRegex(ValueError, "download link"):
            verify_published_html(missing, self.examples, self.url)

    def test_missing_download_attribute_fails(self):
        with self.assertRaisesRegex(ValueError, "download link"):
            verify_published_html(self.document.replace(' download>', '>'), self.examples, self.url)

    def test_missing_runnable_block_fails(self):
        with self.assertRaisesRegex(ValueError, "block count"):
            verify_published_html("<main></main>", self.examples, self.url)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--published", action="store_true", help="Check built downloads against source")
    parser.add_argument("--include-boosters", action="store_true")
    parser.add_argument("--topic", help="Limit execution/coverage to one topic slug")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--report", type=Path, help="Write a JSON execution record")
    args = parser.parse_args()
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(FenceTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(AlgorithmCoverageTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(PublishedHtmlTests),
    ])
    if not unittest.TextTestRunner().run(suite).wasSuccessful():
        return 1

    records, errors, page_count = [], [], 0
    calls_by_topic = {}
    start = time.monotonic()
    for category in ("ml", "deep-learning"):
        for path in sorted((NOTES / category).glob("*.md")):
            if args.topic and path.stem != args.topic:
                continue
            page_count += 1
            examples = collect_examples(path.read_text(encoding="utf-8"))
            relative = path.relative_to(ROOT).as_posix()
            if not examples:
                errors.append(f"{relative}: no independent runnable example")
            if args.published:
                published_page = ROOT / "_site/notes" / category / path.stem / "index.html"
                try:
                    verify_published_html(published_page.read_text(encoding="utf-8"), examples,
                                          f"/notes/{category}/{path.stem}/")
                except (OSError, ValueError) as exc:
                    errors.append(f"{relative}: {exc}")
            for example in examples:
                name = f"{category}/{path.stem}/{example.filename}"
                record = {"name": name, "extras": example.extras, "status": "checked"}
                records.append(record)
                try:
                    tree = ast.parse(example.code, filename=name)
                    if category == "ml":
                        calls_by_topic.setdefault(path.stem, set()).update(called_names(tree))
                    if not has_assertion(tree):
                        raise ValueError("Runnable examples need a checkable assertion")
                    if args.published:
                        artifact = ROOT / "_site/notes" / category / path.stem / "examples" / example.filename
                        if artifact.read_text(encoding="utf-8") != example.download(relative):
                            raise ValueError("Published download differs from displayed source")
                    if args.check_only:
                        continue
                    if example.extras and not args.include_boosters:
                        record["status"] = "skipped-optional"
                        print(f"SKIP {name}: add --include-boosters", flush=True)
                        continue
                    # Fresh processes and disposable working directories prevent hidden
                    # notebook state or inter-example variables from making tests pass.
                    with tempfile.TemporaryDirectory(prefix="ml-example-") as cwd:
                        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                   MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                                   MPLBACKEND="Agg", MPLCONFIGDIR=cwd, PYTHONHASHSEED="0",
                                   PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
                        tick = time.monotonic()
                        result = subprocess.run([sys.executable, "-c", example.code], cwd=cwd,
                                                env=env, text=True, capture_output=True,
                                                timeout=args.timeout)
                        record.update(seconds=round(time.monotonic() - tick, 3),
                                      stdout=result.stdout, stderr=result.stderr)
                        if result.returncode:
                            raise RuntimeError(result.stderr or result.stdout)
                    record["status"] = "passed"
                    print(f"PASS {name} ({record['seconds']}s)", flush=True)
                except (OSError, SyntaxError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
                    record.update(status="failed", error=str(exc))
                    errors.append(f"{name}: {exc}")

    if page_count == 0:
        errors.append("No matching topic pages")
    if not args.topic:
        errors.extend(algorithm_coverage_errors(calls_by_topic))
    versions = {}
    for package in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "gymnasium",
                    "stable-baselines3", "xgboost", "catboost"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    summary = {"pages": page_count, "examples": records, "versions": versions,
               "python": sys.version, "seconds": round(time.monotonic() - start, 2),
               "errors": errors, "mode": "static" if args.check_only else "execute"}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    statuses = {status: sum(r["status"] == status for r in records)
                for status in sorted({r["status"] for r in records})}
    print(f"OK: {page_count} topics, {len(records)} examples, {statuses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
