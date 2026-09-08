"""Run the math chapters' standalone CPU examples and correction regressions."""

from pathlib import Path
import re
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
MATH = ROOT / "content/notes/math"


class MathCorrectionsTests(unittest.TestCase):
    def test_each_chapter_has_independently_runnable_examples(self):
        for path in sorted(MATH.glob("*.md")):
            blocks = re.findall(r"^```python runnable\n(.*?)^```", path.read_text(), re.M | re.S)
            with self.subTest(chapter=path.name):
                self.assertTrue(blocks)
            for index, code in enumerate(blocks, 1):
                with self.subTest(chapter=path.name, block=index):
                    result = subprocess.run(
                        [sys.executable, "-c", code], cwd=ROOT,
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_syntax_local_links_and_display_delimiters(self):
        for path in [ROOT / "content/notes/math.md", *sorted(MATH.glob("*.md"))]:
            text = path.read_text()
            with self.subTest(chapter=path.name):
                self.assertNotIn("\\`\\`\\`", text)
                for code in re.findall(r"^```python(?: runnable)?\n(.*?)^```", text, re.M | re.S):
                    compile(code, str(path), "exec")
                for target in re.findall(r"\]\(([^)]+)\)", text):
                    if target.startswith(("./", "../")):
                        self.assertTrue((path.parent / target.split("#")[0]).resolve().exists(), target)
                self.assertFalse(re.search(r"(?m)^\$$", text), path.name)

    def test_core_corrected_claims(self):
        algebra = (MATH / "linear-algebra.md").read_text()
        self.assertIn("\n+s\\begin{bmatrix}-2", algebra)
        self.assertIn("\n+t\\begin{bmatrix}-3", algebra)
        self.assertIn(r"\operatorname{tr}(A^* A)", algebra)
        self.assertIn("**91.8%** (918 of 1000)", (MATH / "statistics.md").read_text())
        numerics = (MATH / "numerical-methods.md").read_text()
        self.assertIn("smallest_subnormal", numerics)
        self.assertNotIn('os.environ["PYTHONHASHSEED"] =', numerics)

    def test_small_algebra_probability_and_graph_regressions(self):
        import numpy as np
        from scipy.sparse.csgraph import laplacian

        A = np.array([[1., 2., 3.], [2., 4., 6.]])
        for s, t in [(0., 0.), (2., -3.), (-1., 4.)]:
            x = np.array([4., 0., 0.]) + s*np.array([-2., 1., 0.])
            x += t*np.array([-3., 0., 1.])
            np.testing.assert_allclose(A @ x, [4., 8.])
        complex_A = np.array([[1j, 2.], [1., -2j]])
        self.assertAlmostEqual(
            np.sum(np.abs(complex_A)**2),
            np.trace(complex_A.conj().T @ complex_A).real,
        )
        self.assertAlmostEqual(918/1000, .918)
        self.assertAlmostEqual(.1/np.sqrt(.001), np.sqrt(10))
        self.assertAlmostEqual(np.sqrt(32/512), .25)
        values = np.array([-1., 0., 1.])
        self.assertAlmostEqual(np.mean(values**3),
                               np.mean(values)*np.mean(values**2))
        graph = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]])
        L = laplacian(graph)
        normalized = laplacian(graph, normed=True)
        self.assertTrue(np.all(np.linalg.eigvalsh(L) >= -1e-14))
        self.assertEqual(np.count_nonzero(np.abs(np.linalg.eigvalsh(L)) < 1e-12), 2)
        np.testing.assert_allclose(normalized[2], 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
