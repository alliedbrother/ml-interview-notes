"""Execute the published CPU contracts; no engine checkout or GPU is required."""

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import unittest

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1] / "content/courses/inference/html"
PAGES = sorted(
    page
    for directory in ROOT.iterdir()
    if directory.name.startswith(("06-", "07-", "08-"))
    for page in directory.glob("*.html")
)


class PublishedInferenceContracts(unittest.TestCase):
    def test_complete_page_coverage(self):
        self.assertEqual(len(PAGES), 15)
        for page in PAGES:
            with self.subTest(page=page.name):
                soup = BeautifulSoup(page.read_text(), "html.parser")
                self.assertEqual(len(soup.select('[data-cpu-check="inference-06-08"]')), 1)
                self.assertEqual(len(soup.select('#reference-contract')), 1)
                self.assertFalse(soup.select(
                    "section section, pre p, pre div, pre section, p p, p section, p div"
                ))
                self.assertTrue(soup.select_one("#hands-on"))
                self.assertTrue(soup.select_one("#exercises"))

    def test_displayed_independent_references(self):
        for page in PAGES:
            with self.subTest(page=page.name):
                soup = BeautifulSoup(page.read_text(), "html.parser")
                block = soup.select_one('[data-cpu-check="inference-06-08"]')
                code = block.get_text()
                with redirect_stdout(StringIO()) as output:
                    filename = f"<inference-reference:{page.name}>"
                    exec(compile(code, filename, "exec"), {"__name__": "__main__"})
                self.assertTrue(output.getvalue().strip())


if __name__ == "__main__":
    unittest.main(verbosity=2)
