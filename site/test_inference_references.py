"""Local reference, citation-parser and script-publication regressions."""

from pathlib import Path
import json
import tempfile
import subprocess
import sys
import re
import unittest

from bs4 import BeautifulSoup

from check_inference_citations import COURSE, PINS, check_range, inventory, inventory_digest, summarize
from test_build import is_local_markdown_link


ROOT = Path(__file__).resolve().parents[1]


class ReferenceTests(unittest.TestCase):
    def test_current_status_is_provenance_not_runtime_certification(self):
        lessons = sorted(COURSE.glob('[0-9][0-9]-*/*.html'))
        self.assertEqual(len(lessons), 70)
        for path in lessons:
            with self.subTest(page=path.name):
                soup = BeautifulSoup(path.read_text(), 'html.parser')
                self.assertNotIn('VERIFIED', soup.select_one('.fm').get_text())
                self.assertIn('SOURCE PINNED', soup.select_one('.fm').get_text())

    def test_display_delimiters_and_html_entities(self):
        for path in COURSE.rglob('*.html'):
            source = path.read_text()
            with self.subTest(page=str(path.relative_to(COURSE))):
                self.assertIsNone(re.search(r'(?<!&)nbsp;', source))
                main = BeautifulSoup(source, 'html.parser').find('main')
                for node in main.select('.math-display') if main else []:
                    formula = node.get_text().strip()
                    self.assertTrue(formula.startswith('$$') and formula.endswith('$$'), formula[:120])

    def test_external_markdown_references_are_not_local_routes(self):
        self.assertFalse(is_local_markdown_link('https://github.com/example/repo/blob/main/README.md'))
        self.assertFalse(is_local_markdown_link('//github.com/example/repo/README.md'))
        self.assertTrue(is_local_markdown_link('../chapter.md#section'))
        self.assertTrue(is_local_markdown_link('/notes/topic.md'))

    def test_front_door_parity(self):
        pages = [BeautifulSoup((COURSE / name).read_text(), 'html.parser').main
                 for name in ('index.html', 'README.html')]
        self.assertEqual(str(pages[0]), str(pages[1]))

    def test_lab_instructions_and_downloads(self):
        readmes = sorted((COURSE / 'labs').glob('[0-9][0-9]-*/README.html'))
        self.assertEqual(len(readmes), 13)
        for path in readmes:
            with self.subTest(lab=path.parent.name):
                main = BeautifulSoup(path.read_text(), 'html.parser').main
                self.assertIsNotNone(main.select_one('a[href="run.py"][download]'))
                self.assertIsNotNone(main.select_one('a[href="../cpu_checks.py"][download]'))
                self.assertGreaterEqual(len(main.select('section.sec')), 5)
                compile((path.parent / 'run.py').read_text(), str(path), 'exec')

    def test_every_lab_cli_starts_without_an_engine(self):
        for path in sorted((COURSE / 'labs').glob('[0-9][0-9]-*/run.py')):
            with self.subTest(lab=path.parent.name):
                result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True,
                                        text=True, timeout=15)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('usage:', result.stdout.lower())

    def test_glossary_chapters_are_links(self):
        main = BeautifulSoup((COURSE / 'GLOSSARY.html').read_text(), 'html.parser').main
        count = 0
        for row in main.select('tbody tr'):
            cells = row.find_all('td', recursive=False)
            if len(cells) < 3:
                continue
            for code in cells[-1].find_all('code'):
                if len(code.get_text()) == 5 and code.get_text()[2] == '-':
                    self.assertEqual(code.parent.name, 'a')
                    self.assertTrue((COURSE / code.parent['href']).is_file())
                    count += 1
        self.assertGreater(count, 150)

    def test_range_checker_and_inventory_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'sample.py').write_text('first\nsecond\n')
            record = dict(path='sample.py', first=1, last=2)
            self.assertEqual(check_range(record, root), 'range-exists')
            self.assertEqual(check_range(record | {'last': 3}, root), 'range-past-eof')
            self.assertEqual(check_range(record | {'first': 3}, root), 'invalid-range')
            self.assertEqual(check_range(record | {'path': '../outside.py'}, root), 'outside-checkout')
            (root / 'lesson.html').write_text('<main><code>vllm/test.py:L2-L4</code>'
                                              '<div class="code-file"><span class="src">csrc/test.cu:L1</span>'
                                              '<span class="tag">SGLang</span></div></main>')
            records = inventory(root)
            self.assertEqual([r['engine'] for r in records], ['vllm', 'sglang'])
            self.assertEqual(records[0]['last'], 4)

    def test_published_scripts_match_source(self):
        for source in (COURSE / 'labs').rglob('*.py'):
            with self.subTest(source=str(source)):
                published = ROOT / '_site/courses/inference' / source.relative_to(COURSE)
                self.assertTrue(published.is_file())
                self.assertEqual(published.read_bytes(), source.read_bytes())

    def test_citation_summary_preserves_unresolved_status(self):
        records = [dict(page='lesson.html', path='vllm/test.py', first=1, last=2,
                        engine='vllm', status='range-exists'),
                   dict(page='lesson.html', path='test.py', first=1, last=2,
                        engine=None, status='unresolved-engine')]
        report = summarize(records, {'vllm'})
        self.assertEqual(report['counts'], {'range-exists': 1, 'unresolved-engine': 1})
        self.assertEqual(report['pages']['lesson.html'], report['counts'])
        self.assertEqual(report['checked_engines'], ['vllm'])
        self.assertEqual(inventory_digest(records), inventory_digest([
            dict(record, status='checkout-not-provided') for record in records]))
        self.assertNotEqual(inventory_digest(records), inventory_digest(records[:1]))

    def test_pinned_range_audit_matches_current_inventory(self):
        source = COURSE / 'assets/citation-range-audit.json'
        report = json.loads(source.read_text())
        records = inventory()
        self.assertEqual(report['pins'], PINS)
        self.assertEqual(report['checked_engines'], sorted(PINS))
        self.assertEqual(report['inventory_sha256'], inventory_digest(records),
                         'Citation inputs changed: rerun the pinned-checkout audit before updating its evidence.')
        self.assertEqual(sum(report['counts'].values()), len(records))
        self.assertEqual(sum(sum(page.values()) for page in report['pages'].values()), len(records))
        published = ROOT / '_site/courses/inference/assets/citation-range-audit.json'
        self.assertEqual(published.read_bytes(), source.read_bytes())

    def test_summary_cli_writes_evidence_even_when_strict_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'audit.json'
            result = subprocess.run([
                sys.executable, str(ROOT / 'site/check_inference_citations.py'),
                '--summary', '--strict', '--output', str(output),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stdout, '')
            report = json.loads(output.read_text())
            self.assertNotIn('records', report)
            self.assertEqual(report['checked_engines'], [])
            self.assertGreater(report['counts']['checkout-not-provided'], 0)
            self.assertGreater(report['counts']['unresolved-engine'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
