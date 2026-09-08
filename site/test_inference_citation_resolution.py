"""Adversarial fixtures for provenance-aware shortened source references."""

from pathlib import Path
import tempfile
import unittest

from check_inference_citations import check_range, inventory, inventory_digest, resolve_references


def ref(path, engine=None, first=3, last=5, page='lesson.html', section='engine-code'):
    return dict(page=page, section=section, path=path, engine=engine, first=first, last=last)


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.roots = {engine: Path(self.temp.name) / engine for engine in ('vllm', 'sglang')}
        for root in self.roots.values():
            root.mkdir()

    def source(self, engine, path, lines=20):
        target = self.roots[engine] / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('line\n' * lines)

    def test_same_section_anchor_records_full_provenance(self):
        self.source('vllm', 'vllm/worker/runner.py')
        records = [ref('vllm/worker/runner.py', 'vllm', first=1, last=10), ref('runner.py')]
        digest = inventory_digest(records)
        resolve_references(records, self.roots)
        result = records[1]
        self.assertEqual(result['path'], 'runner.py')
        self.assertIsNone(result['engine'])
        self.assertEqual(result['resolved_path'], records[0]['path'])
        self.assertEqual(result['resolution_status'], 'same-section-qualified-anchor')
        self.assertEqual(result['resolution_evidence']['record_index'], 0)
        self.assertEqual(inventory_digest(records), digest)
        records[1]['section'] = 'different-context'
        self.assertNotEqual(inventory_digest(records), digest)

    def test_same_page_containing_interval_without_section_match(self):
        self.source('sglang', 'python/sglang/runner.py')
        records = [ref('python/sglang/runner.py', 'sglang', first=1, last=10, section='source'),
                   ref('runner.py', section='discussion')]
        resolve_references(records, self.roots)
        self.assertEqual(records[1]['resolution_status'], 'same-page-qualified-anchor')
        self.assertEqual(records[1]['resolved_engine'], 'sglang')

    def test_does_not_borrow_from_other_pages_or_disjoint_intervals(self):
        self.source('vllm', 'vllm/worker/runner.py')
        records = [ref('vllm/worker/runner.py', 'vllm', first=1, last=10),
                   ref('runner.py', page='different.html'), ref('runner.py', first=11, last=12)]
        resolve_references(records, self.roots)
        for record in records[1:]:
            self.assertNotIn('resolved_path', record)

    def test_ambiguous_engine_and_file_remain_unresolved(self):
        self.source('vllm', 'vllm/worker/runner.py')
        self.source('sglang', 'python/sglang/runner.py')
        records = [ref('vllm/worker/runner.py', 'vllm'),
                   ref('python/sglang/runner.py', 'sglang'), ref('runner.py')]
        resolve_references(records, self.roots)
        self.assertEqual(records[-1]['resolution_status'], 'ambiguous-local-anchors')
        self.assertNotIn('resolved_engine', records[-1])

    def test_nearest_section_has_explicit_precedence(self):
        self.source('vllm', 'vllm/worker/runner.py')
        self.source('sglang', 'python/sglang/runner.py')
        records = [ref('vllm/worker/runner.py', 'vllm', section='vllm-section'),
                   ref('python/sglang/runner.py', 'sglang', section='sglang-section'),
                   ref('runner.py', section='sglang-section')]
        resolve_references(records, self.roots)
        self.assertEqual(records[-1]['resolved_engine'], 'sglang')

    def test_invalid_anchor_and_suffix_basename_search_are_not_evidence(self):
        self.source('vllm', 'vllm/worker/runner.py', lines=4)
        records = [ref('vllm/worker/runner.py', 'vllm', first=1, last=10), ref('runner.py')]
        resolve_references(records, self.roots)
        self.assertNotIn('resolved_path', records[-1])
        # A unique basename in a checkout is still not an explicit local anchor.
        self.source('vllm', 'vllm/deep/unique.py')
        records = [ref('unique.py')]
        resolve_references(records, self.roots)
        self.assertNotIn('resolved_path', records[0])

    def test_exact_multicomponent_path_requires_both_trees(self):
        self.source('vllm', 'csrc/kernel.cu')
        records = [ref('csrc/kernel.cu')]
        resolve_references(records, {'vllm': self.roots['vllm']})
        self.assertNotIn('resolved_engine', records[0])
        resolve_references(records, self.roots)
        self.assertEqual(records[0]['resolved_engine'], 'vllm')
        self.assertEqual(records[0]['resolution_status'], 'unique-exact-path-in-pinned-trees')
        self.assertEqual(records[0]['resolution_evidence']['checked_engines'], ['sglang', 'vllm'])
        # No stale resolution survives a rerun with fewer checkouts.
        resolve_references(records, {})
        self.assertNotIn('resolved_engine', records[0])

    def test_exact_shared_path_is_ambiguous_even_when_one_interval_fits(self):
        self.source('vllm', 'csrc/kernel.cu', lines=20)
        self.source('sglang', 'csrc/kernel.cu', lines=2)
        records = [ref('csrc/kernel.cu')]
        resolve_references(records, self.roots)
        self.assertNotIn('resolved_engine', records[0])

    def test_exact_path_resolution_does_not_hide_invalid_line_ranges(self):
        self.source('vllm', 'csrc/kernel.cu', lines=2)
        records = [ref('csrc/kernel.cu')]
        resolve_references(records, self.roots)
        self.assertEqual(records[0]['resolved_engine'], 'vllm')
        self.assertEqual(check_range(records[0], self.roots['vllm']), 'range-past-eof')

    def test_basename_root_files_and_path_escape_do_not_resolve(self):
        self.source('vllm', 'README.md')
        records = [ref('README.md'), ref('../vllm/README.md')]
        resolve_references(records, self.roots)
        for record in records:
            self.assertNotIn('resolved_engine', record)

    def test_authored_qualification_requires_symbol_at_cited_lines(self):
        self.source('vllm', 'vllm/worker/runner.py')
        record = ref('runner.py') | dict(declared_engine='vllm',
                                         declared_path='vllm/worker/runner.py', reviewed_symbol='line')
        records = [record]
        resolve_references(records, self.roots)
        self.assertEqual(record['resolution_status'], 'reviewed-context-and-pinned-symbol')
        self.assertEqual(record['resolution_evidence']['symbol'], 'line')
        record['reviewed_symbol'] = 'wrong_function'
        resolve_references(records, self.roots)
        self.assertEqual(record['resolution_status'], 'reviewed-evidence-mismatch')
        self.assertNotIn('resolved_engine', record)

    def test_inventory_preserves_authored_qualification_and_section(self):
        root = Path(self.temp.name)
        (root / 'lesson.html').write_text(
            '<main><section id="worker"><p>Runner: '
            '<code data-source-engine="vllm" data-source-path="vllm/runner.py" '
            'data-source-symbol="prepare_inputs">runner.py:L3-L8</code>'
            '</p></section></main>')
        records = inventory(root)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['section'], 'worker')
        self.assertEqual(records[0]['declared_path'], 'vllm/runner.py')
        self.assertEqual(records[0]['reviewed_symbol'], 'prepare_inputs')
        self.assertIsNone(records[0]['engine'])

    def test_authored_qualification_cannot_escape_or_hide_out_of_bounds(self):
        self.source('vllm', 'vllm/worker/runner.py', lines=2)
        for path in ('../sglang/outside.py', 'vllm/worker/runner.py'):
            records = [ref('runner.py') | dict(declared_engine='vllm', declared_path=path,
                                               reviewed_symbol='line')]
            resolve_references(records, self.roots)
            self.assertEqual(records[0]['resolution_status'], 'reviewed-evidence-mismatch')
            self.assertNotIn('resolved_engine', records[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
