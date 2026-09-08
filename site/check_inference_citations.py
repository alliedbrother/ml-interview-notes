"""Inventory explicit inference source citations; optionally check pinned checkouts.

No network access or source checkout modifications. Range checks establish only
that a path and line interval exist, not that quoted text or its interpretation is
correct. Abbreviations resolve only against explicit local citation anchors and
the supplied pinned trees; ambiguous references remain unresolved.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
COURSE = ROOT / 'content/courses/inference/html'
PINS = {
    'vllm': 'a556f3fccb701e5618d84d547ff454c56a1bfdfb',
    'sglang': '7d893255c359bb8ab74d2870c8ac865fb57230d6',
}
REFERENCE = re.compile(r'(?P<path>[A-Za-z0-9_./-]+\.[A-Za-z0-9_]+):L(?P<first>\d+)(?:[-\u2013]L?(?P<last>\d+))?')


def inventory(course=COURSE):
    records = []
    for path in sorted(course.rglob('*.html')):
        if path.name == 'README.html' and path.parent == course:
            continue  # Deliberate front-door duplicate.
        soup = BeautifulSoup(path.read_text(), 'html.parser')
        main = soup.find('main')
        if main is None:
            continue
        for node in main.select('code, .src'):
            # Avoid counting code descendants twice if a header contains code.
            if node.name == 'code' and node.find_parent(class_='src'):
                continue
            block = node.find_parent(class_='code-file')
            tag = block.select_one('.tag') if block else None
            context = tag.get_text(' ', strip=True).lower() if tag else ''
            for match in REFERENCE.finditer(node.get_text(' ', strip=True)):
                source = match['path']
                engine = ('sglang' if source.startswith('python/sglang/') else
                          'vllm' if source.startswith('vllm/') else
                          context if context in PINS else None)
                records.append(dict(page=str(path.relative_to(course)), path=source,
                                    first=int(match['first']), last=int(match['last'] or match['first']),
                                    engine=engine,
                                    section=(node.find_parent('section') or {}).get('id'),
                                    declared_engine=node.get('data-source-engine'),
                                    declared_path=node.get('data-source-path'),
                                    reviewed_symbol=node.get('data-source-symbol')))
    return records


def resolve_references(records, roots):
    """Resolve shortened paths using qualified local anchors, not basename search.

    The original fields stay unchanged for inventory fingerprinting. A complete
    multi-component path may resolve by exact existence in only one of both
    supplied pinned trees. No suffix/basename search of the trees is performed.
    Otherwise an anchor
    must name a real path in a pinned checkout and contain the abbreviated line
    interval. Same-section evidence takes precedence over same-page evidence;
    two distinct candidate files at the chosen scope are deliberately ambiguous.
    Resolved references never become anchors for further speculative inference.
    """
    anchors = defaultdict(list)
    for index, record in enumerate(records):
        for key in ('resolved_engine', 'resolved_path', 'resolution_status', 'resolution_evidence'):
            record.pop(key, None)
        if record['engine'] in roots and check_range(record, roots[record['engine']]) == 'range-exists':
            anchors[record['page']].append((index, record))
    for record in records:
        if record['engine'] is not None:
            continue
        declared_engine, declared_path = record.get('declared_engine'), record.get('declared_path')
        if declared_engine in roots and declared_path and record.get('reviewed_symbol'):
            declared = dict(record, path=declared_path)
            status = check_range(declared, roots[declared_engine])
            if status == 'range-exists':
                lines = (roots[declared_engine] / declared_path).read_text(errors='replace').splitlines()
                if record['reviewed_symbol'] in '\n'.join(lines[record['first'] - 1:record['last']]):
                    record['resolved_engine'], record['resolved_path'] = declared_engine, declared_path
                    record['resolution_status'] = 'reviewed-context-and-pinned-symbol'
                    record['resolution_evidence'] = dict(engine=declared_engine, path=declared_path,
                                                         symbol=record['reviewed_symbol'],
                                                         method='Authored per-occurrence context qualification; symbol rechecked at cited pinned lines')
                    continue
            record['resolution_status'] = 'reviewed-evidence-mismatch'
            continue
        # A written multi-component path can itself be a complete repository path.
        # Require both pinned trees: an omitted competing checkout is not absence.
        if set(roots) == set(PINS) and '/' in record['path'] and '..' not in Path(record['path']).parts:
            exact = [engine for engine, root in roots.items()
                     if check_range(dict(record, first=1, last=1), root) == 'range-exists']
            if len(exact) == 1:
                record['resolved_engine'], record['resolved_path'] = exact[0], record['path']
                record['resolution_status'] = 'unique-exact-path-in-pinned-trees'
                record['resolution_evidence'] = dict(path=record['path'], checked_engines=sorted(roots),
                                                     engine=exact[0], pins=PINS)
                continue
        candidates = []
        for index, anchor in anchors[record['page']]:
            if (anchor['path'] == record['path'] or anchor['path'].endswith('/' + record['path'])) and (
                    anchor['first'] <= record['first'] <= record['last'] <= anchor['last']):
                candidates.append((index, anchor))
        local = [(index, anchor) for index, anchor in candidates
                 if record.get('section') is not None and anchor.get('section') == record['section']]
        chosen = local or candidates
        identities = {(anchor['engine'], anchor['path']) for _, anchor in chosen}
        if len(identities) != 1:
            record['resolution_status'] = 'ambiguous-local-anchors' if identities else 'no-qualified-local-anchor'
            continue
        index, anchor = min(chosen, key=lambda pair: pair[1]['last'] - pair[1]['first'])
        resolved = dict(record, engine=anchor['engine'], path=anchor['path'])
        if check_range(resolved, roots[anchor['engine']]) != 'range-exists':
            record['resolution_status'] = 'anchor-range-not-verified'
            continue
        record['resolved_engine'], record['resolved_path'] = anchor['engine'], anchor['path']
        record['resolution_status'] = 'same-section-qualified-anchor' if local else 'same-page-qualified-anchor'
        record['resolution_evidence'] = dict(record_index=index, page=anchor['page'],
                                             section=anchor.get('section'), engine=anchor['engine'],
                                             path=anchor['path'], first=anchor['first'], last=anchor['last'])
    return records


def check_range(record, checkout):
    if record['first'] < 1 or record['last'] < record['first']:
        return 'invalid-range'
    root = checkout.resolve()
    target = (root / record['path']).resolve()
    if not target.is_relative_to(root):
        return 'outside-checkout'
    if not target.is_file():
        return 'missing-path'
    count = len(target.read_text(errors='replace').splitlines())
    return 'range-exists' if record['last'] <= count else 'range-past-eof'


def inventory_digest(records):
    """Fingerprint inputs, excluding status and URLs added during verification."""
    fields = ('page', 'path', 'first', 'last', 'engine', 'section',
              'declared_engine', 'declared_path', 'reviewed_symbol')
    canonical = [{key: record.get(key) for key in fields} for record in records]
    return hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()


def summarize(records, checked_engines):
    pages = defaultdict(Counter)
    unresolved = defaultdict(lambda: dict(pages=Counter(), reasons=Counter()))
    for record in records:
        pages[record['page']][record['status']] += 1
        if record['status'] == 'unresolved-engine':
            unresolved[record['path']]['pages'][record['page']] += 1
            unresolved[record['path']]['reasons'][record.get('resolution_status', 'no-resolution-evidence')] += 1
    return dict(
        scope='Explicit path:line references in code and source headers only; not a complete bibliography or quote verifier',
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        pins=PINS,
        checked_engines=sorted(checked_engines),
        inventory_sha256=inventory_digest(records),
        counts=dict(sorted(Counter(record['status'] for record in records).items())),
        resolution_counts=dict(sorted(Counter(record.get('resolution_status', 'explicit-engine')
                                              for record in records).items())),
        unresolved_by_path={path: {key: dict(sorted(counts.items())) for key, counts in details.items()}
                            for path, details in sorted(unresolved.items())},
        pages={page: dict(sorted(counts.items())) for page, counts in sorted(pages.items())},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root', type=Path)
    parser.add_argument('--sglang-root', type=Path)
    parser.add_argument('--strict', action='store_true', help='Fail on any unchecked or invalid citation')
    parser.add_argument('--summary', action='store_true', help='Emit counts by page without individual records')
    parser.add_argument('--output', type=Path, help='Write the JSON report to this path instead of stdout')
    args = parser.parse_args()
    roots = {'vllm': args.vllm_root, 'sglang': args.sglang_root}
    checked_roots = {}
    for engine, root in roots.items():
        if root is None:
            continue
        result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != PINS[engine] or dirty.returncode or dirty.stdout.strip():
            parser.error(f'{engine} requires a clean checkout at {PINS[engine]}')
        checked_roots[engine] = root
    records = resolve_references(inventory(), checked_roots)
    for record in records:
        engine = record.get('resolved_engine', record['engine'])
        source = record.get('resolved_path', record['path'])
        record['status'] = ('unresolved-engine' if engine is None else
                            'checkout-not-provided' if engine not in checked_roots else
                            check_range(dict(record, path=source), checked_roots[engine]))
        if engine:
            repo = 'vllm-project/vllm' if engine == 'vllm' else 'sgl-project/sglang'
            record['url'] = f"https://github.com/{repo}/blob/{PINS[engine]}/{source}#L{record['first']}-L{record['last']}"
    report = summarize(records, checked_roots)
    if not args.summary:
        report['records'] = records
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output, encoding='utf-8')
    else:
        print(output, end='')
    if args.strict and any(r['status'] != 'range-exists' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
