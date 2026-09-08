"""Inventory explicit inference source citations; optionally check pinned checkouts.

No network access or source checkout modifications. Range checks establish only
that a path and line interval exist, not that quoted text or its interpretation is
correct. Unqualified paths outside engine-tagged blocks remain unresolved.
"""

import argparse
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
                                    engine=engine))
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vllm-root', type=Path)
    parser.add_argument('--sglang-root', type=Path)
    parser.add_argument('--strict', action='store_true', help='Fail on any unchecked or invalid citation')
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
    records = inventory()
    for record in records:
        engine = record['engine']
        record['status'] = ('unresolved-engine' if engine is None else
                            'checkout-not-provided' if engine not in checked_roots else
                            check_range(record, checked_roots[engine]))
        if engine:
            repo = 'vllm-project/vllm' if engine == 'vllm' else 'sgl-project/sglang'
            record['url'] = f"https://github.com/{repo}/blob/{PINS[engine]}/{record['path']}#L{record['first']}-L{record['last']}"
    counts = {status: sum(r['status'] == status for r in records) for status in sorted({r['status'] for r in records})}
    print(json.dumps(dict(scope='Explicit path:line references in code and source headers only; not a complete bibliography or quote verifier',
                          pins=PINS, counts=counts, records=records), indent=2))
    if args.strict and any(r['status'] != 'range-exists' for r in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
