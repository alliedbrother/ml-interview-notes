#!/usr/bin/env python3
"""
Verify the built site.

A static site fails silently: a link rots, an asset moves, a diagram stops
rendering, and nothing errors — the page just quietly gets worse. These checks
run in CI so a regression blocks the deploy rather than shipping.

Usage:  python site/build.py && python site/test_build.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
OUT = ROOT / "_site"

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def resolves(url: str) -> bool:
    """Does a root-absolute site URL correspond to a real built file?"""
    path = unquote(urlparse(url).path)
    target = OUT / path.lstrip("/")
    return target.is_file() or (target / "index.html").is_file()


def main() -> int:
    if not OUT.is_dir():
        print("_site/ not found — run `python site/build.py` first", file=sys.stderr)
        return 1

    pages = sorted(OUT.rglob("index.html"))
    check(bool(pages), "no pages were built")

    # ---- every source markdown produced a page
    sources = [p for p in CONTENT.rglob("*.md")]
    check(bool(sources), "no markdown sources found in content/")
    for src in sources:
        rel = src.relative_to(CONTENT)
        if rel.parts[0] == "courses":
            course, stem = rel.parts[1], src.stem
            num = re.match(r"(\d+)", stem)
            expected = (
                OUT / "courses" / course / "index.html" if num and int(num.group(1)) == 0
                else OUT / "courses" / course / stem / "index.html"
            )
        elif rel.parts[0] == "notes":
            expected = OUT / "notes" / src.stem / "index.html"
        else:
            continue
        check(expected.is_file(), f"no page built for source {rel}")

    # ---- links and assets resolve
    link_re = re.compile(r'(?:href|src)="(/[^"#]*)"')
    for page in pages:
        html_text = page.read_text(encoding="utf-8")
        rel_page = page.relative_to(OUT)
        for url in set(link_re.findall(html_text)):
            check(resolves(url), f"{rel_page}: dead link -> {url}")

        # ---- no unrewritten markdown links escaped into the output
        for bad in set(re.findall(r'href="([^"]*\.md(?:#[^"]*)?)"', html_text)):
            check(False, f"{rel_page}: unrewritten markdown link -> {bad}")

        # ---- shared stylesheet is linked, not inlined per page
        check('href="/assets/style.css"' in html_text,
              f"{rel_page}: missing shared stylesheet link")

    # ---- assets exist
    for asset in ("style.css", "page.js", "mermaid.js", "katex-init.js", "favicon.svg"):
        check((OUT / "assets" / asset).is_file(), f"missing asset assets/{asset}")

    # ---- Pages plumbing
    check((OUT / "CNAME").is_file(), "missing CNAME")
    check((OUT / ".nojekyll").is_file(), "missing .nojekyll")
    check((OUT / "index.html").is_file(), "missing site home")
    check((OUT / "courses" / "index.html").is_file(), "missing courses index")

    # ---- every mermaid fence became a diagram shell
    for src in (CONTENT / "courses").rglob("*.md"):
        fences = len(re.findall(r"^```mermaid\s*$", src.read_text(encoding="utf-8"), re.M))
        if not fences:
            continue
        stem, course = src.stem, src.relative_to(CONTENT / "courses").parts[0]
        num = re.match(r"(\d+)", stem)
        built = (
            OUT / "courses" / course / "index.html" if num and int(num.group(1)) == 0
            else OUT / "courses" / course / stem / "index.html"
        )
        if not built.is_file():
            continue
        shells = built.read_text(encoding="utf-8").count('class="diagram-shell"')
        check(shells == fences,
              f"{stem}: {fences} mermaid fences produced {shells} diagram shells")

    # ---- math survived the pipeline
    math_pages = [p for p in pages if "math-display" in p.read_text(encoding="utf-8")
                  or "math-inline" in p.read_text(encoding="utf-8")]
    check(bool(math_pages), "no rendered math found anywhere — the KaTeX pipeline may be broken")

    # ---- course navigation is wired up
    course_pages = sorted((OUT / "courses").rglob("index.html"))
    module_pages = [p for p in course_pages if p.parent.parent.name != "courses"
                    and p.parent.name != "courses"]
    for page in module_pages:
        text = page.read_text(encoding="utf-8")
        check('class="nav__i' in text, f"{page.relative_to(OUT)}: missing module rail")
        check('class="pager"' in text, f"{page.relative_to(OUT)}: missing prev/next pager")

    if failures:
        print(f"FAILED — {len(failures)} of {checks} checks\n", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"OK — {checks} checks passed across {len(pages)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
