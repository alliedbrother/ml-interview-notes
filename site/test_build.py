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
    sources = [p for p in CONTENT.rglob("*.md")
               if not any(part.startswith("_") for part in p.relative_to(CONTENT).parts)]
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
            # a category is notes/<slug>.md; a topic is notes/<slug>/<topic>.md
            expected = OUT.joinpath(*rel.parts[:-1], src.stem, "index.html")
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

        # ---- generated pages link the shared stylesheet; prebuilt course pages
        #      carry their own CSS and are re-themed by the bridge instead
        if 'href="/assets/prebuilt-bridge.css"' not in html_text:
            check('href="/assets/style.css"' in html_text,
                  f"{rel_page}: missing shared stylesheet link")

    # ---- prebuilt courses: every page re-themed and given the site bar, and
    #      every relative link inside the tree resolves to a real file
    import yaml
    for cfg_file in (CONTENT / "courses").glob("*/course.yml"):
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        if cfg.get("type") != "prebuilt":
            continue
        slug = cfg.get("slug", cfg_file.parent.name)
        root = OUT / "courses" / slug
        check(root.is_dir(), f"prebuilt course {slug}: not built")
        pages_p = sorted(root.rglob("*.html"))
        check(bool(pages_p), f"prebuilt course {slug}: no pages")
        check((root / cfg.get("entry", "index.html")).is_file(),
              f"prebuilt course {slug}: missing entry page")
        rel_re = re.compile(r'(?:href|src)="(?!https?:|/|#|mailto:|data:)([^"#?]+)')
        for page in pages_p:
            text = page.read_text(encoding="utf-8")
            rel_page = page.relative_to(OUT)
            check('href="/assets/prebuilt-bridge.css"' in text,
                  f"{rel_page}: prebuilt page not re-themed")
            check('href="/assets/favicon.svg"' in text,
                  f"{rel_page}: missing favicon link (browser falls back to /favicon.ico)")
            # every page with the course rail must also carry the Labs group;
            # a mis-anchored injection fails by silently changing nothing
            if '<nav class="nav"' in text:
                check('nav__grp">Labs<span>' in text,
                      f"{rel_page}: course rail is missing the Labs group")
            # must be in the BODY: a naive injection can bury it in a CSS
            # comment inside <head>, where the browser discards it silently
            head_end = text.find("</head>")
            bar_at = text.find('class="topbar"')
            check(bar_at != -1 and head_end != -1 and bar_at > head_end,
                  f"{rel_page}: site bar missing or not inside <body>")
            for target in set(rel_re.findall(text)):
                check((page.parent / target).exists(),
                      f"{rel_page}: dead relative link -> {target}")
        check((OUT / "assets" / "prebuilt-bridge.css").is_file(),
              "missing assets/prebuilt-bridge.css")

    # ---- mermaid loaders must degrade instead of blaming the content.
    #      A static elk import cannot be caught, and mermaid reports a missing
    #      layout engine as "Syntax error in text" — telling readers a perfectly
    #      valid diagram is broken.
    loaders = sorted(OUT.rglob("mermaid.js"))
    check(bool(loaders), "no mermaid loader found in the output")
    for loader in loaders:
        src = loader.read_text(encoding="utf-8")
        rel = loader.relative_to(OUT)
        check("import elkLayouts from" not in src,
              f"{rel}: static elk import — a CDN failure cannot be caught")
        check("await import(" in src and "layout-elk" in src,
              f"{rel}: elk is not loaded dynamically")
        check("layoutEngine" in src and "'dagre'" in src,
              f"{rel}: no dagre fallback when elk is unavailable")

    # ---- roadmap.yml integrity. An auto-resolved merge once kept BOTH sides of
    #      a conflict, leaving "Libraries — 9 written (done)" beside a stale
    #      "Libraries — 6 topics (in progress)". The file stayed valid YAML and
    #      every page built, so nothing caught it. Titles are keyed on the part
    #      before " — " so a status change that also rewords the suffix is still
    #      recognised as the same item.
    roadmap = CONTENT / "roadmap.yml"
    if roadmap.is_file():
        cfg = yaml.safe_load(roadmap.read_text(encoding="utf-8"))
        allowed = {"done", "progress", "planned"}
        for section in cfg.get("sections", []):
            seen: dict[str, str] = {}
            for item in section.get("items", []):
                title = item.get("title", "")
                key = title.split(" — ")[0].split(" - ")[0].strip().lower()
                check(item.get("status") in allowed,
                      f"roadmap [{section['name']}] '{title}': bad status {item.get('status')!r}")
                check(key not in seen,
                      f"roadmap [{section['name']}]: duplicate item '{title}' "
                      f"(also '{seen.get(key)}') — likely a mis-resolved merge")
                seen.setdefault(key, title)

    # ---- assets exist
    for asset in ("style.css", "page.js", "mermaid.js", "katex-init.js", "favicon.svg"):
        check((OUT / "assets" / asset).is_file(), f"missing asset assets/{asset}")

    # ---- Pages plumbing
    check((OUT / "CNAME").is_file(), "missing CNAME")
    check((OUT / ".nojekyll").is_file(), "missing .nojekyll")
    check((OUT / "index.html").is_file(), "missing site home")
    check((OUT / "courses" / "index.html").is_file(), "missing courses index")
    check((OUT / "status" / "index.html").is_file(), "missing build status page")

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
