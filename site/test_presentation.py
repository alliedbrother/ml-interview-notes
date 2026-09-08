"""Check content preservation and navigation across every published page."""

import json
from pathlib import Path
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "_site"


def main():
    pages = {p: BeautifulSoup(p.read_text(), "html.parser") for p in OUT.rglob("*.html")}
    errors = []
    for path, soup in pages.items():
        if soup.select_one('.library-nav a[href="/status/"]'):
            errors.append(f"{path.relative_to(OUT)}: build status remains in left navigation")
        if "The Inference Engineering Book" in str(soup):
            errors.append(f"{path.relative_to(OUT)}: outdated inference course title")
        if path.is_relative_to(OUT / "courses/transformers") and soup.select_one("footer.foot"):
            errors.append(f"{path.relative_to(OUT)}: removed course footer remains")
        for selector in ("main#main-content", "#search-dialog", "#settings-dialog", '[src="/assets/reader.js"]', '[href="/assets/reader.css"]'):
            if not soup.select_one(selector):
                errors.append(f"{path.relative_to(OUT)}: missing {selector}")
        ids = [node["id"] for node in soup.select("[id]")]
        if len(ids) != len(set(ids)):
            errors.append(f"{path.relative_to(OUT)}: duplicate IDs")
        for node in soup.select("a[href], script[src], link[href], img[src]"):
            href = node.get("href") or node.get("src")
            url = urlsplit(href)
            if url.scheme or url.netloc:
                continue
            target = (OUT / unquote(url.path.lstrip("/")) if url.path.startswith("/") else path.parent / unquote(url.path)) if url.path else path
            target = target.resolve()
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                errors.append(f"{path.relative_to(OUT)}: missing {href}")
            elif url.fragment and target in pages:
                if not pages[target].find(id=unquote(url.fragment)):
                    errors.append(f"{path.relative_to(OUT)}: missing anchor {href}")

    transformers = pages[OUT / "courses/transformers/index.html"]
    if transformers.find(id="source-material") or "Source material" in transformers.get_text():
        errors.append("Transformers overview: removed source material section remains")

    download_roots = (
        (ROOT / "content/notes/_examples", OUT / "assets/examples"),
        (ROOT / "content/courses/transformers/code", OUT / "courses/transformers/code"),
    )
    for standalone, destination in download_roots:
        for source in standalone.rglob("*"):
            if (not source.is_file() or "__pycache__" in source.parts
                    or source.suffix == ".pyc" or source.name == ".DS_Store"):
                continue
            published = destination / source.relative_to(standalone)
            if not published.is_file() or published.read_bytes() != source.read_bytes():
                errors.append(f"{source.name}: standalone download missing or different from source")

    preserved = 0
    source = ROOT / "content/courses/inference/html"
    for path in source.rglob("*.html"):
        before = BeautifulSoup(path.read_text(), "html.parser")
        after = BeautifulSoup((OUT / "courses/inference" / path.relative_to(source)).read_text(), "html.parser")
        for tools in after.select(".reader-tools"):
            tools.decompose()
        original = " ".join(before.main.get_text(" ", strip=True).split())
        published = " ".join(after.main.get_text(" ", strip=True).split())
        if original != published:
            errors.append(f"{path.name}: imported lesson text changed")
        for selector in ("main pre", ".diagram-source"):
            if [x.get_text() for x in before.select(selector)] != [x.get_text() for x in after.select(selector)]:
                errors.append(f"{path.name}: {selector} changed")
        if any("import elkLayouts from" in (s.string or "") for s in after.find_all("script")):
            errors.append(f"{path.name}: embedded legacy diagram loader remains")
        preserved += 1
    index = json.loads((OUT / "assets/search.json").read_text())
    assert "The Inference Engineering Book" not in json.dumps(index), "Outdated inference title in search"
    assert len({row["url"] for row in index}) == len(index), "Duplicate search URLs"
    assert sum(row["kind"] == "labs" for row in index) == 13, "Labs missing from search"
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"OK: {len(pages)} pages and their links verified; {preserved} imported lessons preserved; {len(index)} search entries")


if __name__ == "__main__":
    main()
