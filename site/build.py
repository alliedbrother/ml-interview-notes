#!/usr/bin/env python3
"""
Build ml-interview-notes as a static site for GitHub Pages.

Reads Markdown from content/ and writes HTML to _site/. Sources and build
artifacts never share a directory.

Content model
-------------
  content/site.yml                     site title, tagline, top nav
  content/courses/<slug>/course.yml    course title, blurb, track definitions
  content/courses/<slug>/NN-*.md       modules; 00 becomes the course index
  content/notes/<slug>.md              a note CATEGORY page
  content/notes/<slug>/<topic>.md      a TOPIC inside that category
                                       (front matter: order, description, meta,
                                        scripts)

The home page and section indexes are generated from this content, so adding a
course or a note needs no code change.

URLs are directory-style and every internal link is rendered root-absolute
(`/courses/transformers/03-self-attention-from-scratch/`), which removes all
relative-depth arithmetic. This is correct under an apex custom domain; it would
need revisiting only if the site moved to a `user.github.io/repo/` path.

Usage:  python site/build.py [--serve]
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import sys
from pathlib import Path

import yaml
from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin

ROOT = Path(__file__).resolve().parent.parent
CONTENT = ROOT / "content"
THEME = Path(__file__).resolve().parent / "theme"
OUT = ROOT / "_site"


# ---------------------------------------------------------------- helpers

def slugify(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"[`*_]", "", text)
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-{2,}", "-", text) or "section"


def strip_md(text: str) -> str:
    """Markdown source -> bare text, for titles and meta descriptions."""
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    return re.sub(r"[*_]", "", text).strip()


def first_paragraph(body: str) -> str:
    """The lead paragraph of a markdown body, skipping blockquotes and fences."""
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block or block.startswith((">", "#", "```", "|", "-", "*")):
            continue
        return " ".join(strip_md(block).split())
    return ""


def truncate(text: str, limit: int = 180) -> str:
    """Trim to a word boundary rather than mid-word."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:.") + "…"


def write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


# ---------------------------------------------------------------- markdown

def build_md() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": True})
    md.enable("table").enable("strikethrough")
    md.use(dollarmath_plugin, double_inline=True)

    def esc_tex(s: str) -> str:
        return html.escape(s, quote=False)

    def r_fence(self, tokens, idx, options, env):
        tok = tokens[idx]
        info = (tok.info or "").strip().lower()
        content = tok.content.rstrip("\n")
        if info == "html":
            # Raw passthrough for interactive widgets. Content in this repo is
            # trusted (it arrives by reviewed pull request), so this is a
            # deliberate escape hatch, not a hole — markdown-it still runs with
            # html:False for everything else.
            env.setdefault("raw_html", []).append(content)
            return content + "\n"
        if info == "mermaid":
            env.setdefault("diagrams", []).append(content)
            # <script type="text/plain"> is a RAW TEXT element: entities are not
            # decoded, so the source must go in verbatim. Only the closing-tag
            # sequence needs neutralising.
            src = re.sub(r"</(script)", r"<\\/\1", content, flags=re.I)
            return DIAGRAM_SHELL.format(src=src)
        label = info if info else "text"
        # long single-line output benefits from wrapping; code does not
        wrap = " code-block--wrap" if info in ("", "text") and max(
            (len(l) for l in content.split("\n")), default=0) > 100 else ""
        return (
            f'<div class="code-file">'
            f'<div class="code-file__hd">{html.escape(label)}</div>'
            f'<pre class="code-block{wrap}"><code>{html.escape(content, quote=False)}</code></pre>'
            f"</div>\n"
        )

    def r_table_open(self, tokens, idx, options, env):
        return '<div class="table-wrap"><div class="table-scroll"><table class="dt">\n'

    def r_table_close(self, tokens, idx, options, env):
        return "</table></div></div>\n"

    def r_bq_open(self, tokens, idx, options, env):
        return '<div class="callout">\n'

    def r_bq_close(self, tokens, idx, options, env):
        return "</div>\n"

    def r_math_inline(self, tokens, idx, options, env):
        return f'<span class="math-inline">\\({esc_tex(tokens[idx].content)}\\)</span>'

    def r_math_block(self, tokens, idx, options, env):
        return f'<div class="math-display">\\[{esc_tex(tokens[idx].content)}\\]</div>\n'

    def r_link_open(self, tokens, idx, options, env):
        """Rewrite .md links to the target page's root-absolute URL."""
        tok = tokens[idx]
        href = tok.attrGet("href") or ""
        if href.startswith("http"):
            tok.attrSet("target", "_blank")
            tok.attrSet("rel", "noopener")
        elif ".md" in href:
            path, _, frag = href.partition("#")
            src_dir = env.get("src_dir")
            urls = env.get("urls", {})
            if src_dir is not None:
                target = (src_dir / path).resolve()
                url = urls.get(target)
                if url:
                    tok.attrSet("href", url + (f"#{frag}" if frag else ""))
                else:
                    env.setdefault("broken", []).append(href)
        return self.renderToken(tokens, idx, options, env)

    md.add_render_rule("fence", r_fence)
    md.add_render_rule("table_open", r_table_open)
    md.add_render_rule("table_close", r_table_close)
    md.add_render_rule("blockquote_open", r_bq_open)
    md.add_render_rule("blockquote_close", r_bq_close)
    md.add_render_rule("math_inline", r_math_inline)
    md.add_render_rule("math_inline_double", r_math_block)
    md.add_render_rule("math_block", r_math_block)
    md.add_render_rule("link_open", r_link_open)
    return md


DIAGRAM_SHELL = """<section class="diagram-shell">
  <p class="diagram-shell__hint">Ctrl/Cmd + wheel to zoom &middot; drag to pan &middot; double-click to fit &middot; &#x26F6; full size</p>
  <div class="mermaid-wrap">
    <div class="zoom-controls">
      <button type="button" data-action="zoom-in" title="Zoom in">+</button>
      <button type="button" data-action="zoom-out" title="Zoom out">&minus;</button>
      <button type="button" data-action="zoom-fit" title="Smart fit">&#8634;</button>
      <button type="button" data-action="zoom-one" title="Actual size">1:1</button>
      <button type="button" data-action="zoom-expand" title="Open full size">&#x26F6;</button>
      <span class="zoom-label">Loading&hellip;</span>
    </div>
    <div class="mermaid-viewport"><div class="mermaid mermaid-canvas"></div></div>
  </div>
  <script type="text/plain" class="diagram-source">
{src}
  </script>
</section>
"""


# ---------------------------------------------------------------- parsing

FENCE_RE = re.compile(r"^(```|~~~)")


def split_sections(body: str):
    """Split markdown on top-level '## ' headings, ignoring fenced regions."""
    out, cur_title, buf, in_fence = [], None, [], False
    for line in body.split("\n"):
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not in_fence and line.startswith("## "):
            out.append((cur_title, "\n".join(buf)))
            cur_title, buf = line[3:].strip(), []
        else:
            buf.append(line)
    out.append((cur_title, "\n".join(buf)))
    return out


NAV_LINE_RE = re.compile(
    r"^\s*\*\*(?:Next|Back to|Start here)\s*(?:&rarr;|→|&larr;|←)?.*$", re.I)


def strip_nav_lines(text: str) -> str:
    """Drop hand-written prev/next lines; the generated pager replaces them."""
    keep = [l for l in text.split("\n") if not NAV_LINE_RE.match(l)]
    text = "\n".join(keep)
    return re.sub(r"\n(?:\s*---\s*\n)+\s*$", "\n", text).rstrip() + "\n"


def section_kind(title: str | None) -> str:
    if not title:
        return "intro"
    t = title.lower()
    if t.startswith("key takeaway"):
        return "takeaways"
    if t.startswith("self-check"):
        return "selfcheck"
    if t.startswith("reconciling"):
        return "reconcile"
    return "normal"


# ---------------------------------------------------------------- model

class Page:
    """One markdown source file and the page it becomes."""

    def __init__(self, path: Path, url: str, eyebrow: str, track: str = "foundations"):
        self.path = path
        self.url = url
        self.track = track
        self.src_dir = path.parent
        self.topics: list["Page"] = []   # categories only
        self.parent: "Page | None" = None  # topics only
        m = re.match(r"(\d+)", path.stem)
        self.num = int(m.group(1)) if m else 0

        raw = path.read_text(encoding="utf-8")

        # optional YAML front matter: order, description, title
        self.meta: dict = {}
        fm = re.match(r"---\n(.*?)\n---\n", raw, re.S)
        if fm:
            self.meta = yaml.safe_load(fm.group(1)) or {}
            raw = raw[fm.end():]
        self.order = self.meta.get("order", self.num)

        h1 = re.search(r"^# (.+)$", raw, re.M)
        self.h1 = h1.group(1).strip() if h1 else path.stem

        # "01 — Motivation & History" -> eyebrow "Module 01", title "Motivation & History"
        parts = re.split(r"\s+[—–-]\s+", self.h1, maxsplit=1)
        if len(parts) == 2 and re.fullmatch(r"\d+", parts[0].strip()):
            self.eyebrow = f"{eyebrow} {parts[0].strip()}"
            self.title = parts[1].strip()
        else:
            self.eyebrow = eyebrow
            self.title = self.h1
        self.short = re.sub(r"\s*[:—–].*$", "", self.title)

        body = raw[h1.end():] if h1 else raw
        body = strip_nav_lines(body)
        body = re.sub(r"^\s*(?:---\s*\n)+", "", body)

        # a leading blockquote is the prerequisites box
        self.prereq = ""
        bq = re.match(r"\s*((?:^>.*\n?)+)", body, re.M)
        if bq:
            self.prereq = re.sub(r"^> ?", "", bq.group(1), flags=re.M).strip()
            body = body[bq.end():]
        self.body = re.sub(r"^\s*(?:---\s*\n)+", "", body)

    @property
    def description(self) -> str:
        """Card blurb and meta description: front matter wins, else the lead paragraph."""
        return self.meta.get("description") or truncate(first_paragraph(self.body))

    @property
    def out_path(self) -> Path:
        return OUT / self.url.strip("/") / "index.html" if self.url != "/" else OUT / "index.html"


class Course:
    def __init__(self, directory: Path):
        self.dir = directory
        cfg = yaml.safe_load((directory / "course.yml").read_text(encoding="utf-8"))
        self.order = cfg.get("order", 99)
        self.slug = cfg.get("slug", directory.name)
        self.title = cfg["title"]
        self.subtitle = cfg.get("subtitle", "")
        self.blurb = " ".join(cfg.get("blurb", "").split())
        self.level = cfg.get("level", "")
        self.brand_kicker = cfg.get("brand_kicker", self.title)
        self.brand_title = cfg.get("brand_title", self.subtitle or self.title)
        self.modules_label = cfg.get("modules_label", "Modules")
        self.footer_html = cfg.get("footer_html", "")
        self.tracks = cfg.get("tracks", [])
        self.base = f"/courses/{self.slug}/"

        # A "prebuilt" course ships as finished HTML rather than Markdown. Its
        # pages are copied verbatim and re-themed through CSS custom properties,
        # so no conversion can corrupt the content. See render_prebuilt().
        self.prebuilt = cfg.get("type") == "prebuilt"
        self.module_count = cfg.get("module_count", 0)
        self.pages, self.modules, self.index = [], [], None
        if self.prebuilt:
            self.html_dir = directory / cfg.get("source", "html")
            if not self.html_dir.is_dir():
                raise SystemExit(f"course {self.slug}: missing prebuilt source {self.html_dir}")
            return

        files = sorted(p for p in directory.glob("*.md") if re.match(r"\d\d-", p.name))
        if not files:
            raise SystemExit(f"course {self.slug}: no NN-*.md modules found")
        self.pages = [
            Page(p, self._url_for(p), "Module", self._track_of(int(re.match(r"(\d+)", p.stem).group(1))))
            for p in files
        ]
        # module 00 is the course index; it carries the course title, not a module number
        self.index = self.pages[0]
        self.index.eyebrow = "Course"
        self.modules = self.pages[1:]

    def _url_for(self, path: Path) -> str:
        num = int(re.match(r"(\d+)", path.stem).group(1))
        return self.base if num == 0 else f"{self.base}{path.stem}/"

    def _track_of(self, num: int) -> str:
        for t in self.tracks:
            if t["from"] <= num <= t["to"]:
                return t["key"]
        return "foundations"


class Site:
    def __init__(self):
        cfg = yaml.safe_load((CONTENT / "site.yml").read_text(encoding="utf-8"))
        self.title = cfg["title"]
        self.tagline = cfg["tagline"]
        self.domain = cfg["domain"]
        self.repo = cfg["repo"]
        self.nav = cfg.get("nav", [])
        self.courses = sorted(
            (Course(d) for d in sorted((CONTENT / "courses").iterdir()) if d.is_dir()),
            key=lambda c: (c.order, c.title))
        # Notes are two levels: a category is content/notes/<slug>.md, and its
        # topics live in content/notes/<slug>/*.md. A topic is a page inside its
        # category, never a sibling of it.
        notes_dir = CONTENT / "notes"
        self.notes: list[Page] = []
        if notes_dir.is_dir():
            self.notes = sorted(
                (Page(p, f"/notes/{p.stem}/", "Notes") for p in sorted(notes_dir.glob("*.md"))
                 if not p.stem.startswith("_")),
                key=lambda p: (p.order, p.title),
            )
            for cat in self.notes:
                topic_dir = notes_dir / cat.path.stem
                cat.topics = sorted(
                    (Page(t, f"/notes/{cat.path.stem}/{t.stem}/", cat.title)
                     for t in sorted(topic_dir.glob("*.md")) if not t.stem.startswith("_")),
                    key=lambda p: (p.order, p.title),
                ) if topic_dir.is_dir() else []
                for t in cat.topics:
                    t.parent = cat

    @property
    def note_topics(self) -> list[Page]:
        return [t for c in self.notes for t in c.topics]

    def url_map(self) -> dict[Path, str]:
        """Source path -> URL, for rewriting inter-page markdown links."""
        m = {}
        for c in self.courses:
            for p in c.pages:
                m[p.path.resolve()] = p.url
        for p in self.notes + self.note_topics:
            m[p.path.resolve()] = p.url
        return m



# ---------------------------------------------------------------- chrome

def topbar(site: Site, active: str = "") -> str:
    links = "\n".join(
        f'<a class="topbar__l{" is-active" if l["href"] == active else ""}" '
        f'href="{l["href"]}">{html.escape(l["label"])}</a>'
        for l in site.nav
    )
    return f"""<header class="topbar">
  <div class="topbar__in">
    <a class="topbar__brand" href="/">{html.escape(site.title)}</a>
    <nav class="topbar__nav">{links}</nav>
  </div>
</header>"""


def document(site: Site, *, title: str, description: str, body: str,
             track: str = "foundations", scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(description)}">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;9..40,500;9..40,700&family=Fira+Code:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" integrity="sha384-nB0miv6/jRmo5UMMR1wu3Gz6NLsoTkbqJghGIsx//Rlm+ZU03BU6SQNC66uf4l5+" crossorigin="anonymous">
<link rel="stylesheet" href="/assets/style.css">
<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">
</head>
<body data-track="{track}">
{body}
{scripts}
</body>
</html>
"""


MATH_SCRIPTS = """<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js" integrity="sha384-7zkQWkzuo3B5mTepMUcHkMB5jZaolc2xDwL6VFqjFALcbeS9Ggm/Yr2r3Dy4lfFg" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js" integrity="sha384-43gviWU0YVjaDtb/GhzOouOXtZMP/7XUzwPTstBeZFe/+rCMvRwr4yROQP43s0Xk" crossorigin="anonymous"></script>
<script src="/assets/katex-init.js"></script>
<script type="module" src="/assets/mermaid.js"></script>
<script src="/assets/page.js"></script>
<script src="/assets/rail-scroll.js" defer></script>"""


# ---------------------------------------------------------------- render

def render_module(site: Site, course: Course, page: Page, md: MarkdownIt,
                  urls: dict[Path, str]) -> tuple[str, dict]:
    env = {"src_dir": page.src_dir, "urls": urls, "broken": [], "diagrams": []}
    sections, rendered, toc = split_sections(page.body), [], []
    intro_html = ""
    i = 0
    for title, chunk in sections:
        chunk = chunk.strip()
        if title is None:
            if chunk:
                intro_html = md.render(chunk, env)
            continue
        kind = section_kind(title)
        sid = slugify(title)
        toc.append((sid, title))
        i += 1
        rendered.append(
            f'<section class="sec sec--{kind}" id="{sid}" style="--i:{i}">'
            f'<div class="sec__h"><span class="sec__n">{i:02d}</span>'
            f'<h2 class="sec__t">{md.renderInline(title, env)}</h2></div>'
            f'<div class="sec__body">{md.render(chunk, env)}</div>'
            f"</section>"
        )

    prereq_html = f'<div class="prereq">{md.render(page.prereq, env)}</div>' if page.prereq else ""

    # left rail: every module in the course, grouped by track
    rail = []
    for t in course.tracks:
        members = [p for p in course.pages if t["from"] <= p.num <= t["to"]]
        if not members:
            continue
        rail.append(f'<div class="nav__grp">{html.escape(t["label"])}</div>')
        for p in members:
            cur = " is-current" if p is page else ""
            rail.append(
                f'<a class="nav__i{cur}" href="{p.url}">'
                f'<span class="nav__n">{p.num:02d}</span>'
                f"<span>{html.escape(p.short)}</span></a>"
            )

    toc_html = ""
    if len(toc) > 1:
        links = "\n".join(
            f'<a href="#{sid}">{html.escape(re.sub(r"[`*]", "", t))}</a>' for sid, t in toc)
        toc_html = f'<aside class="toc" id="toc"><div class="toc__h">On this page</div>{links}</aside>'

    idx = course.pages.index(page)
    prev_p = course.pages[idx - 1] if idx > 0 else None
    next_p = course.pages[idx + 1] if idx < len(course.pages) - 1 else None
    prev_html = (
        f'<a class="pager__prev" href="{prev_p.url}"><div class="pager__k">&larr; Previous</div>'
        f'<div class="pager__t">{html.escape(prev_p.title)}</div></a>'
        if prev_p else '<div class="pager__spacer"></div>'
    )
    next_html = (
        f'<a class="pager__next" href="{next_p.url}"><div class="pager__k">Next &rarr;</div>'
        f'<div class="pager__t">{html.escape(next_p.title)}</div></a>'
        if next_p else '<div class="pager__spacer"></div>'
    )

    body = f"""{topbar(site, "/courses/")}
<div class="shell">
  <nav class="nav" id="nav">
    <a class="nav__brand" href="{course.base}">
      <span>
        <span class="nav__brand-k">{html.escape(course.brand_kicker)}</span>
        <span class="nav__brand-t">{html.escape(course.brand_title)}</span>
      </span>
      <button class="navtoggle" id="navtoggle" type="button" aria-expanded="false">{html.escape(course.modules_label)}</button>
    </a>
    <div class="nav__list">
{chr(10).join(rail)}
    </div>
  </nav>

  <main class="main">
    <header class="hd">
      <div class="hd__k">{html.escape(page.eyebrow)}</div>
      <h1>{html.escape(page.title)}</h1>
      {prereq_html}
    </header>

    {intro_html}
    {"".join(rendered)}

    <nav class="pager">
      {prev_html}
      {next_html}
    </nav>

    <footer class="foot">{course.footer_html}</footer>
  </main>

  {toc_html}
</div>"""

    doc_title = (
        f"{page.title} — {course.title}" if page is not course.index
        else f"{course.title} — {site.title}"
    )
    return document(site, title=doc_title, description=f"{page.title} — {course.title}",
                    body=body, track=page.track, scripts=MATH_SCRIPTS), env


def render_note(site: Site, page: Page, md: MarkdownIt, urls: dict[Path, str]) -> tuple[str, dict]:
    env = {"src_dir": page.src_dir, "urls": urls, "broken": [], "diagrams": []}
    body_html = md.render(page.body, env)

    # A topic links back to the category it belongs to.
    crumb = (
        f'<a class="crumb" href="{page.parent.url}">&larr; {html.escape(page.parent.title)}</a>'
        if page.parent else ""
    )

    # A category page's own topic table is the index — the build does not add a
    # second, generated list of the same topics underneath it.

    body = f"""{topbar(site, "/notes/")}
<div class="shell shell--plain">
  <main class="main">
    <header class="hd">
      <div class="hd__k">{html.escape(page.eyebrow)}</div>
      {crumb}
      <h1>{html.escape(page.title)}</h1>
    </header>
    <div class="sec__body">{body_html}</div>
  </main>
</div>"""
    # Pages may pull in extra scripts (interactive widgets) via front matter.
    extra = "".join(
        f'\n<script src="{html.escape(s, quote=True)}" defer></script>'
        for s in page.meta.get("scripts", [])
    )
    return document(site, title=f"{page.title} — {site.title}",
                    description=page.description or page.title,
                    body=body, scripts=MATH_SCRIPTS + extra), env


def course_meta(c: "Course") -> str:
    n = c.module_count or len(c.modules)
    unit = "chapters" if c.prebuilt else "modules"
    return f"{n} {unit} · {c.level}" if c.level else f"{n} {unit}"


def card(href: str, kicker: str, title: str, blurb: str, meta: str = "") -> str:
    meta_html = f'<div class="card__meta">{html.escape(meta)}</div>' if meta else ""
    return (
        f'<a class="card" href="{href}">'
        f'<div class="card__k">{html.escape(kicker)}</div>'
        f'<div class="card__t">{html.escape(title)}</div>'
        f'<p class="card__b">{html.escape(blurb)}</p>'
        f"{meta_html}</a>"
    )


def read_labs(root: Path) -> list[tuple[Path, str]]:
    """(page path, title) for each lab, in the order LABS.html lists them.

    LABS.html is the source of truth rather than the directory listing, which
    also contains a _shared/ helper directory that is not a lab.
    """
    index = root / "LABS.html"
    if not index.is_file():
        return []
    text = index.read_text(encoding="utf-8")
    out = []
    for href in dict.fromkeys(re.findall(r'href="(labs/[^"]+README\.html)"', text)):
        page = root / href
        if not page.is_file():
            continue
        m = re.search(r"<title>(.*?)</title>", page.read_text(encoding="utf-8"), re.S)
        title = m.group(1).split("—")[0].strip() if m else page.parent.name
        out.append((page, title))
    return out


def inject_lab_nav(text: str, page: Path, dest: Path,
                   labs: list[tuple[Path, str]]) -> str:
    """Append a Labs group to the end of the course's own left rail.

    The rail links relatively and every page sits at a different depth, so each
    href is computed from that page's directory rather than assumed.
    """
    if not labs:
        return text
    # Anchor on the COURSE rail, not just any <nav>: the grafted-on site top bar
    # contains a <nav class="topbar__nav"> that appears earlier in the document,
    # and searching from there finds the wrong closing tag.
    m = re.search(r'<nav class="nav"[^>]*>', text)
    if not m:
        return text
    nav_end = text.find("</nav>", m.end())
    if nav_end == -1:
        return text
    close = text.rfind("</div>", m.end(), nav_end)   # closes nav__list
    if close == -1:
        return text

    rows = [f'<div class="nav__grp">Labs<span>{len(labs)}</span></div>']
    for i, (lab_page, title) in enumerate(labs, 1):
        href = os.path.relpath(lab_page, page.parent).replace(os.sep, "/")
        cur = " is-current" if lab_page.resolve() == page.resolve() else ""
        rows.append(
            f'<a class="nav__i{cur}" href="{href}">'
            f'<span class="nav__n">{i:02d}</span><span>{title}</span></a>'
        )
    return text[:close] + "\n" + "\n".join(rows) + "\n" + text[close:]


def render_prebuilt(site: Site, course: Course) -> int:
    """Copy a prebuilt HTML course into the site and graft the site chrome on.

    The content is never parsed or rewritten — only two injections happen per
    page: a stylesheet link that re-points the course's CSS custom properties at
    the site palette, and the shared top bar. Both go in as late as possible so
    the page's own inline <style> loses the cascade to ours.
    """
    dest = OUT / course.base.strip("/")
    shutil.copytree(course.html_dir, dest,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))

    bar = topbar(site, "/courses/")
    bridge_link = ('<link rel="stylesheet" href="/assets/prebuilt-bridge.css">\n'
                   '<link rel="icon" type="image/svg+xml" href="/assets/favicon.svg">\n'
                   '<script src="/assets/rail-scroll.js" defer></script>\n</head>')
    labs = read_labs(dest)
    touched = 0
    for page in sorted(dest.rglob("*.html")):
        text = page.read_text(encoding="utf-8")
        if "prebuilt-bridge.css" in text:
            continue
        head_end = text.find("</head>")
        if head_end == -1:
            continue
        # Anchor the body search AFTER </head>. These pages inline their CSS, and
        # one of the comments in it reads "overridden by <body data-track>" — a
        # naive search matches that comment and buries the bar inside <head>,
        # where the browser silently discards it.
        m = re.search(r"<body[^>]*>", text[head_end:])
        if not m:
            continue
        body_end = head_end + m.end()
        text = (text[:head_end] + bridge_link
                + text[head_end + len("</head>"):body_end] + "\n" + bar + text[body_end:])
        text = inject_lab_nav(text, page, dest, labs)
        page.write_text(text, encoding="utf-8")
        touched += 1
    return touched


STATUS_LABEL = {"done": "Done", "progress": "In progress", "planned": "Planned"}


def render_status(site: Site, roadmap_file: Path) -> str:
    """Render the build-status checklist from roadmap.yml."""
    cfg = yaml.safe_load(roadmap_file.read_text(encoding="utf-8"))
    all_items = [i for s in cfg["sections"] for i in s["items"]]
    done = sum(1 for i in all_items if i["status"] == "done")

    def bar(items: list) -> str:
        n = len(items)
        d = sum(1 for i in items if i["status"] == "done")
        p = sum(1 for i in items if i["status"] == "progress")
        pct = round(100 * d / n) if n else 0
        return (
            f'<div class="prog"><div class="prog__track">'
            f'<span class="prog__fill" style="width:{pct}%"></span>'
            f'<span class="prog__fill prog__fill--p" style="width:{round(100 * p / n) if n else 0}%"></span>'
            f'</div><div class="prog__n">{d} of {n} done</div></div>'
        )

    blocks = []
    for section in cfg["sections"]:
        rows = "\n".join(
            f'<li class="chk chk--{i["status"]}">'
            f'<span class="chk__mark" aria-hidden="true"></span>'
            f'<div class="chk__body"><div class="chk__t">{html.escape(i["title"])}'
            f'<span class="chk__s">{STATUS_LABEL[i["status"]]}</span></div>'
            f'<p class="chk__d">{html.escape(" ".join(i.get("detail", "").split()))}</p></div></li>'
            for i in section["items"]
        )
        blurb = f'<p class="lp__p">{html.escape(" ".join(section.get("blurb", "").split()))}</p>' if section.get("blurb") else ""
        blocks.append(
            f'<section class="lp__sec"><h2 class="lp__h">{html.escape(section["name"])}</h2>'
            f'{blurb}{bar(section["items"])}<ul class="chks">{rows}</ul></section>'
        )

    body = f"""{topbar(site, "/status/")}
<div class="shell shell--plain">
  <main class="main main--wide">
    <header class="hd">
      <div class="hd__k">{html.escape(cfg.get("title", "Build status"))}</div>
      <h1>{html.escape(cfg.get("title", "Build status"))}</h1>
      <p class="lede">{html.escape(" ".join(cfg.get("lede", "").split()))}</p>
      <div class="tally"><strong>{done}</strong> of <strong>{len(all_items)}</strong> tracked items complete</div>
    </header>
    {"".join(blocks)}
  </main>
</div>"""
    return document(site, title=f'{cfg.get("title", "Build status")} — {site.title}',
                    description=" ".join(cfg.get("lede", "").split()), body=body)


def render_landing(site: Site, *, active: str, kicker: str, title: str, lede: str,
                   sections: list[tuple[str, str]], hero: bool = False) -> str:
    blocks = "\n".join(
        f'<section class="lp__sec"><h2 class="lp__h">{html.escape(h)}</h2>{c}</section>'
        for h, c in sections
    )
    hero_cls = " hd--hero" if hero else ""
    body = f"""{topbar(site, active)}
<div class="shell shell--plain">
  <main class="main main--wide">
    <header class="hd{hero_cls}">
      <div class="hd__k">{html.escape(kicker)}</div>
      <h1>{html.escape(title)}</h1>
      <p class="lede">{html.escape(lede)}</p>
    </header>
    {blocks}
  </main>
</div>"""
    return document(site, title=title if title == site.title else f"{title} — {site.title}",
                    description=lede, body=body)


# ---------------------------------------------------------------- main

def build() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    site = Site()
    urls = site.url_map()
    md = build_md()
    broken: list[str] = []
    pages_written = 0
    diagrams = 0

    # ---- courses
    for course in site.courses:
        if course.prebuilt:
            n = render_prebuilt(site, course)
            pages_written += n
            print(f"  {course.base:<58} {n:>4} prebuilt pages")
            continue
        for page in course.pages:
            doc, env = render_module(site, course, page, md, urls)
            write(page.out_path, doc)
            broken += [f"{page.path.name}: {b}" for b in env["broken"]]
            diagrams += doc.count('class="diagram-shell"')
            pages_written += 1
            print(f"  {page.url:<58} {len(doc)//1024:>4} KB")
        code_dir = course.dir / "code"
        if code_dir.is_dir():
            shutil.copytree(code_dir, OUT / course.base.strip("/") / "code",
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    # ---- notes: categories, then the topics inside each
    for page in site.notes + site.note_topics:
        doc, env = render_note(site, page, md, urls)
        write(page.out_path, doc)
        broken += [f"{page.path.name}: {b}" for b in env["broken"]]
        pages_written += 1

    # ---- courses index
    course_cards = "\n".join(
        card(c.base, "Course", f"{c.title}{f' — {c.subtitle}' if c.subtitle else ''}",
             c.blurb, course_meta(c))
        for c in site.courses
    )
    write(OUT / "courses" / "index.html", render_landing(
        site, active="/courses/", kicker="Courses", title="Courses",
        lede="Long-form, sequential courses that build a topic from first principles.",
        sections=[("Available now", f'<div class="cards">{course_cards}</div>')],
    ))
    pages_written += 1

    # ---- notes index
    if site.notes:
        note_cards = "\n".join(
            card(p.url, "Notes", p.title, p.description or "Notes in progress.",
                 p.meta.get("meta", ""))
            for p in site.notes          # categories only — topics live inside them
        )
        write(OUT / "notes" / "index.html", render_landing(
            site, active="/notes/", kicker="Notes", title="Notes",
            lede="Topic-by-topic reference notes. Contributions welcome — each page is a Markdown file.",
            sections=[("Topics", f'<div class="cards">{note_cards}</div>')],
        ))
        pages_written += 1

    # ---- build status / roadmap
    roadmap_file = CONTENT / "roadmap.yml"
    if roadmap_file.is_file():
        write(OUT / "status" / "index.html", render_status(site, roadmap_file))
        pages_written += 1

    # ---- home
    home_sections = [("Courses", f'<div class="cards">{course_cards}</div>')]
    if site.notes:
        home_sections.append(("Notes", f'<div class="cards">{note_cards}</div>'))
    home_sections.append((
        "Contributing",
        '<p class="lp__p">Every page on this site is a Markdown file in the repository. '
        f'Add or improve one and open a pull request: <a href="{site.repo}" target="_blank" '
        f'rel="noopener">{html.escape(site.repo)}</a>.</p>',
    ))
    write(OUT / "index.html", render_landing(
        site, active="/", kicker="Open source", title=site.title, lede=site.tagline,
        sections=home_sections, hero=True,
    ))
    pages_written += 1

    # ---- assets and Pages plumbing
    assets = OUT / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    # style.css is concatenated from every *.css in the theme, base first, so
    # separate concerns can live in separate files without an @import round trip.
    css_files = [THEME / "style.css"] + sorted(
        f for f in THEME.glob("*.css") if f.name != "style.css")
    (assets / "style.css").write_text(
        "\n".join(f.read_text(encoding="utf-8") for f in css_files), encoding="utf-8")
    for name in ("page.js", "mermaid.js", "katex-init.js", "favicon.svg", "rail-scroll.js"):
        shutil.copyfile(THEME / name, assets / name)
    page_assets = CONTENT / "notes" / "_assets"
    if page_assets.is_dir():
        dest = assets / "pages"
        dest.mkdir(parents=True, exist_ok=True)
        for f in page_assets.glob("*.js"):
            shutil.copyfile(f, dest / f.name)

    bridge_src = THEME / "prebuilt" / "bridge.css"
    if bridge_src.is_file():
        shutil.copyfile(bridge_src, assets / "prebuilt-bridge.css")

    (OUT / "CNAME").write_text(site.domain + "\n", encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")

    print(f"\n{pages_written} pages, {diagrams} diagrams -> {OUT}")
    if broken:
        print(f"\nBROKEN LINKS ({len(broken)}):", file=sys.stderr)
        for b in broken:
            print(f"  {b}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", action="store_true", help="serve _site on :8000 after building")
    args = ap.parse_args()

    rc = build()
    if rc == 0 and args.serve:
        import functools
        import http.server
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(OUT))
        print("serving http://localhost:8000 — Ctrl+C to stop")
        http.server.ThreadingHTTPServer(("", 8000), handler).serve_forever()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
