"""Shared reading experience for generated Markdown and the imported HTML book."""

import html
import json
import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup


def fragment(markup):
    return BeautifulSoup(markup, "html.parser")


def icon(name):
    return f'<i data-lucide="{name}" aria-hidden="true"></i>'


def library_nav(site, active=""):
    parts = ['<aside class="library-nav" aria-label="Library"><div class="library-nav__label">Library</div>']
    for url, name, glyph in [("/", "Overview", "library"), ("/courses/", "Courses", "book-open"),
                             ("/notes/", "Notes", "notebook-pen")]:
        state = ' aria-current="page"' if url == active else ""
        parts.append(f'<a href="{url}"{state}>{icon(glyph)}{name}</a>')
    parts.append('<div class="library-nav__label">Explore topics</div>')
    for p in site.notes:
        parts.append(f'<a href="{p.url}"><span class="nav-dot nav-dot--{p.path.stem}"></span>{html.escape(p.title)}<span class="nav-count">{len(p.topics)}</span></a>')
    parts.append(f'<a class="library-nav__source" href="{site.repo}" target="_blank" rel="noopener">{icon("github")}Open source{icon("arrow-up-right")}</a></aside>')
    return "".join(parts)


def notes_nav(site, page):
    category = page.parent or page
    parts = [f'<nav class="nav" id="nav" aria-label="Topic navigation"><div class="nav__heading"><a class="nav__brand" href="{category.url}"><span class="nav__brand-k">Reference notes</span><span class="nav__brand-t">{html.escape(category.title)}</span></a><button class="navtoggle" id="navtoggle" aria-expanded="false" aria-controls="note-nav-list" type="button">{icon("list")}<span>Topics</span></button></div><div class="nav__list" id="note-nav-list"><a class="nav__back" href="/notes/">{icon("arrow-left")}All notes</a>']
    for cat in site.notes:
        parts.append(f'<div class="nav__grp"><a href="{cat.url}">{html.escape(cat.title)}</a></div>')
        if cat is category:
            for i, topic in enumerate(cat.topics, 1):
                cur = ' is-current' if topic is page else ''
                parts.append(f'<a class="nav__i{cur}" href="{topic.url}"><span class="nav__n">{i:02d}</span><span>{html.escape(topic.title.split(":")[0])}</span></a>')
    return "".join(parts) + "</div></nav>"


DIALOGS = """
<dialog class="search-dialog" id="search-dialog" aria-labelledby="search-title">
  <div class="dialog-heading"><h2 id="search-title">Explore the library</h2><button class="icon-button" data-close-dialog type="button" aria-label="Close search" title="Close search"><i data-lucide="x"></i></button></div>
  <div class="search-field"><i data-lucide="search"></i><input id="search-input" type="search" placeholder="Search concepts, courses, and notes..." autocomplete="off" aria-label="Search the library"></div>
  <div class="search-filters" role="group" aria-label="Filter search results">
    <button type="button" data-filter="all" aria-pressed="true">All</button><button type="button" data-filter="courses" aria-pressed="false">Courses</button><button type="button" data-filter="notes" aria-pressed="false">Notes</button><button type="button" data-filter="labs" aria-pressed="false">Labs</button><button type="button" data-filter="saved" aria-pressed="false"><i data-lucide="bookmark"></i>Saved</button>
  </div>
  <p class="search-status" id="search-status" role="status"></p><div class="search-results" id="search-results"></div>
</dialog>
<dialog class="settings-dialog" id="settings-dialog" aria-labelledby="settings-title">
  <div class="dialog-heading"><h2 id="settings-title">Reading preferences</h2><button class="icon-button" data-close-dialog type="button" aria-label="Close preferences" title="Close preferences"><i data-lucide="x"></i></button></div>
  <div class="setting"><span id="theme-label">Appearance</span><div class="segmented" role="group" aria-labelledby="theme-label"><button type="button" data-theme-choice="light" aria-label="Light appearance" title="Light appearance"><i data-lucide="sun"></i></button><button type="button" data-theme-choice="dark" aria-label="Dark appearance" title="Dark appearance"><i data-lucide="moon"></i></button><button type="button" data-theme-choice="system" aria-label="System appearance" title="System appearance"><i data-lucide="monitor"></i></button></div></div>
  <div class="setting"><label for="reading-size">Text size</label><output id="size-output" for="reading-size">18 px</output></div><input id="reading-size" type="range" min="16" max="22" step="1" value="18">
  <div class="setting"><label for="focus-setting">Focus mode</label><input id="focus-setting" class="switch" type="checkbox" role="switch"></div>
  <button type="button" class="reset-preferences" id="reset-preferences"><i data-lucide="rotate-ccw"></i>Reset preferences</button>
</dialog>
"""


def enhance_site(out: Path, theme: Path, site):
    assets = out / "assets"
    for name in ("reader.css", "reader.js", "appearance.js", "lucide.min.js", "lucide.LICENSE"):
        shutil.copyfile(theme / name, assets / name)
    shutil.copytree(theme / "covers", assets / "covers")
    note_pages = {p.url: p for p in site.notes + site.note_topics}
    search = []
    for path in sorted(out.rglob("*.html")):
        relative = path.relative_to(out).as_posix()
        url = "/" + relative.removesuffix("index.html") if path.name == "index.html" else "/" + relative
        soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")
        main = soup.find("main")
        if main is None:
            continue
        main["id"] = "main-content"
        is_article = url in note_pages or url.startswith("/courses/") and url != "/courses/"
        soup.body["class"] = soup.body.get("class", []) + (["is-reading"] if is_article else ["is-library"])
        soup.head.insert(0, fragment('<script src="/assets/appearance.js"></script>'))
        soup.head.append(fragment('<link rel="stylesheet" href="/assets/reader.css">'))
        soup.head.append(fragment(f'<link rel="canonical" href="https://{site.domain}{url}">'))
        soup.body.insert(0, fragment('<a class="skip-link" href="#main-content">Skip to content</a><div class="reading-progress" aria-hidden="true"></div>'))
        shell = soup.select_one(".shell")

        # The imported book embeds copies of its old runtime. Replace those
        # copies with the shared implementation while retaining every lesson.
        if "/inference/" in url:
            for script in list(soup.find_all("script")):
                text = script.string or ""
                if script.get("type") == "module" and "mermaid.render" in text:
                    script.clear()
                    script["src"] = "/assets/mermaid.js"
                elif "navtoggle" in text and not script.get("type"):
                    script.decompose()
            soup.body.append(fragment('<script src="/assets/page.js" defer></script>'))

        if url in note_pages:
            shell["class"] = ["shell"]
            shell.insert(0, fragment(notes_nav(site, note_pages[url])))
        elif not is_article:
            shell["class"] = ["shell", "library-shell"]
            shell.insert(0, fragment(library_nav(site, url)))

        # Move the mobile toggle out of its link: expanding a rail must not
        # navigate back to the course index.
        toggle = soup.select_one("#navtoggle")
        if toggle and toggle.find_parent("a"):
            brand = toggle.find_parent("a")
            toggle.extract()
            heading = soup.new_tag("div", attrs={"class": "nav__heading"})
            brand.wrap(heading)
            heading.append(toggle)
            toggle.clear()
            toggle.append(fragment(icon("list") + "<span>Contents</span>"))
            nav_list = soup.select_one(".nav__list")
            nav_list["id"] = "course-nav-list"
            toggle["aria-controls"] = "course-nav-list"

        if is_article:
            toc = soup.select_one("#toc")
            headings = main.find_all("h2")
            if not toc and headings:
                toc = soup.new_tag("aside", attrs={"class": "toc", "id": "toc", "aria-label": "On this page"})
                toc.append(fragment('<div class="toc__h">On this page</div>'))
                used = {tag.get("id") for tag in soup.select("[id]")}
                for h in headings:
                    base = re.sub(r"[^\w]+", "-", h.get_text().lower()).strip("-") or "section"
                    sid = h.get("id") or base
                    n = 2
                    while not h.get("id") and sid in used:
                        sid, n = f"{base}-{n}", n + 1
                    h["id"] = sid
                    used.add(sid)
                    a = soup.new_tag("a", href="#" + sid)
                    a.string = h.get_text(" ", strip=True)
                    toc.append(a)
                shell.append(toc)
            words = len(main.get_text(" ", strip=True).split())
            minutes = max(1, round(words / 200))
            category = (note_pages[url].parent or note_pages[url]).title if url in note_pages else next((c.title for c in site.courses if url.startswith(c.base)), "Course")
            tools = f'<div class="reader-tools"><span class="reading-meta">{minutes} min read<span class="meta-dot"></span>{len(headings)} sections</span><div class="reader-actions"><button class="icon-button" type="button" data-bookmark aria-pressed="false" title="Save this page" aria-label="Save this page">{icon("bookmark")}</button><button class="icon-button" type="button" data-focus aria-pressed="false" title="Focus mode" aria-label="Focus mode">{icon("scan")}</button><button class="icon-button" type="button" data-print title="Print page" aria-label="Print page">{icon("printer")}</button></div></div>'
            main.insert(0, fragment(tools))
            if toc:
                toc.append(fragment('<div class="toc__end"><a href="#main-content">Back to top ' + icon("arrow-up") + '</a></div>'))
            if url in note_pages and note_pages[url].parent:
                page = note_pages[url]
                siblings = page.parent.topics
                index = siblings.index(page)
                prev = siblings[index - 1] if index else None
                nxt = siblings[index + 1] if index + 1 < len(siblings) else None
                pager = '<nav class="pager" aria-label="Related topics">'
                for target, label in ((prev, "Previous topic"), (nxt, "Next topic")):
                    pager += (f'<a href="{target.url}"><div class="pager__k">{label}</div><div class="pager__t">{html.escape(target.title)}</div></a>' if target else '<div class="pager__spacer"></div>')
                main.append(fragment(pager + "</nav>"))
        else:
            category = "Library"

        # Index the authored text before adding global dialogs and footer.
        article = BeautifulSoup(str(main), "html.parser")
        for tag in article.select("script, .reader-tools, .pager, .foot"):
            tag.decompose()
        title = main.find("h1").get_text(" ", strip=True)
        if title == "Front door":
            title = "The Inference Engineering Course"
        description = soup.find("meta", attrs={"name": "description"})
        if not (url.startswith("/courses/inference/") and path.name == "README.html" and path.parent.name == "inference"):
            search.append({"url": url, "title": title, "category": category,
                           "kind": "labs" if "/labs/" in url else "notes" if url.startswith("/notes/") else "courses" if url.startswith("/courses/") else "library",
                           "description": description.get("content", "") if description else "",
                           "text": " ".join(article.get_text(" ", strip=True).split())})
        soup.body.append(fragment(f'<footer class="site-footer"><a href="/">ML Interview Notes<span class="footer-dot">.</span></a><span>Learn deeply. Build thoughtfully.</span><a href="{site.repo}" target="_blank" rel="noopener">{icon("github")}GitHub</a></footer>'))
        soup.body.append(fragment(DIALOGS))
        soup.body.append(fragment('<div id="reader-announcement" class="sr-only" role="status"></div><script src="/assets/lucide.min.js" defer></script><script src="/assets/reader.js" defer></script>'))
        path.write_text(str(soup), encoding="utf-8")
    (assets / "search.json").write_text(json.dumps(search, ensure_ascii=False), encoding="utf-8")
