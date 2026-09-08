# ML Interview Notes

Open-source notes for machine learning interviews — live at
**[mlinterviewnotes.com](https://mlinterviewnotes.com)**.

The site is built from Markdown notes and courses, plus an imported HTML book.
Adding or improving a page is a pull request.

## What's here

| Section | Contents |
|---|---|
| [Courses](https://mlinterviewnotes.com/courses/) | Long-form, sequential material. **Transformers Deep Dive** — 17 modules from "why did we abandon RNNs?" to the configuration choices in production 2026 models. **The Inference Engineering Course** — 14 chapters on how LLM serving works, read out of the vLLM and SGLang source. |
| [Notes](https://mlinterviewnotes.com/notes/) | Topic-by-topic reference notes: Math, Libraries, ML, Deep Learning, NLP. In progress — contributions welcome. |

## Repository layout

```
content/
  site.yml                      site title, tagline, top nav
  roadmap.yml                   drives the /status/ page
  courses/<slug>/course.yml     course title, blurb, track definitions
  courses/<slug>/NN-*.md        course modules; 00 becomes the course index
  courses/<slug>/html/          a prebuilt course, with shared reader chrome
  notes/<slug>.md               a note category
  notes/<slug>/<topic>.md       a topic inside that category
site/
  build.py                      the static site generator
  presentation.py               shared navigation, reader controls, search index
  test_build.py                 build verification, runs in CI
  test_presentation.py          all-page links and imported-content preservation
  test_browser.py               optional desktop/mobile interaction checks
  theme/                        stylesheet, scripts, favicon
legacy/                         the retired Django app (see legacy/README.md)
```

## Building locally

```bash
pip install -r site/requirements.txt
python site/build.py --serve      # builds to _site/ and serves on :8000
```

`_site/` is generated and gitignored — never edit it by hand. To check your work
the way CI does:

```bash
python site/build.py
python site/test_build.py
python site/test_presentation.py
python site/test_curriculum_examples.py --check-only --published
```

The verification passes check page coverage, internal links and fragments,
diagram containers, math markup, search entries, and exact preservation of the
imported book's lesson text, code, and diagram sources. Actual runtime rendering
and interactions are checked separately in a browser.

ML and DL examples marked with a `python runnable` fence are independent CPU
programs. The build publishes a downloadable `.py` file from that same fence;
there is no second copy to keep in sync. Ordinary `python` fences may be clearly
labelled fragments and are not executed automatically. To run the full example
suite in a Python 3.11 virtual environment:

```bash
pip install -r site/requirements.txt -r site/requirements-examples.txt -r site/requirements-boosters.txt
python site/test_curriculum_examples.py --include-boosters --published --report /tmp/curriculum-examples.json
```

On Linux, install `torch==2.8.0` from the PyTorch CPU wheel index before the
requirements if no accelerator runtime is wanted. Each example runs in a fresh
process and temporary directory with a timeout. The report records outputs,
package versions, failures, and explicitly skipped optional boosters. CI runs
the complete CPU suite separately and requires it before deployment. This does
not validate distributed/GPU fragments or full production workloads.

For browser checks, install the optional test tooling and leave the preview
server running in another terminal:

```bash
pip install playwright
playwright install chromium
python site/test_browser.py
```

Screenshots are written to `/tmp/ml-notes-design` by default. The test covers
desktop and mobile layouts, search, saved pages, focus mode, theme and text-size
preferences, interactive figures, rendered math and diagrams, and code copying.

## Reading experience

All courses, reference notes, and labs share the same reading layout. Search is
client-side and includes full lesson text. Bookmarks, reading position, theme,
text size, and focus preferences stay in the reader's browser using local storage;
there is no account or tracking service. Existing KaTeX, Mermaid, and font assets
still load from their external providers.

Course covers are checked-in WebP diagrams. Regenerate them with
`python site/generate_covers.py` when needed (requires Pillow, not a build
dependency). Shared reader styles and scripts live in `site/theme/`.

## Writing content

Markdown, with a few conveniences the build understands:

- **Math** — `$inline$` and `$$display$$`, rendered with KaTeX.
- **Diagrams** — ` ```mermaid ` fences become pan/zoomable diagrams.
- **Tables, code fences, and blockquotes** get styled treatments; a blockquote at
  the top of a course module becomes its prerequisites box.
- **Links between pages** — link to the other file (`[text](./05-positional-encodings.md)`)
  and the build rewrites it to the right URL.

Course modules are split into sections on `## ` headings, which drive the
right-hand table of contents. Sections titled `Key takeaways`, `Self-check`, or
`Reconciling…` get their own visual treatment.

Note pages accept optional YAML front matter:

```yaml
---
order: 1
description: One-sentence blurb used on cards and in meta tags.
meta: 5 topics planned
---
```

### Prebuilt courses

A course that already exists as finished HTML can be published without being
converted to Markdown. Set `type: prebuilt` in its `course.yml` and put the pages
under `html/`. The build copies the book, applies the shared palette, and uses
BeautifulSoup to add navigation, reading controls, search, and a footer. Embedded
navigation and diagram runtimes are replaced with the shared scripts. Lesson
text, code examples, and diagram sources are preserved and compared against the
original HTML by `site/test_presentation.py`. Edit the book in its source HTML;
it cannot be edited as Markdown until someone converts it.

### Adding a course

Create `content/courses/<slug>/` with a `course.yml` and `NN-*.md` modules. The
home page, the courses index, and the module rail all pick it up automatically —
no code change needed.

## Deployment

Pushes to `main` trigger `.github/workflows/pages.yml`, which builds, runs the
verification pass, and publishes to GitHub Pages. Pull requests run the same
build and verification without deploying.

## License

Content and code are open source. Contributions are welcome — open an issue or a
pull request.
