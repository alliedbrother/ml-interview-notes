# ML Interview Notes

Open-source notes for machine learning interviews — live at
**[mlinterviewnotes.com](https://mlinterviewnotes.com)**.

Every page on the site is a Markdown file in this repository. Adding or improving
a page is a pull request.

## What's here

| Section | Contents |
|---|---|
| [Courses](https://mlinterviewnotes.com/courses/) | Long-form, sequential material. **Transformers Deep Dive** — 17 modules from "why did we abandon RNNs?" to the configuration choices in production 2026 models. **The Inference Engineering Book** — 14 chapters on how LLM serving works, read out of the vLLM and SGLang source. |
| [Notes](https://mlinterviewnotes.com/notes/) | Topic-by-topic reference notes: Math, Libraries, ML, Deep Learning, NLP. In progress — contributions welcome. |

## Repository layout

```
content/
  site.yml                      site title, tagline, top nav
  roadmap.yml                   drives the /status/ page
  courses/<slug>/course.yml     course title, blurb, track definitions
  courses/<slug>/NN-*.md        course modules; 00 becomes the course index
  courses/<slug>/html/          a prebuilt course, published as-is (see below)
  notes/<slug>.md               a note category
  notes/<slug>/<topic>.md       a topic inside that category
site/
  build.py                      the static site generator
  test_build.py                 build verification, runs in CI
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
python site/build.py && python site/test_build.py
```

The verification pass asserts that every source file produced a page, every
internal link resolves, every Mermaid diagram rendered, and math survived the
pipeline. A static site fails silently, so this is what stops a broken link from
shipping.

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
under `html/`. The build copies them verbatim and injects two things per page: the
shared top bar, and a stylesheet that re-points the course's CSS custom properties
at the site palette. The content itself is never parsed or rewritten, so nothing
can be corrupted in translation — but such a course cannot be edited as Markdown
until someone converts it.

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
