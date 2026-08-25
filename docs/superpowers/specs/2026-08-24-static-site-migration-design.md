# Static site migration — design

**Date:** 2026-08-24
**Status:** approved

## Problem

The site is a Django 5.2 app rendering templates from PostgreSQL. GitHub Pages
serves static files only, so it cannot host the app at any price, and the domain
`mlinterviewnotes.com` has nowhere to point.

The database holds no content. Audit of `backups/ml_interview_notes_20250622_013520.sql`:

| Table | Rows | Content |
|---|---|---|
| `ml_deep_dives_category` | 5 | real descriptions |
| `ml_deep_dives_topic` | 29 | every body is the literal `"Content coming soon..."` |
| `question_answer_question` | 0 | empty |
| `system_design_systemdesign` | 0 | empty |

Meanwhile `transformers-course/` is 1.5 MB of finished material — 17 modules plus
an index — already built to static HTML by its own generator.

So the backend guards an empty room, and the only real content is already static.

## Decision

Convert to a static site built from Markdown, deployed to GitHub Pages at
`mlinterviewnotes.com`. Retire the Django scaffold to `legacy/`.

Rationale beyond the hosting question: an open-source notes platform needs
contributors to submit notes. Nobody can open a pull request against a Postgres
row. Markdown in git is the contribution model the project's own goal requires.

Migration cost is near zero today — 5 category descriptions and 29 placeholder
strings. It grows with every note added to the database.

## Theme

The course's design system becomes the site's design system, not the reverse.

`templates/base.html` is Tailwind-via-CDN with a Font Awesome kit and a gradient
animation — generic. The course theme is a considered blueprint aesthetic:
per-track accent colours, DM Sans / Fira Code, real dark mode, numbered sections,
and distinct section kinds (`takeaways`, `selfcheck`, `reconcile`).

The course is therefore *rebuilt* rather than dropped in as-is: shared brand,
shared site header, generated through the common pipeline — but everything else
on the site inherits the blueprint language.

## Architecture

```
content/
  index.md                          # site home
  courses/transformers/
    course.yml                      # title, blurb, track definitions
    00-README.md … 17-glossary-and-cheatsheet.md
    code/modern_decoder.py
  notes/                            # the 5 ported categories, as stubs
site/
  build.py                          # generalized from tools/build_site.py
  theme/{style.css,page.js,mermaid.js}
  test_build.py
_site/                              # build output, gitignored
legacy/                             # retired Django app
.github/workflows/pages.yml
```

### Generator changes

`tools/build_site.py` (407 lines) is already a competent SSG: markdown-it with
KaTeX math, Mermaid diagram shells with zoom/pan, real `<table>` markup,
scroll-spy TOC, prev/next paging, track grouping. It is hardcoded to one course.
Five changes generalize it:

1. **Content in, output out.** It currently writes `.html` beside each `.md`, so
   sources and artifacts interleave in git. Read `content/`, write `_site/`,
   gitignore the output.
2. **`TRACKS` becomes data** — moved from a module constant to per-course
   `course.yml`, so adding a course needs no Python edit.
3. **Brand, footer, and home link become parameters** rather than literals baked
   into the page template.
4. **A site shell** — top header carrying the site brand and section nav, wrapping
   the existing left module rail, which stays scoped to the course.
5. **Assets emitted as files** rather than inlined into all 18 pages: one
   cacheable `/assets/style.css`.

### URLs

Directory-style: `/courses/transformers/03-self-attention-from-scratch/`.

Decided now, before the domain is live and anything is indexed, because URL
structure is the one thing that is genuinely painful to change later. The
`.md` → `.html` rewriter in `r_link_open` becomes `.md` → `../slug/`. Assets are
referenced root-absolute (`/assets/…`), which is correct under an apex custom
domain and would need revisiting only if the site ever moved to a
`user.github.io/repo/` path.

## Deployment

`.github/workflows/pages.yml` builds on push to `main` and deploys to Pages.

DNS at the registrar: four A records for the apex to `185.199.108.153`,
`185.199.109.153`, `185.199.110.153`, `185.199.111.153`; a `www` CNAME to
`alliedbrother.github.io`. Repo Settings → Pages → source "GitHub Actions",
custom domain set, Enforce HTTPS on. A `CNAME` file ships in the output.

## Verification

A static site fails silently — a link rots and nobody notices. `site/test_build.py`
asserts, and runs in CI so a failure blocks the deploy:

- every source `.md` produced an output page
- every internal link resolves to a file that exists in `_site/`
- every ` ```mermaid ` fence became a diagram shell
- math delimiters survived the pipeline
- referenced assets exist

## Django retirement

The 5 category descriptions port to `content/notes/` stubs so the
Math / Libraries / ML / Deep Learning / NLP structure survives. The 29 topics
become titled placeholders.

`core/`, `ml_deep_dives/`, `system_design/`, `question_answer/`,
`ml_interview_notes/`, `manage.py`, `templates/`, `scripts/`, `staticfiles/`, and
`check_db.py` move to `legacy/` — retained rather than deleted, by request.
`backups/*.sql` stays where it is.

## Out of scope

Search, contributor tooling beyond a README section, and any re-introduction of a
backend. If accounts, submissions, or progress tracking are ever needed, that is a
separate design.
