# Legacy Django application

This directory holds the Django 5.2 + PostgreSQL application the site ran on
before it became a static site. It is **not built, deployed, or tested**. Nothing
in the live site depends on it.

## Why it was retired

GitHub Pages serves static files only, so the Django app could not be hosted
there at any price — and the site needed a home for `mlinterviewnotes.com`.

The deciding factor was that the database held no content. From
`backups/ml_interview_notes_20250622_013520.sql`:

| Table | Rows | Content |
|---|---|---|
| `ml_deep_dives_category` | 5 | real descriptions |
| `ml_deep_dives_topic` | 29 | every body was the literal `"Content coming soon..."` |
| `question_answer_question` | 0 | empty |
| `system_design_systemdesign` | 0 | empty |

So the backend was a scaffold around an empty room, and migration cost was near
zero — a cost that would have grown with every note added.

The second reason is the project's own goal. An open-source notes platform needs
contributors, and nobody can open a pull request against a Postgres row. Markdown
in git is the contribution model the project requires.

## What was carried over

The 5 category descriptions and all 29 topic titles now live in
`content/notes/*.md` as stubs, preserving the Math / Libraries / ML /
Deep Learning / NLP structure. The database backup remains in `backups/`.

## If you need to run it

It still works as it did. From this directory:

```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py runserver
```

It expects the PostgreSQL settings in `ml_interview_notes/settings.py`.

Design notes for the migration: `docs/superpowers/specs/2026-08-24-static-site-migration-design.md`.
