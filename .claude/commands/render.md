---
description: Render a single Quarto chapter and report build status
allowed-tools:
  - Bash(quarto render *)
---

Render only the touched chapter: `quarto render chapters/<chapter>.qmd --to html`,
and `--to pdf` as well when the change affects the handout.
Do not render the whole site for a check.

If the change is in a file under `chapters/_subfiles/`,
render the parent chapter that includes it, not the subfile.

Report:

- whether the render succeeded
- every ERROR or WARNING, with where it came from
  (call out math, missing-resource, unresolved cross-reference and link warnings explicitly)

If the render fails, diagnose the root cause and suggest a fix before proceeding.
If it fails because a tool or package is missing, install it and render again.
Do not commit `_site/` or `_freeze/`; they are build artifacts.
