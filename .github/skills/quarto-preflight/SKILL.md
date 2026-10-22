---
name: quarto-preflight
description: Run the required local validation for changed chapters, subfiles, R or Python code, and config files in this repository. Use when preparing a change for review.
argument-hint: Optionally provide changed file paths or a chapter path to validate
---

Run the local preflight checks
for Quarto, R, Python and config changes,
in this order,
and stop short of unrelated cleanup.

1. Identify the changed files
   (user-provided paths, or `git diff --name-only origin/main`).
2. Map changed `chapters/_subfiles/` files to their parent `chapters/*.qmd` files
   (the `find-parent-chapter` skill).
3. Install anything the checks need that is missing
   (R packages, Python packages, TinyTeX, `rsvg-convert`) rather than skipping a check.
4. Run `quarto render chapters/<chapter>.qmd --to html` on each affected parent chapter,
   and `--to pdf` when the change affects math, figures, tables or the handout.
   Read the output for errors and warnings,
   and call out math, missing-resource, cross-reference and link warnings explicitly.
5. Open the rendered output of every changed part
   and check that code output, numbers, tables and figures are what the prose says.
6. Run `Rscript -e 'lintr::lint("path/to/file")'` on each changed `.R` file
   and each changed `.qmd` with R code.
7. Run `typos` on the changed files.
8. Run `python3 tools/check_code_examples.py` if plain (unexecuted) code fences were added or changed.
   Every code example on the site is an executed chunk as of 2026-09-28,
   so with no plain fence the checker reports nothing to check.
9. Summarize: checks passed, warnings fixed, and anything still blocking review.

Fix only issues introduced by the current change,
and say why any failure outside it appears to be pre-existing.
