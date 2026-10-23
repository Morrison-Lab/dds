---
description: Lint R code in changed .R and .qmd files
allowed-tools:
  - Bash(Rscript -e 'lintr::lint*)
---

For each changed `.R` file, and each changed `.qmd` file with R code,
run `Rscript -e 'lintr::lint("path/to/file")'`.

Report issues grouped by file, with the line number, the rule violated,
and a one-line fix.
Fix only issues your change introduced.
If there are none, say so.
