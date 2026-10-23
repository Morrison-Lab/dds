---
applyTo: "**/*.{R,r,py,yml,yaml}"
description: "Use when editing R or Python source, Quarto configuration, or GitHub workflow files in this repository."
---

# R, Python And Config Work

- Check [_quarto.yml](../../_quarto.yml), [_quarto-website.yml](../../_quarto-website.yml)
  and the relevant workflow file before changing render or CI behavior.
- Pages execute Python (PyTorch) through R's `reticulate`, so every render needs R
  with `knitr`, `rmarkdown` and `reticulate`.
  Python packages are declared on the page with `reticulate::py_require()`,
  and `reticulate` provisions them.
  The preview and publish workflows install the R packages;
  a new R package a chunk needs goes into both workflows in the same PR.
- For changed `.R` files, and `.qmd` files with R code,
  run `Rscript -e 'lintr::lint("path/to/file")'`.
- If your change affects rendering,
  render the touched parent chapter to HTML (and PDF when relevant),
  not the whole site.
- Workflow changes go in their own PR, not mixed with lecture content.
  Most workflows call a reusable workflow in `Morrison-Lab/gha`;
  read `CLAUDE.md`'s CI section before changing one, because several inputs are easy to get wrong.

See also:

- [CLAUDE.md](../../CLAUDE.md)
- [copilot-instructions.md](../copilot-instructions.md)
