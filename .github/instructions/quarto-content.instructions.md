---
applyTo: "**/*.{qmd,Rmd}"
description: "Use when editing Quarto chapters, subfiles, slides, handouts, or math-heavy narrative content in this repository."
---

# Quarto Content

- Treat the parent chapter as the render target.
  If you edit a file under `chapters/_subfiles/`,
  find the `chapters/*.qmd` file that includes it,
  and run `quarto render chapters/<chapter>.qmd --to html` on that parent.
- Preserve the source formatting style:
  prose and comments are written phrase by phrase on separate lines.
- Link to source `.qmd` files, not rendered `.html` files.
- Subfiles under `chapters/_subfiles/` must not start with a heading,
  and must not contain a references section.
- Use Quarto div wrappers for figures and tables.
- Never type or paste output, and do not hard-code computed values in narrative text.
  Run the code in a `{python}` chunk on a local copy of the data in `chapters/data/`,
  and cite results with inline R.
- Read `AUTHORING.md` before adding or restructuring lecture content:
  exercises and solutions come in pairs,
  and nothing that answers a graded item belongs in this repository.

See also:

- [CLAUDE.md](../../CLAUDE.md)
- [AUTHORING.md](../../AUTHORING.md)
- [copilot-instructions.md](../copilot-instructions.md)
