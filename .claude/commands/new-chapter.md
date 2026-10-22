---
description: Scaffold a new chapter
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash(quarto render *)
  - Bash(python3 tools/check_site.py*)
---

Create a new chapter from `$ARGUMENTS` (e.g. `/new-chapter kernels Kernel Methods`).
The first token is the **slug** (the filename, without `.qmd`);
the rest is the **title** (default it to the slug).

1. Create `chapters/<slug>.qmd` with a front matter that names a distinct output file
   for each format, so the slide deck does not overwrite the web page:

   ```yaml
   ---
   title: "<title>"
   format:
     html: default
     revealjs:
       output-file: <slug>-slides.html
     pdf:
       output-file: <slug>-handout.pdf
   ---
   ```

   The deck's name must end in `-slides.html`, or it silently loses its "Webpage version" link.
   Do not set `date:`; the project sets `date: last-modified`.
   Give the chapter headings: a chapter with none makes a one-slide deck,
   which `tools/check_site.py` fails.
2. Add the page to the `render:` list in `_quarto-website.yml`,
   and to the navbar if it should be navigable.
3. Split the content into subfiles under `chapters/_subfiles/<slug>/`,
   one unit per file, following `AUTHORING.md`.
4. Render it (`quarto render chapters/<slug>.qmd --to html`, then `--to pdf`)
   and fix any warning.
