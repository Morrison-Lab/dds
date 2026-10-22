---
name: find-parent-chapter
description: Find the parent chapter for a changed Quarto subfile and print the exact render command to run. Use when working in `chapters/_subfiles/` or when you need the owning `chapters/*.qmd` file.
argument-hint: Optionally provide a subfile path to resolve
---

Resolve a changed content file
to the parent chapter that includes it.

This skill is for Quarto routing,
not content editing.

1. Determine the target path:
   the user-provided path if present,
   otherwise the active file.
2. If the target is already a parent chapter (`chapters/*.qmd`),
   report it and print
   `quarto render chapters/<chapter>.qmd --to html`.
3. If the target is under `chapters/_subfiles/`,
   search `chapters/*.qmd` for literal include lines
   `{{< include _subfiles/... >}}` naming it.
   A subfile can also be included from another subfile,
   so follow the chain up to a `chapters/*.qmd` file.
4. Report every parent found, with its render command.
   If none is found, say so,
   and note that an unreferenced subfile is not rendered anywhere.

Keep the response compact.
Do not start editing content.
