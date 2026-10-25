# CLAUDE.md --- dds (CSCI 481 Deep Learning notes)

The repo-wide instructions for every agent are in
[`.github/copilot-instructions.md`](.github/copilot-instructions.md),
with file-scoped ones in `.github/instructions/`;
the course's lecture conventions are in [`AUTHORING.md`](AUTHORING.md).
Read all three before changing anything under `chapters/`.

## Project Overview

This is a Quarto website of lecture notes for CSCI 481, Deep Learning,
taught by Ezra Morrison at Western Washington University in Spring 2027
(CSCI 481 only; there is no CSCI 581 section).
The course follows the module sequence of Brian Hutchinson's CSCI 481/581,
whose labs, project and lectures may be adapted with credit (his permission, per Ezra, 2026-10-22).
It is taught live from these notes, not flipped.
It builds on DATA 371/471:
link to [lds](https://morrison-lab.github.io/lds/), the notes for that course, for anything lds defines,
rather than defining it again (ai-config `course-sequence.md`).

Content lives in `.qmd` files under `chapters/`; chapters include subfiles from `chapters/_subfiles/`.
Pages execute Python (PyTorch) through R's `reticulate`, inside knitr.

| Repository | Holds |
| --- | --- |
| `Morrison-Lab/dds` (here) | The course notes website |
| `Morrison-Lab/dlg` | Homework, exams, solutions --- whoever wrote them |
| `Morrison-Lab/mlr` | Copyrighted material, large files, past offerings' originals |

This repository is meant to be public, like the lab's other notes sites;
`dlg` and `mlr` stay private (Ezra, 2026-10-22).
This repository is what students read,
so **nothing that reveals an answer to a graded item belongs in it**.
Before writing an exercise or solution, check it against the graded material in `dlg`.
**Student records** (gradebooks, rosters, submissions) belong in none of the three repositories.

Brian Hutchinson's video lectures sit behind a login that his site asks not be shared,
and this site is served by Pages,
so never link the videos or name the login here.

## Structure

- `chapters/` --- the Quarto pages, the source of truth.
- `chapters/_subfiles/<chapter>/` --- the units each chapter includes, one div per file.
- `chapters/data/` --- local copies of datasets pages use, each listed under its page's `resources:`.
- `chapters/outtakes/` --- removed content kept for reuse:
  move good content there rather than deleting it,
  as a `_`-prefixed subfile, and leave its include disabled where it used to be,
  escaped as well as commented out: `<!-- {{</* include outtakes/_name.qmd */>}} -->`.
- `tools/` --- the checkers (`check_site.py`, `check_list_markers.py`, `check_example_after_def.py`,
  `check_code_examples.py`, `check_source_overlap.py`), the dependency-graph builder,
  their self-tests in `tools/tests/`,
  and the `pre-render` scripts:
  `install-extensions.py` fetches the unlicensed `custom-callout` extension at a pinned commit,
  `skip-pdf-in-previews.py` drops the PDF format from a pull request's preview,
  `set-mtimes-from-git.py` makes `date: last-modified` show a page's last commit,
  `build_math_macros.py` turns the shared math macros into header files,
  and `provision-python.R` has reticulate fetch Python before any page loads.
  `use-torch.R` is sourced by each page that uses PyTorch, from its setup chunk,
  and applies `reticulate_logging_patch.py`.
- `filters/` --- Lua filters:
  `unlink-pages.lua` (PDF only) turns links to other pages into text, since each handout is shared on its own;
  `rule-table-rows.lua` (PDF only) rules the rows of a `.ruled-rows` table;
  `section-slide-break.lua` (`revealjs` only) gives each level 1 header its own slide;
  `strip-html-comments.lua` keeps HTML comments out of the served page;
  `watch-callouts.lua` gives the "Watch" and "Read" callouts their ids.
- `_extensions/` --- vendored Quarto extensions; do not review or modify them here.
- `latex-macros/` --- the [`Morrison-Lab/macros`](https://github.com/Morrison-Lab/macros) submodule,
  loaded on every page in every format.
  Prefer a macro wherever one names the concept.
  Never write a bare `\exp`; write `\expf{...}` or `\expt` with your own delimiters.

## Build & Test Commands

```bash
git submodule update --init latex-macros
quarto render chapters/<chapter>.qmd --to html
quarto render                     # all formats in ONE pass; never split by format
python3 tools/check_site.py
python3 tools/check_list_markers.py
python3 tools/check_example_after_def.py
for t in tools/tests/test_*.py; do python3 "$t"; done
typos path/to/changed/files
```

Render all formats in one pass:
each render prunes the supporting files of the formats it is not producing,
so separate per-format renders leave all but the last without their CSS and JavaScript.

Every executed chunk must run in seconds on a CPU, since CI has no GPU (`AUTHORING.md`).
Rendering needs R with `knitr`, `rmarkdown` and `reticulate`;
a page declares its Python packages with `reticulate::py_require()`.
The PDF handouts need a LaTeX toolchain (`quarto install tinytex`) and `rsvg-convert`.
In a cloud session the SessionStart hook `.claude/hooks/render-toolchain.sh` installs these
(see [`.claude/render-toolchain.md`](.claude/render-toolchain.md)).

## Pre-Commit Checklist (MANDATORY)

1. Render each parent chapter whose subfiles you edited, to HTML,
   and to PDF when the change affects the handout, and fix every warning.
2. Open the rendered output and check that code output, numbers, tables and figures say what the prose says.
3. Run `typos` on the changed files.
4. Commit only after all of these pass.

CI is not the test.

## Key Conventions

- Never type or fake output: a number in the prose comes from inline code.
- Lectures are exercise and solution pairs, with a slide break between them.
- One term per definition div.
- `{{< slidebreak >}}` rather than `---`.
- Every page names a distinct `output-file` per format,
  and a deck's ends in `-slides.html`.
- A new page goes in the `render:` list, the navbar and the sidebar of `_quarto-website.yml`.

`tools/check_site.py` enforces the output-file, deck-naming and render-list rules on a full render.

## Pull Requests

- Workflow and other infrastructure changes go in their own PRs, never mixed with lecture content.
- Merging `main` into any PR branch is always allowed.
- Ezra merges content; tooling and configuration merge once clean.

## CI

The workflows call reusable workflows in [`Morrison-Lab/gha`](https://github.com/Morrison-Lab/gha);
[`.claude/ci-workflows.md`](.claude/ci-workflows.md) describes them.

## Cross-project agent rules (ai-config)

This repository follows the maintainer's cross-project agent rules in
[Morrison-Lab/ai-config](https://github.com/Morrison-Lab/ai-config).
If your harness has not already loaded them (Claude Code loads them through
the ai-config plugin), read
[AGENTS.md](https://github.com/Morrison-Lab/ai-config/blob/main/AGENTS.md)
before starting work, and follow it alongside this file.
This file's own instructions add to those rules, and win only where they are
more specific.
