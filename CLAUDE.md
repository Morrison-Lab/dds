# CLAUDE.md --- dds (CSCI 481 Deep Learning notes)

This repository was scaffolded from [`Morrison-Lab/lds`](https://github.com/Morrison-Lab/lds) (DATA 571),
and follows lds's `CLAUDE.md` as its baseline.
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
link to [lds](https://morrison-lab.github.io/lds/) for anything lds defines,
rather than defining it again (ai-config `course-sequence.md`).

Content lives in `.qmd` files under `chapters/`; chapters include subfiles from `chapters/_subfiles/`.
Pages execute Python (PyTorch) through R's `reticulate`, inside knitr.
There is no Julia here, unlike lds.

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
- `chapters/_subfiles/<chapter>/` --- the units each chapter includes.
- `chapters/data/` --- local copies of datasets pages use, each listed under its page's `resources:`.
- `chapters/outtakes/` --- removed content kept for reuse (lds's "Outtakes" rule).
- `tools/`, `filters/`, `_extensions/`, the HTML and SCSS includes --- copied from lds;
  lds's `CLAUDE.md` describes each.
- `latex-macros/` --- the [`Morrison-Lab/macros`](https://github.com/Morrison-Lab/macros) submodule.
  Never write a bare `\exp`; write `\expf{...}` or `\expt` with your own delimiters.

## Build & Test Commands

```bash
git submodule update --init latex-macros
quarto render chapters/<chapter>.qmd --to html
quarto render                     # all formats in ONE pass; never split by format
python3 tools/check_site.py
python3 tools/check_list_markers.py
python3 tools/check_example_after_def.py
typos path/to/changed/files
```

Every executed chunk must run in seconds on a CPU, since CI has no GPU (`AUTHORING.md`).

## Pre-Commit Checklist (MANDATORY)

Follow lds's checklist: render each parent chapter you changed, fix every warning,
check the rendered output against the prose, and run `typos` on the changed files.
CI is not the test.

## Key Conventions

lds's key conventions apply unchanged, apart from the language:
never type or fake output;
exercise and solution pairs;
one term per definition div;
`{{< slidebreak >}}` rather than `---`;
every page names a distinct `output-file` per format,
and a deck's ends in `-slides.html`;
a new page goes in the `render:` list,
the navbar and the sidebar of `_quarto-website.yml`.

## Pull Requests

- Workflow and other infrastructure changes go in their own PRs, never mixed with lecture content.
- Merging `main` into any PR branch is always allowed.
- Ezra merges content; tooling and configuration merge once clean.

## CI

The workflows were copied from lds and call reusable workflows in
[`Morrison-Lab/gha`](https://github.com/Morrison-Lab/gha).
`.claude/ci-workflows.md` is lds's description of them;
`publish.yml` differs from lds's: it publishes this repository's own `gh-pages` branch,
with no source-free `-site` mirror.

## Cross-project agent rules (ai-config)

This repository follows the maintainer's cross-project agent rules in
[Morrison-Lab/ai-config](https://github.com/Morrison-Lab/ai-config).
If your harness has not already loaded them (Claude Code loads them through
the ai-config plugin), read
[AGENTS.md](https://github.com/Morrison-Lab/ai-config/blob/main/AGENTS.md)
before starting work, and follow it alongside this file.
This file's own instructions add to those rules, and win only where they are
more specific.
