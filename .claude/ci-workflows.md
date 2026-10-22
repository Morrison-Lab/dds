# CI workflows

Moved out of `CLAUDE.md` unchanged.

Most workflows call a reusable workflow in [`Morrison-Lab/gha`](https://github.com/Morrison-Lab/gha), pinned at `@v2`.
Six pin `@v3` instead:
`preview.yml` and `publish.yml`, because only `@v3` has the `setup-julia` input;
`preview-deploy.yml`, because only `@v3` checks that a preview's PR number matches the branch that built it
([gha#1033](https://github.com/Morrison-Lab/gha/pull/1033));
`claude-review.yml`, whose review job can read check runs (`checks: read`);
and `check-new-line-breaks.yml` and `check-typos.yml`, which on a PR diff from the merge ref's first parent
rather than the event payload's stale `base.sha`
([#184](https://github.com/Morrison-Lab/lds/issues/184), [gha#1008](https://github.com/Morrison-Lab/gha/pull/1008)).
The workflows:

- `publish.yml` --- renders all formats and deploys to `gh-pages` on push to `main`.
- `preview.yml` / `preview-deploy.yml` --- PR previews: a read-only build half and a base-repo deploy half.
  `preview.yml`'s `paths-ignore` skips the build on a PR that touches only files the site is not built from
  (`.github/`, `.claude/`, root `*.md`), so such a PR gets no preview and no check-site run.
  A preview leaves out the PDF handouts, which are about half of its render time:
  `tools/skip-pdf-in-previews.py` strips `pdf` from the project file and every chapter before Quarto reads them,
  because a profile cannot drop a declared format.
  A PR labelled `preview:pdf` renders them, as do a local render and the publish build (Ezra, 2026-10-12).
  In CI full-project renders (`GITHUB_ACTIONS=true` with `QUARTO_PROJECT_RENDER_ALL=1`, i.e., `publish.yml` on `main` and `preview:pdf` PRs, or when `LDS_SPLIT_PDF=1`),
  `tools/skip-pdf-in-previews.py` first runs `quarto render --to pdf` in its own child process
  and then strips `pdf` (updating the raw-file MD5 `"hash"` in `_freeze/chapters/<stem>/execute-results/html.json`)
  so the outer `quarto render` builds `html` and `revealjs` together without exceeding the 16 GB runner memory limit.
  When the handouts are skipped (not split), the script ships `preview-no-pdf.txt` in the site, so `check_site.py` does not report the missing handouts;
  in split mode no `preview-no-pdf.txt` marker is written, so `check_site.py` still requires every `*-handout.pdf` in `_site/chapters/`.
- `check-equation-renders.yml` --- finds equations MathJax cannot render in the built site.
  It skips the crawl for a PR whose changes are all under `chapters/` with no math-like character
  (`$`, backslash, `^`, `_`, `{`, `=`) on a changed line, and runs it for anything else.
- `cleanup-pr-previews.yml` --- weekly housekeeping on `gh-pages`.
- `lint-workflows.yml` (actionlint and zizmor over `.github/`), blocking.
  `.github/zizmor.yml` accepts tag pins rather than commit SHAs,
  and `.github/dependabot.yml` keeps the tags current.
- `check-secrets.yml` (gitleaks over the full history on pushes to `main` and weekly, and over only the commits a PR adds on PRs), blocking.
- `check-junk-files.yml` (tracked OS and editor detritus), blocking.
- `check-typos.yml` (misspellings on the lines a PR adds), blocking;
  it replaces rme's spellcheck.
- `check-bibliography-dois.yml` (each book and article in `references.bib` has a DOI that resolves and matches its Crossref record), blocking.
  Its `exclude-keys` lists the entries with no DOI in Crossref, the ones whose author field the checker cannot parse,
  and the ones whose publisher answers with status 202;
  drop a key once its entry passes.
- `check-new-line-breaks.yml` (packed sentences on added lines), blocking since 2026-09-26: write one sentence per source line.
- `check-ai-tells.yml` (AI-prose tells on added lines) --- warn-only.

`check-orphaned-submodules.yml`, `check-site.yml` and `check-list-markers.yml` are this repository's own:
the first fails on a gitlink with no `.gitmodules` entry,
such as a nested repository swept up by `git add -A`,
the second runs `tools/check_site.py` over the preview build,
and the third runs `tools/check_list_markers.py` and its self-test on any PR that changes `chapters/`.
`check-examples-after-defs.yml` runs `tools/check_example_after_def.py` and its self-test the same way, and only warns for now.

Both render workflows install R with `knitr`, `rmarkdown`, `reticulate`, `tinytex`, `JuliaCall`, `exams` and `exams2forms`, and execute every chunk;
`reticulate` provisions each page's declared Python packages,
and `setup-julia: true` installs Julia and JuliaCall's Julia dependencies before the render.
Neither workflow sets `julia-version`, so both get gha's default, `1`, the latest 1.x release;
gha also sets `R_LD_LIBRARY_PATH` for the render.
An R package a new chunk needs goes into both workflows' `r-packages` in the same PR.

Easy to get wrong, each in a way that looks like something else:

- **The `formats` input means opposite things in the two `gha` workflows.**
  In `quarto-publish.yml`, empty is the single bare `quarto render` this site needs.
  In `preview.yml`, empty selects a legacy branch that renders formats in separate invocations
  (and `revealjs` only on a PR labelled `preview:revealjs`),
  and the bare render is spelled `formats: default`.
  Getting it wrong publishes pages with no CSS rather than failing.
- **`tinytex: true` on `preview.yml` needs the R `tinytex` package**, which it shells out to for extra TeX packages.
  The two workflows still differ here:
  the preview installs `luacolor` and `lua-ul`, and `quarto-publish.yml` does not,
  so a handout needing either would go green on the PR preview and red on the push to `main`.
- **`workflow_run` triggers only fire for workflow files already on the default branch**,
  so `preview-deploy.yml`, `check-equation-renders.yml` and `check-site.yml` do not run on the PR that changes them.
- **A caller must not name a `concurrency` group `gh-pages`.**
  `quarto-publish.yml`'s `deploy` job holds that group, so the caller deadlocks against it,
  failing with no runner, no steps and no log (the logs endpoint returns 404),
  so the site just stops publishing.
  A block on the calling job deadlocks the same way,
  and `publish.yml` runs only on push to `main`, so no PR check can catch it.
  Its first run failed this way
  ([#5](https://github.com/Morrison-Lab/machine_learning_lecture_materials/issues/5));
  `publish.yml`'s group is now `quarto-publish-${{ github.ref }}`.

