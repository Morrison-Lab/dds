# CI workflows

The workflows were copied from lds (2026-10-22), whose `.claude/ci-workflows.md` explains each one in full.
Most call a reusable workflow in [`Morrison-Lab/gha`](https://github.com/Morrison-Lab/gha), pinned at `@v2` or `@v3`.
Read lds's file before editing `.github/workflows/`, and keep these differences in mind:

- `preview.yml` and `publish.yml` install only `knitr`, `rmarkdown`, `reticulate`, `downlit`, `xml2` and `tinytex`,
  and set `setup-julia: false`: pages run Python (PyTorch) through reticulate, and there is no Julia.
  Keep the two `r-packages` lists identical.
- `publish.yml` deploys this repository's own `gh-pages` branch.
  lds mirrors its site to a source-free `lds-site` repository; dds has no such mirror.
- lds's `check-calendar-refs.yml`, `check-course-overview-numbers.yml` and `check-generated-figures.yml`
  were left out, with the tools they run, since they check lds's own chapters.
- `claude-review.yml` and `claude-bot.yml` need the `CLAUDE_CODE_OAUTH_TOKEN` (or `ANTHROPIC_API_KEY`)
  secret, as in lds.
