# CI workflows

Most workflows call a reusable workflow in [`Morrison-Lab/gha`](https://github.com/Morrison-Lab/gha), pinned at `@v2` or `@v3`.
Each file's header comment says what it checks and why it is set up that way;
read it before editing that workflow.

- **Site build and deploy.**
  `preview.yml` renders a pull request's site, `preview-deploy.yml` publishes it under `pr-preview/pr-<N>/`,
  and `cleanup-pr-previews.yml` removes the previews of closed PRs.
  `publish.yml` renders `main` and deploys it to this repository's `gh-pages` branch.
  `preview.yml` and `publish.yml` install only `knitr`, `rmarkdown`, `reticulate`, `downlit`, `xml2` and `tinytex`,
  and set `setup-julia: false`: pages run Python (PyTorch) through reticulate.
  Keep their two `r-packages` lists identical.
- **Checks on the rendered site.**
  `check-site.yml` runs `tools/check_site.py`,
  and `check-equation-renders.yml` finds equations MathJax cannot render.
- **Checks on the source.**
  `check-list-markers.yml`, `check-examples-after-defs.yml`, `check-new-line-breaks.yml`,
  `check-term-links.yml`, `check-definite-antecedents.yml`, `check-ai-tells.yml`, `check-typos.yml`,
  `check-bibliography-dois.yml` and `check-self-credits.yml` check the prose and its sources,
  and `check-dependency-graph.yml` checks the cross-references between pages;
  `check-junk-files.yml`, `check-secrets.yml`, `check-orphaned-submodules.yml` and `lint-workflows.yml`
  check the repository itself.
- **Review.**
  `claude-review.yml` reviews each pull request, and `claude-bot.yml` answers `@claude` mentions.
  Both need the `CLAUDE_CODE_OAUTH_TOKEN` (or `ANTHROPIC_API_KEY`) secret.
  `annotation-gate.yml` fails any finished run that left a CI annotation not on its allowlist.
