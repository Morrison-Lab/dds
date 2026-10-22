# Repository instructions

These are the repo-wide instructions for every agent working here
(Claude Code reads them through `CLAUDE.md`).
They follow [`Morrison-Lab/rme`](https://github.com/Morrison-Lab/rme)'s
`.github/copilot-instructions.md` as the baseline,
adapted to this repository.
`AUTHORING.md` adds the conventions specific to this course's lectures.

> [!IMPORTANT]
> **MANDATORY TESTING BEFORE EVERY COMMIT:**
>
> Before committing ANY change to a `.qmd`, `.R`, `.py` or config file:
>
> 1. **Run `quarto render chapters/<chapter>.qmd --to html` on each chapter whose subfiles were edited**
>    (the parent chapter, not the subfile; not the whole site).
>    Also render `--to pdf` when the change touches math, figures, tables or anything else
>    the PDF handout renders differently.
> 2. **Verify it completes successfully** (exit code 0, no errors).
> 3. **Address ALL rendering warnings**:
>    math that does not typeset, missing resources, unresolved cross-references (`?@...`),
>    broken links, deprecated syntax.
>    Fix every warning you can, and document any you cannot.
> 4. **Lint changed R code**: `Rscript -e 'lintr::lint("path/to/file.qmd")'`.
>    Fix issues in code you wrote or modified; ignore pre-existing ones.
> 5. **Spell-check**: `typos path/to/changed/files`.
>    Fix misspellings you introduced;
>    add correctly spelled terms it misreads to `_typos.toml`.
> 6. **If you added or changed a plain (unexecuted) code fence**,
>    run `python3 tools/check_code_examples.py`:
>    a render does not run plain fences, so it cannot catch a broken one.
>    Every code example on the site is an executed chunk as of 2026-09-28,
>    so this step applies only if a plain fence is added.
> 7. Only then commit.
>
> **CRITICAL RULES:**
> - **CI is NOT the test.** Test locally before pushing.
> - **Install whatever a render or check needs** (R and its packages, Python and its packages,
>   TinyTeX, `rsvg-convert`) rather than skipping the check.
>   R and Python are always available here;
>   never design content, tooling or CI around their absence.
> - **Only fix lint and spelling issues in code you changed.**

## General Development Principles

**CRITICAL**: Do not make assumptions about what code will do;
always test it yourself.

- **Test your changes**: run the actual commands.
- **Verify output**: check that the expected files are created, with the expected content.
- **Never claim success without evidence**: report that something works only after you have seen it work.

## Executed code and local data

- **Never type, paste or fake output.**
  Code whose output a page shows is an executable `{r}` or `{python}` chunk,
  so what the reader sees is what the code printed when the page was rendered.
  No hand-written block of "output" under a code fence,
  and no `# ->` comment standing in for a printed value.
- **No hard-coded results in narrative text.**
  Compute every number in a chunk and cite it with inline code, `` `r n_spam` ``.
  Leave that chunk echoing (folded, by default), so readers can see how the number was made.
- **Analyse a local copy of the data.**
  A chunk reads its data from `chapters/data/`, never from a URL,
  and the page lists the file under `resources:` so readers can download the same file.
  The prose names the data's original source,
  with a link to where it is published and a `[@key]` citation,
  and says the file is a copy.
  If the copy is not byte-for-byte what the source publishes,
  a script under `tools/` rebuilds it and says what it changed.
- Python chunks run through R's `reticulate`, inside knitr.
  A page with Python chunks declares the Python packages they import
  in an R setup chunk before the first of them,
  `reticulate::py_require("pandas")`;
  `reticulate` then provisions that Python environment itself, locally and in CI.
- Code examples are Python, with PyTorch (`AUTHORING.md`);
  this site has no Julia, unlike lds.
- Every code example on the site is an executed chunk as of 2026-09-28.
  `tools/check_code_examples.py` runs any plain fence a page adds,
  and over `chapters/` with none it reports nothing to check and exits 0.

## Scoped Instructions

Use the matching file-scoped instructions in `.github/instructions/`
in addition to this file:

- `quarto-content.instructions.md` for `.qmd` authoring
- `r-and-config.instructions.md` for R, Quarto YAML and workflow/config changes

Reusable workflows live under `.github/skills/` and `.claude/commands/`.

## Narrative Flow

- Introduce concepts before they are used.
  Never refer to a section, definition or term
  that has not been introduced at that point in the document.
- Avoid forward references within a chapter
  ("See @sec-later-section for details")
  when those concepts are needed to understand the current content.
  Reorder sections so that foundational material precedes applied material.
- **After every definition or concept, include a concrete example**,
  preferably numerical, in an `#exm-` div immediately following the `#def-` div.
  The example shows specific numbers or a small dataset,
  not the definition restated in other words.
  Add a counterexample after it when one is feasible.
- After every theorem, lemma or corollary:
  when a proof is feasible and in scope, a proof, then an example that uses the result.
  Keep related items adjacent: theorem, proof, examples, counterexamples.
- Never use "above" or "below" to refer to content; cross-reference it with `@id`.
- Never let `@id.`, `@id)` or `(@id)` start a line:
  Pandoc can read it as an example-list marker and swallow the reference.
  `tools/check_site.py` fails a page where this happened.
- After "This", "That" and "Those", name the referent ("This estimator", not "This").

## Code Review Scope

`_extensions/` holds Quarto extensions vendored from other repositories.
Treat them as third-party code: read them for context,
but do not review or modify them here.

## Pull Request Guidelines

- When you start working on a PR, remove any existing review requests.
- Merging `main` into any PR branch is always allowed, and never needs asking (Ezra, 2026-10-22).
  See ai-config's `shared/workflow/sync-with-main.md`.
- Verify every changed hyperlink before requesting review:
  internal links point to existing files and ids,
  external URLs resolve,
  and relative paths are correct.
- Workflow, `.github/`, CI and other infrastructure changes go in their own PRs,
  never mixed with lecture-content PRs.
- The "clear freezer" label is only for forcing a full re-render.
  Quarto's freeze cache does not notice changes in included files
  ([quarto-dev/quarto-cli#6793](https://github.com/quarto-dev/quarto-cli/issues/6793)),
  but the preview and publish workflows drop the frozen results of every page
  whose own file, included subfiles or named data files changed
  ([Morrison-Lab/gha#1022](https://github.com/Morrison-Lab/gha/pull/1022)),
  so an edited subfile no longer needs the label.
- **Fact-check prose and reasoning when reviewing content changes**
  (ai-config `shared/writing/fact-check-prose.md`):
  check factual claims against sources,
  work through derivations and informal arguments step by step,
  and check every computed value the prose describes against the rendered output.
  Use the PR preview (linked in a sticky comment on the PR)
  or render the page locally.

## Linking Within the Quarto Website

Link to the **source `.qmd` file**, not the rendered `.html`:
`[the gradient](gradient-descent.qmd#sol-the-gradient)`, not `gradient-descent.html#...`.
This is a Quarto `website` project, so `@id` resolves only within one rendered page
(the parent chapter and everything it includes).
For an element on another page, use an explicit link: `[text](other-chapter.qmd#id)`.

## Attribution for Adapted Content

When content is adapted from a published source,
**always provide explicit attribution** with a `[@key]` citation
and a note saying what was adapted,
at the top of the chapter (its Source note) or beside the adapted content,
in a `::: notes` div.

Include chapter and page locators: `[@key, Chapter 8, p. 194]`.
Cite a page only after checking it in the source itself;
otherwise cite the chapter or section.
Name the exact exercise or example adapted, with its number, when the source numbers them.
Do not reproduce copyrighted text verbatim without quotation marks and a citation.

### Avoiding unintentional plagiarism

A model that has read a textbook can reproduce its sentences from memory without knowing it,
so every agent and author follows these rules for new prose and checks old prose by the same standard.

- Write each explanation from the mathematics or the data in front of you, in your own sentence structure.
  Never reword a source sentence by sentence: swapping synonyms into someone else's sentence is still copying.
- A passage adapted from a source carries a citation with a locator at the passage
  and, for more than a sentence of adapted structure, a Source note (see above).
  A passage with no source in mind that you cannot derive yourself is a warning sign, not a license.
- Quote only with quotation marks (or a blockquote) and a `[@key, locator]` citation, and keep quotes short.
- Run `python3 tools/check_source_overlap.py --corpus <dir of extracted .txt> --git-diff origin/main`
  before pushing prose.
  The corpus is text from the textbooks in `Morrison-Lab/mlr`, extracted with `pdftotext` into a scratch directory;
  never commit that text.
  Fix every flagged paragraph by rewriting it from scratch, or quote and cite it.
  Mark a flagged paragraph checked by hand, for a generic definition or a standard formula, with `<!-- overlap-ok: reason -->`.
- A bulk-generated chapter (a commit message such as "content from multiple sources") is unreviewed for provenance:
  check it with the tool before building on it.

## Evidence and Citations

- Every factual claim not proved in the text gets a specific citation.
- Cite through `references.bib` and Pandoc syntax (`[@key]`), never plaintext author-date.
- A citation used as a sentence's subject takes a singular verb:
  "@hastie2009elements describes ...", not "describe".

## Decomposing content into subfiles

- A subfile under `chapters/_subfiles/` does not begin with a heading;
  the heading goes in the parent file, immediately before the `{{< include >}}`.
- One subtopic per subfile.
- No references section in a subfile.
- Keep files short: aim for under about 100 lines, and split longer ones into named subfiles.

## Variable Definitions in Exercises

List an exercise's variables as bullet points or a table,
with the symbol, its plain-language meaning, and the dataset column it maps to,
not inline in prose.

## Solution Wording

State the conclusion directly.
Do not phrase an answer conditionally when the data or figure supports a specific conclusion.

## Source Formatting

Put each phrase or clause on its own source line,
in prose, comments and docstrings alike.
It keeps diffs readable.
Do not reflow lines you are not otherwise changing.

## ggplot2 Layer Style

Put `aes()` in its own layer, never as an argument to `ggplot()`:

```r
ggplot(my_data) +
  aes(x = time, y = value, color = group) +
  geom_line()
```

## Notes Divs

Put asides, parenthetical references (`c.f. @key, Section 3.4`)
and multi-sentence explanation that would overflow a slide
in a `::: notes` div.
It renders as an aside on the web page and as speaker notes in the slides.

## Fenced Divs and Lists

Do not indent `:::` fence markers as list-continuation content;
indented markers can render as literal `:::`.
End the list first, then start the div at the left margin with blank lines around it.

## Theorem-Type Divs

Theorem-type divs are `#thm-`, `#lem-`, `#cor-`, `#prp-`, `#cnj-`, `#def-`, `#exm-`, `#exr-`, `#rem-` and `#sol-`.

- Put the div's title inside the div, as a `####` heading or deeper.
  A `##` or `###` inside a div becomes a numbered section and breaks the div numbering.
- Never move a div title outside its div, or change its level when fixing heading-level jumps;
  it is not a section heading.
- Every worked example lives in an `#exm-` div, never a plain section.
- A definition that uses another technical term relies on that term having its own `#def-` div.
- Aliases of a term ("also called") stay in the same div, in bold and in the div title.
  Cite a source that uses an alias on the bold alias itself,
  as in `**training error** [@hastie2009esl, sec. 7.2]`;
  never write "the book uses that name",
  because the citation already says so (Ezra, 2026-10-15).
- Define one term or state one result per `#def-`, `#thm-`, `#lem-`, `#cor-` or `#prp-` div (Ezra, 2026-10-13).
  A second new bolded term gets its own div and its own id.
  A title that joins two different terms with "and" is the same mistake in a definition;
  a theorem title such as "Existence and Uniqueness" is fine when the div states one result.
  An alias ("also called ...") stays in the same div, in bold and in the title in parentheses,
  as in "Penalty weight (regularization strength)".
  A parametrized family is one term, defined in one div with its parameter named there;
  its members are instances, not terms of their own.
  A property and the object it describes are two terms:
  "converges" (a property) and "limit" (the value it names) each get their own div,
  and a named property such as "symmetric" is defined apart from the matrices that have it.
  Order the split divs so that none uses a term before its own div defines it,
  and keep the parent's id on the original term so inbound links still land.
  Commentary goes in a separate `#rem-` div; move it, never remove it.
  The gha `check-informal-definitions` check flags a div that breaks this
  ([Morrison-Lab/gha#1072](https://github.com/Morrison-Lab/gha/pull/1072)).

## Slide Breaks

- Use `{{< slidebreak >}}`, never `---`.
  The shortcode breaks the slide only in `revealjs` output and renders nothing elsewhere,
  while `---` draws a rule in every format.
  When moving a div next to an existing `---`, replace the `---`.
- Put `{{< slidebreak >}}` on its own line, with blank lines around it,
  before every theorem-type div.
- When a subfile begins with a theorem-type div,
  the slide break goes in the parent, before the `{{< include >}}`.
- A heading immediately before a div may share its slide:
  omit that slide break and mark the spot with
  `<!-- heading shares the slide; do not re-flag -->`.

## Code Chunk Options

- Code folds by default.
  Keep chunks echoing, so readers can see how a figure or number was made,
  and set `#| code-fold:` to `show`, `true` or `false` by the rule in
  [Set `code-fold` by what the code does and whether the prose mentions it](https://morrison-lab.github.io/psw/chapters/teaching.html#set-code-fold-by-what-the-code-does-and-whether-the-prose-mentions-it)
  (Ezra, 2026-10-16).
- Do not hide code from readers without a good reason,
  and say the reason in a comment in the chunk.
  A computation whose results the prose cites inline is not one:
  readers should be able to see how those numbers were made (Ezra, 2026-10-05, on lds#401).
  Loading packages and setting options are not reasons either:
  readers need them to run the code themselves.
  One reason that does hold is code that would give away an exercise's answer before its solution.
  Then move the code into the solution's chunk when it can wait until there.
  When the exercise itself quotes the code's results, so it has to run first,
  hide only the lines that give the answer away, give that chunk a `label`,
  and show it in the solution with a chunk that has `#| ref.label: <label>`
  and `#| eval: false`, so both copies are the same code
  (lds's `chapters/backpropagation.qmd` and `_sol-relu-fix.qmd` do this, lds#417).

## Figures and Tables

- Caption figures and tables with the div form, not chunk options:

  ````markdown
  ::: {#fig-my-figure}

  ```{r}
  plot(x, y)
  ```

  Caption text here.

  :::
  ````

- Use a `fig-` or `tbl-` label only on something that renders a figure or table.
- Every figure has real alt text (`fig-alt`), describing what it shows.
- Wrap a table too wide for a portrait PDF page in a `::: {.landscape}` div.

## Math

- Include every intermediate step in a derivation,
  one operation per aligned line,
  each annotated with the rule it uses (`&& \text{(...)}`)
  (ai-config `shared/writing/math-derivation-steps.md`).
- Use the shared semantic macros from the `latex-macros` submodule
  ([`Morrison-Lab/macros`](https://github.com/Morrison-Lab/macros))
  wherever one names the concept: `\E{}`, `\Var{}`, `\vx`, `\vbeta`, `\norm{}` and the rest.
  `tools/build_math_macros.py` loads them on every page, in every format.
- Mark an equality that holds by definition, not a bare `=`: `\eqdef`.
- Check the dimensions of every matrix expression,
  and label them in display math with `\underbrace{X}_{n \times p}`.
  Subscript zero matrices: `\mathbf{0}_{p \times 1}`.
- Put an estimator's accent over the whole vector: `\hat{\vec\beta}`, not `\vec{\hat\beta}`.
- Colour with purpose, when a colour macro is defined:
  one colour for the focal or extra term,
  another for the shared term
  (teal rather than blue, which is hard to read in dark mode),
  and the same colour for a term and its later expansion,
  and for the two factors of a chain-rule step.
- Prefer constructs that render in all three formats:
  MathJax-only commands break the PDF,
  and LaTeX-only ones are dropped by the browser.

## Computer Algebra

Check a non-trivial derivative, integral, simplification or number
with SymPy (`python3 -c "import sympy; ..."`) or by computing it in code
before writing it down.

## Rendering

Before requesting review:

1. Render each parent chapter you changed (`quarto render chapters/<chapter>.qmd --to html`),
   and `--to pdf` where relevant.
2. Wait for it to finish, and check the exit code.
3. Read the output for warnings and fix them.
4. Open what the reader sees for anything you changed,
   including code output, tables, figures and math.

A full-site render renders every format in one `quarto render`;
never split it into per-format invocations (see `CLAUDE.md`).

## CI Workflow Debugging

- Read the failing job's log (the GitHub MCP tools `list_workflow_runs` and `get_job_logs`)
  before theorizing about a failure.
- Reproduce the failure locally, then show the same check passing.

## Determining Responsibility for Workflow Failures

You are responsible for failures in files or lines you changed,
or caused by your changes (a new dependency, a configuration change).
You are not responsible for pre-existing failures in code you did not touch,
or for infrastructure outages;
document those in the PR description instead of fixing them in an unrelated PR.

## Reporting

When summarizing PRs, issues, workflow runs or jobs,
include the GitHub URL with the reference
([#1](https://github.com/Morrison-Lab/dds/issues/1), not just "#1").
