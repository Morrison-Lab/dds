# Project memory for dds

Working notes that agents found while writing chapters,
kept here so a later session does not have to learn them again.

## Writing chapters (2026-11-04)

Learned while writing the dl-experiments, applications and ethics chapters.

- Inline R cannot call `py$len(...)` or read a tensor's shape.
  Compute the value or the formatted string in Python, and read it with `py$name`.
- Pandoc parses only CSL terms as citation locators (section, chapter, figure, page).
  "table 4" is not one, so name a table in the prose instead.
- `AUTHORING.md` makes a calculation shown in an example an exercise,
  so an example should rest on published figures.
- Caption a table with the div form (`::: {#tbl-x}` around the chunk, caption last), not `tbl-cap`.
- In dds prose, lds is "the DATA 471 notes".
- gha `check-term-links` treats each blank-line paragraph as a passage,
  and learns terms from every page,
  so a new definition (such as "model card") can make it flag a sentence on another page.
- Pretraining a small MLP on half of optdigits gains nothing over training it fresh,
  so do not build a lesson on that comparison.
- Guard every claim in a figure's alt text with an `assert`; the Claude review asks for one.
- A thread with several chapters can run them through its one branch in turn:
  once its PR merges, restart the branch from main for the next chapter,
  rather than asking for extra branches (2026-11-04, dds#34 then dds#36).
- gpt-oss:120b on Ollama cloud misreads reshapes and R indexing of `py$` values;
  answer with the shape or value the render printed, then re-run it.
- Fetch the gha checkers fresh before a push; a scratch copy goes stale.
  check-definite-antecedents gained a bare-demonstrative rule ("Of these,", "That is")
  that a copy fetched hours earlier did not have (dds#36, 2026-11-04).
  The script is `check-definite-antecedents/check-definite-antecedents.py` in gha at `v3`.
