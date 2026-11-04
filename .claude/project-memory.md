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
