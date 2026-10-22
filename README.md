# dds: Deep Learning notes

Lecture notes for CSCI 481, Deep Learning, at Western Washington University,
taught by Ezra Morrison in Spring 2027.
The site is a Quarto website, published at <https://morrison-lab.github.io/dds/>.

The course follows the module sequence of Brian Hutchinson's CSCI 481/581,
and builds on the DATA 571 notes,
[*Machine Learning for Data Science* (lds)](https://morrison-lab.github.io/lds/).

## The three repositories

| Repository | Holds |
| --- | --- |
| `Morrison-Lab/dds` (here) | The student-facing notes website |
| `Morrison-Lab/dlg` | Homework, exams, solutions and answer keys |
| `Morrison-Lab/mlr` | Copyrighted material, large files, and past offerings' originals |

All three are private.
Nothing that reveals an answer to a graded item belongs here.

## Building

```bash
git submodule update --init latex-macros
quarto render            # every format in one pass
quarto render --to html  # the website only, while drafting
python3 tools/check_site.py
```

Rendering needs R with `knitr`, `rmarkdown` and `reticulate`;
pages declare their Python packages with `reticulate::py_require()`,
and `reticulate` builds that environment itself.
The PDF handouts need a LaTeX toolchain (`quarto install tinytex`) and `rsvg-convert`.

The site's tools, filters, extensions and workflows were copied from
[`Morrison-Lab/lds`](https://github.com/Morrison-Lab/lds), which documents them in full.

See [`LICENSE.md`](LICENSE.md) for the license terms.
