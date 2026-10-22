#!/usr/bin/env python3
"""Self-tests for tools/check_site.py.

The checks pass vacuously on every page if their patterns never fire, or if
``main()`` never reports what a pattern finds. So the cases pinned here are
Pandoc's own output for a swallowed reference, unlinked ``data/*.csv`` inline
code, the nearby markup that must not count, and runs of ``main()`` over a
throwaway site that must fail or pass as expected.

Run with ``python3 tools/tests/test_check_site.py``. Needs no pytest.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import check_site as cs  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


# Verbatim output of `quarto pandoc -f markdown -t html` (Pandoc 3.10) for
#     1. Compare with
#         @exr-foo.
SWALLOWED = (
    '<ol type="1">\n<li>Compare with\n<ol class="example" type="1">\n'
    "<li></li>\n</ol></li>\n</ol>\n"
)
# Hand-written, not Pandoc output: an `<ol>` with `start=` before `class=`,
# to pin that attribute order does not matter.
LATER = '<ol start="2" class="example" type="1">\n<li>x</li>\n</ol>\n'

CASES = {
    "counts Pandoc's example list": (SWALLOWED, 1),
    "counts a list with start= before class=": (LATER, 1),
    "ignores a plain numbered list": ('<ol type="1"><li>x</li></ol>', 0),
    "ignores a class that only contains the word": ('<ol class="example-list"><li>x</li></ol>', 0),
    "counts example among other classes": ('<ol class="a example b"><li>x</li></ol>', 1),
    "counts single quotes and upper case": ("<OL CLASS='example'><li>x</li></OL>", 1),
    "ignores a data-class attribute": ('<ol data-class="example"><li>x</li></ol>', 0),
    "ignores markup inside a script body": (
        "<script>el.innerHTML = '<ol class=\"example\">';</script>",
        0,
    ),
}

for name, (html, expected) in CASES.items():
    got = cs.count_example_lists(html)
    check(name, got == expected, f"expected {expected}, got {got}")

CSV_CASES = {
    "flags bare inline data/*.csv": (
        "<p>In <code>data/wcgs.csv</code>, the outcome is binary.</p>",
        ["data/wcgs.csv"],
    ),
    "ignores linked data/*.csv": (
        '<p>Load <a href="data/wcgs.csv"><code>data/wcgs.csv</code></a>.</p>',
        [],
    ),
    "ignores data/*.csv inside a code block": (
        '<pre class="sourceCode python"><code>data/wcgs.csv</code></pre>',
        [],
    ),
    "ignores local save-path instructions": (
        "<p>Save it as <code>data/auto.csv</code> next to your code.</p>",
        [],
    ),
    "ignores local save-path instructions across a line break": (
        "<p>and save it as\n<code>data/boston.csv</code> next to your code.</p>",
        [],
    ),
    "ignores data/*.csv inside a script body": (
        "<script>el.innerHTML = '<code>data/wcgs.csv</code>';</script>",
        [],
    ),
}

for name, (html, expected_csvs) in CSV_CASES.items():
    got_csvs = cs.unlinked_data_csvs(html)
    check(name, got_csvs == expected_csvs, f"expected {expected_csvs}, got {got_csvs}")


def run_main(page_html: str, front_matter: str = "title: Page", marker: bool = False) -> tuple[int, str]:
    """Run main() over a one-page throwaway site; return (exit code, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "chapters").mkdir()
        (root / "chapters" / "page.qmd").write_text(f"---\n{front_matter}\n---\n", encoding="utf-8")
        (root / "_quarto-website.yml").write_text(
            "project:\n  render:\n    - chapters/page.qmd\n", encoding="utf-8"
        )
        (root / "_site" / "chapters").mkdir(parents=True)
        (root / "_site" / "chapters" / "page.html").write_text(page_html, encoding="utf-8")
        if marker:
            (root / "_site" / "preview-no-pdf.txt").write_text("no handouts\n", encoding="utf-8")
        cwd, argv = os.getcwd(), sys.argv
        err = io.StringIO()
        try:
            os.chdir(root)
            sys.argv = ["check_site.py"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = cs.main()
        finally:
            os.chdir(cwd)
            sys.argv = argv
        return code, err.getvalue()


code, err = run_main(f"<html><body>{SWALLOWED}</body></html>")
check("main() fails a page with an example list", code == 1, f"exit {code}")
check("main() names the example list", "Pandoc example list" in err, err)

code, err = run_main("<html><body><p>In <code>data/wcgs.csv</code></p></body></html>")
check("main() fails a page with an unlinked data CSV", code == 1, f"exit {code}")
check("main() names the unlinked data CSV", "unlinked data CSV reference(s) in prose (data/wcgs.csv)" in err, err)

code, err = run_main("<html><body><p>clean</p></body></html>")
check("main() passes a clean page", code == 0, f"exit {code}: {err}")

# A declared handout that the site lacks fails, unless the preview carries the
# marker that says it was built without PDFs (tools/skip-pdf-in-previews.py).
PDF_FRONT = "title: Page\nformat:\n  pdf:\n    output-file: page-handout.pdf"
CLEAN = "<html><body><p>clean</p></body></html>"

code, err = run_main(CLEAN, PDF_FRONT)
check("main() fails a missing handout", code == 1 and "page-handout.pdf" in err, f"exit {code}: {err}")

code, err = run_main(CLEAN, PDF_FRONT, marker=True)
check("main() accepts a missing handout in a no-PDF preview", code == 0, f"exit {code}: {err}")

if FAILURES:
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1)
print("\nall passed")
