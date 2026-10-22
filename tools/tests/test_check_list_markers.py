#!/usr/bin/env python3
"""Self-test for tools/check_list_markers.py.

The check passes vacuously if its pattern never fires, or if ``main()`` never
reports what the pattern finds. So the cases pinned here are the lines Pandoc
reads as list markers (each checked with ``quarto pandoc -t html``, Pandoc
3.10), the nearby lines it leaves as text, the contexts the check must skip,
and a run of ``main()`` over the poll that lds#370 found, which must fail.

Run with ``python3 tools/tests/test_check_list_markers.py``. Needs no pytest.
"""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import check_list_markers as clm  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


# Each line, as a continuation line of a list item, became a new list item.
MARKERS = [
    "    (C) either, or (D) neither?",
    "C) either",
    "(ii) two",
    "iv) four",
    "c. either",
    "C.  either",
    "x) y",
]
# Each line stayed text in the same position.
TEXT = [
    "C. either",
    "(ab) no",
    "e.g. no",
    "(A)first, with no space",
    "An (A) in the middle of a line",
]

for line in MARKERS:
    check(f"flags {line.strip()!r}", bool(clm.flagged_lines(line)))
for line in TEXT:
    check(f"leaves {line.strip()!r}", not clm.flagged_lines(line))

SKIPPED = {
    "backtick fence": "```{ojs}\n(d) => d[1]\n```\n",
    "tilde fence": "~~~\n(c) => c[0]\n~~~\n",
    "long fence holding a short one": "````\n```\n(a) x\n```\n````\n",
    "YAML front matter": "---\ntitle: x\n(a) y\n---\n",
    "multi-line HTML comment": "<!--\n(a) note\n-->\n",
}
for name, text in SKIPPED.items():
    check(f"skips a marker inside a {name}", not clm.flagged_lines(text))

check(
    "flags a marker after a closed fence",
    clm.flagged_lines("```\nx\n```\n(a) y\n") == [(4, "(a) y")],
)
check(
    "reports the right line number",
    clm.flagged_lines("text\n\n3.  Item\n    (C) either\n") == [(4, "    (C) either")],
)

# The poll from chapters/_subfiles/metrics/_exr-r-squared.qmd before lds#366.
POLL = (
    "3.  Could a negative $R^2$ be caused by\n"
    "    (A) an underfit model, (B) an overfit model,\n"
    "    (C) either, or (D) neither?\n"
)
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp) / "chapters"
    (root / "_subfiles" / "metrics").mkdir(parents=True)
    (root / "_subfiles" / "metrics" / "_exr-r-squared.qmd").write_text(POLL)
    (root / "outtakes").mkdir()
    (root / "outtakes" / "_old.qmd").write_text("(a) skipped\n")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        rc = clm.main(["--root", str(root)])
    report = out.getvalue()
    check("main() fails on the lds#370 poll", rc == 1, f"rc={rc}")
    check("main() reports the (C) line", ":3:" in report and "(C) either" in report, report)
    check("main() skips chapters/outtakes/", "outtakes" not in report, report)

    (root / "_subfiles" / "metrics" / "_exr-r-squared.qmd").write_text("Plain text.\n")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = clm.main(["--root", str(root)])
    check("main() passes once the poll is reflowed", rc == 0, f"rc={rc}")

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rc = clm.main(["--root", str(Path(tmp) / "missing")])
    check("main() errors on a missing root rather than passing", rc == 2, f"rc={rc}")

if FAILURES:
    print(f"{len(FAILURES)} case(s) failed")
    sys.exit(1)
print("All check_list_markers cases passed.")
