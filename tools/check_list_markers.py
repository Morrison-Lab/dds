#!/usr/bin/env python3
"""Flag source lines that Pandoc would read as a lettered or roman list marker.

Pandoc's ``fancy_lists`` extension reads a line that opens with ``(A)``,
``C)``, ``c.``, ``(ii)``, ``iv)`` and the like as an ordered-list marker, even
as a continuation line inside a list item. The rest of the line becomes a new
list, and Pandoc renumbers it, so a poll written as

    3.  Could it be
        (A) underfit, (B) overfit,
        (C) either, or (D) neither?

renders "(B) either": the third line became the second item of a list that
started at (A). Nothing in the build warns (lds#370). An upper-case letter or
roman numeral followed by a period counts only before two or more spaces,
since "C. Smith" is left alone.

The site uses no lettered or roman lists on purpose, so every such line under
``chapters/`` is reported, except inside fenced code (where OJS arrow
functions such as ``(d) => d[1]`` start lines), YAML front matter, and HTML
comments, none of which Pandoc reads as Markdown text. ``chapters/outtakes/``
is skipped, as it is not rendered.

Reflow the line so the marker does not open it.

Run from the repository root: ``python3 tools/check_list_markers.py``.
Exits 1 if any line is flagged. Needs only the standard library.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROMAN_LOWER = r"[ivxlcdm]+"
ROMAN_UPPER = r"[IVXLCDM]+"
LETTER = r"[A-Za-z]"

MARKER = re.compile(
    r"^\s*(?:"
    rf"\((?:{LETTER}|{ROMAN_LOWER}|{ROMAN_UPPER})\)\s"  # (A)  (c)  (ii)
    rf"|(?:{LETTER}|{ROMAN_LOWER}|{ROMAN_UPPER})\)\s"  # A)  c)  iv)
    rf"|(?:[a-z]|{ROMAN_LOWER})\.\s"  # c.  ii.
    rf"|(?:[A-Z]|{ROMAN_UPPER})\.\s{{2}}"  # C.<two spaces>
    r")"
)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})")


def flagged_lines(text: str) -> list[tuple[int, str]]:
    """Return ``(line number, line)`` for each line that opens with a marker."""
    hits: list[tuple[int, str]] = []
    lines = text.splitlines()
    fence: str | None = None
    in_comment = False
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                start = i + 1
                break
    for number, line in enumerate(lines[start:], start=start + 1):
        if fence is not None:
            m = FENCE.match(line)
            if m and m.group(1)[0] == fence[0] and len(m.group(1)) >= len(fence):
                fence = None
            continue
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        m = FENCE.match(line)
        if m:
            fence = m.group(1)
            continue
        if line.lstrip().startswith("<!--"):
            if "-->" not in line:
                in_comment = True
            continue
        if MARKER.match(line):
            hits.append((number, line))
    return hits


def sources(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*.qmd")
        if "outtakes" not in p.relative_to(root).parts
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=Path("chapters"), type=Path)
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"{args.root} not found -- run from the repository root", file=sys.stderr)
        return 2
    files = sources(args.root)
    if not files:
        print(f"no .qmd files under {args.root}", file=sys.stderr)
        return 2
    count = 0
    for path in files:
        for number, line in flagged_lines(path.read_text(encoding="utf-8")):
            count += 1
            print(
                f"{path}:{number}: opens with a list marker Pandoc will read "
                f"as a new list item; reflow it: {line.strip()}"
            )
    if count:
        print(f"{count} line(s) flagged in {len(files)} file(s)")
        return 1
    print(f"no list-marker lines in {len(files)} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
