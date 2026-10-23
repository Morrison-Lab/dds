#!/usr/bin/env python3
"""Report definition and result divs not immediately followed by an example.

Expands each chapter's includes, then walks its top-level blocks.  A `#def-`,
`#thm-`, `#lem-`, `#prp-` or `#cor-` div counts as followed when the next block
(skipping slide breaks, blank lines, HTML comments and supporting divs:
`proof`, `notes`, `reading`, `videos`, callouts, `#rem-` remarks and `#fig-`
figures, whether a div or a code chunk labelled `fig-`) is an
`#exm-` div, or an `#exr-` exercise that applies it.  `--strict` accepts only
`#exm-`.  A run of adjacent `#def-` divs shares the example after the run.
Warns only (exit 0) unless `--fail` is given.
"""
import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INCLUDE = re.compile(r"^\s*\{\{<\s*include\s+(\S+)\s*>\}\}\s*$")
FENCE = re.compile(r"^(:{3,})\s*(\{.*\}|\S*)\s*$")
SLIDEBREAK = re.compile(r"^\{\{<\s*slidebreak\s*>\}\}$")
SUPPORTING = {"proof", "notes", "reading", "videos",
              "callout-note", "callout-tip", "callout-important", "callout-warning"}
TARGETS = ("def", "thm", "lem", "prp", "cor")
CHUNK_LABEL = re.compile(r"#\|\s*label:\s*(\S+)")


def expand(path, seen=()):
    """Return the file's lines with includes expanded, as (file, line, text)."""
    out = []
    in_code = False
    for n, text in enumerate(path.read_text().splitlines(), 1):
        if text.lstrip().startswith("```"):
            in_code = not in_code
        m = INCLUDE.match(text)
        if m and not in_code:
            inc = (path.parent / m.group(1)).resolve()
            if inc.exists() and inc not in seen:
                out += expand(inc, seen + (path,))
                continue
        out.append((path, n, text))
    return out


def blocks(lines):
    """Yield top-level blocks as (kind, id, classes, file, line)."""
    i, in_code, in_comment = 0, False, False
    while i < len(lines):
        f, n, t = lines[i]
        if not in_code and t.lstrip().startswith("```"):
            # A code chunk labelled `fig-` is a figure: one block, however long.
            j = i + 1
            while j < len(lines) and not lines[j][2].lstrip().startswith("```"):
                j += 1
            label = next((m.group(1) for _, _, x in lines[i + 1:j]
                          if (m := CHUNK_LABEL.match(x.strip()))), "")
            if label.startswith("fig-"):
                yield ("div", label, ["figure"], f, n)
                i = j + 1
                continue
        if t.lstrip().startswith("```"):
            in_code = not in_code
        m = None if in_code else FENCE.match(t)
        if m and m.group(2):
            colons, attrs = len(m.group(1)), m.group(2)
            ident = re.search(r"#([\w-]+)", attrs)
            classes = re.findall(r"\.([\w-]+)", attrs) or [attrs]
            depth, j = 1, i + 1
            while j < len(lines) and depth:
                mm = FENCE.match(lines[j][2])
                if mm and len(mm.group(1)) == colons and not mm.group(2):
                    depth -= 1
                elif mm and mm.group(2) and len(mm.group(1)) == colons:
                    depth += 1
                j += 1
            yield ("div", ident.group(1) if ident else "", classes, f, n)
            i = j
            continue
        s = t.strip()
        if in_comment or s.startswith("<!--"):
            # An HTML comment, possibly over several lines, is not content.
            in_comment = "-->" not in s
        elif s and not SLIDEBREAK.match(s):
            yield ("text", "", [], f, n)
        i += 1


def audit(chapter, strict=False):
    """Return (id, file, line, ok, next) for each definition and result div.

    A run of adjacent definitions is followed when the block after the run is
    an example; a theorem-type div must be followed by its own.
    """
    bl = [b for b in blocks(expand(chapter))
          if not SUPPORTING.intersection(b[2])
          and not b[1].startswith(("rem-", "fig-"))]
    res = []
    for k, b in enumerate(bl):
        if b[0] != "div" or b[1].split("-")[0] not in TARGETS:
            continue
        j = k + 1
        if b[1].startswith("def-"):
            while j < len(bl) and bl[j][0] == "div" and bl[j][1].startswith("def-"):
                j += 1
        nxt = bl[j] if j < len(bl) else None
        ok = bool(nxt and nxt[0] == "div" and (
            nxt[1].startswith("exm-") or (not strict and nxt[1].startswith("exr-"))))
        label = nxt[1] if nxt and nxt[0] == "div" else (nxt[0] if nxt else "end")
        res.append((b[1], b[3], b[4], ok, label))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="accept only #exm- divs, not #exr- exercises")
    ap.add_argument("--list", action="store_true", help="list every div, not only the unfollowed")
    ap.add_argument("--fail", action="store_true", help="exit 1 when any div lacks an example (default: warn only)")
    ap.add_argument("--root", type=lambda p: Path(p).resolve(), default=ROOT,
                    help="repository root (default: the one holding this script)")
    a = ap.parse_args(argv)
    bad = total = 0
    for ch in sorted((a.root / "chapters").glob("*.qmd")):
        for ident, f, n, ok, nxt in audit(ch, a.strict):
            total += 1
            if not ok:
                bad += 1
                rel = f.relative_to(a.root)
                print(f"{ch.name}\t{ident}\t{rel}:{n}\tnext={nxt}")
                if os.environ.get("GITHUB_ACTIONS"):
                    print(f"::warning file={rel},line={n}::{ident} is not immediately followed by an example")
            elif a.list:
                print(f"{ch.name}\t{ident}\tOK({nxt})")
    print(f"{bad} of {total} definition and result divs lack an example", file=sys.stderr)
    return 1 if bad and a.fail else 0


if __name__ == "__main__":
    sys.exit(main())
