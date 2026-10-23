#!/usr/bin/env python3
"""Load the shared math macros (Morrison-Lab/macros) on every page.

Run as a Quarto pre-render script. It reads the `latex-macros` submodule's
`macros.qmd` and writes two header files that `_quarto-website.yml` includes
in every format:

- `_math-macros.html`, a MathJax configuration (HTML pages and decks).
  MathJax loads with `defer`, so this inline script runs first.
- `_math-macros.tex`, the same definitions for the PDF preamble.

The generated files are build products and are not committed.
Vectors are drawn with tildes, as the macros file defines them and as in the
lab's other repositories.

The macros file redefines `\\exp` to take a braced argument, so `\\exp(x)` and
`\\exp\\left(...\\right)` fail (a LaTeX error in the PDF, and literal braces in
the browser). Write `\\expf{x}`, which prints exp{x} with curly braces, the lab form.

The macros repository ships `macros-header.html`, which fetches `macros.qmd`
at run time; it looks for the file beside the page, which fails for pages
under `chapters/`. Hence this generator.
"""

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE = ROOT / "latex-macros" / "macros.qmd"
HTML_OUT = ROOT / "_math-macros.html"
TEX_OUT = ROOT / "_math-macros.tex"

COMMAND_START = re.compile(r"^\\(def|providecommand|newcommand|renewcommand)\b")
# TeX's delimited \def: `\def\sumi1n{...}` defines `\sumi`, to be followed by
# the literal text `1n`. LaTeX handles it; MathJax's macro table cannot.
DELIMITED_DEF = re.compile(r"^\\def\\([A-Za-z]+)([^A-Za-z{\s#\\][^{#]*)\{")


# The macros file defines the transpose as `\def\'{^{\top}}`, which in the PDF
# preamble replaces LaTeX's text accent \' everywhere, so any use in math
# fails (#495). Save the accent before the definition, then make \' a robust
# command that transposes in math mode and is the accent in text.
TEX_ACCENT_SAVE = "\\let\\ddsacute\\'\n"
TEX_ACCENT_RESTORE = (
    "\\DeclareRobustCommand{\\'}"
    "{\\ifmmode^{\\top}\\else\\expandafter\\ddsacute\\fi}\n"
)

# unicode-math's \mathbf prints nothing for Greek letters, so bold matrices such
# as \matr{\Eigvals} (bold Lambda) came out blank in the PDF. \symbfup is the
# same upright bold, and it has the Greek glyphs.
TEX_BOLD_GREEK = "\\AtBeginDocument{\\let\\mathbf\\symbfup}\n"


def brace_balance(text):
    balance = 0
    backslashes = 0
    for ch in text:
        if ch == "\\":
            backslashes += 1
            continue
        escaped = backslashes % 2 == 1
        backslashes = 0
        if ch == "{" and not escaped:
            balance += 1
        elif ch == "}" and not escaped:
            balance -= 1
    return balance


def brace_content(text, start):
    """Return (content, next index) for the brace group opening at `start`."""
    if start >= len(text) or text[start] != "{":
        return None
    depth = 0
    backslashes = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "\\":
            backslashes += 1
            continue
        escaped = backslashes % 2 == 1
        backslashes = 0
        if escaped:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    return None


def skip_space(text, i):
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def parse_definition(text):
    """Parse one definition into (name, body, argument count), or None."""
    if text.startswith("\\def\\"):
        i = len("\\def\\")
        name = ""
        while i < len(text) and not (text[i].isspace() or text[i] == "{"):
            name += text[i]
            i += 1
        body = brace_content(text, skip_space(text, i))
        # A TeX control sequence is a run of letters or one other character;
        # anything else (`\def\name#1{...}`) is a signature this parser
        # does not read, so it goes to the failures.
        valid_name = name.isalpha() or (len(name) == 1 and not name.isalpha())
        if not valid_name or body is None:
            return None
        args = [int(n) for n in re.findall(r"#(\d)", body[0])]
        return name, body[0], max(args, default=0)
    match = re.match(r"^\\(providecommand|newcommand|renewcommand)", text)
    if not match:
        return None
    i = skip_space(text, match.end())
    name_block = brace_content(text, i)
    if name_block is None or not name_block[0].startswith("\\"):
        return None
    name = name_block[0][1:]
    i = skip_space(text, name_block[1])
    argc = 0
    if i < len(text) and text[i] == "[":
        close = text.index("]", i)
        argc = int(text[i + 1:close] or 0)
        i = skip_space(text, close + 1)
    body = brace_content(text, i)
    if body is None:
        return None
    return name, body[0], argc


def parse_macros(source, failures=None, delimited=None):
    r"""Parse every definition.

    Delimited `\def`s go to `delimited` (they work in the PDF only), and any
    other definition that cannot be parsed goes to `failures`.
    """
    macros = {}
    current = ""
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("<!--"):
            continue
        if current or COMMAND_START.match(line):
            current = f"{current}\n{line}" if current else line
        else:
            continue
        balance = brace_balance(current)
        if balance < 0:
            current = ""
            continue
        if balance != 0:
            continue
        delim = DELIMITED_DEF.match(current)
        if delim:
            if delimited is not None:
                delimited.append(delim.group(1) + delim.group(2))
            current = ""
            continue
        parsed = parse_definition(current)
        if parsed:
            name, body, argc = parsed
            macros[name] = [body, argc] if argc else body
        elif failures is not None:
            failures.append(current)
        current = ""
    return macros


def main():
    if not SOURCE.exists():
        sys.exit(
            f"{SOURCE.relative_to(ROOT)} is missing: run "
            "`git submodule update --init latex-macros`"
        )
    source = SOURCE.read_text(encoding="utf-8")
    failures = []
    delimited = []
    macros = parse_macros(source, failures, delimited)
    if failures:
        # A definition this parser cannot read (parameter text in a \def
        # signature, an optional-argument default, ...) would otherwise vanish
        # from the HTML pages while still working in the PDF.
        listed = "\n  ".join(f.splitlines()[0] for f in failures)
        sys.exit(f"build_math_macros: cannot parse {len(failures)} definition(s):\n  {listed}")
    if len(macros) < 100:
        sys.exit(f"parsed only {len(macros)} macros from {SOURCE}; refusing to write")

    config = json.dumps(macros, ensure_ascii=False, sort_keys=True)
    html = (
        "<!-- Generated by tools/build_math_macros.py from latex-macros/macros.qmd. -->\n"
        "<script>\n"
        "window.MathJax = window.MathJax || {};\n"
        "window.MathJax.tex = window.MathJax.tex || {};\n"
        "window.MathJax.tex.macros = Object.assign({}, window.MathJax.tex.macros || {},\n"
        f"  {config});\n"
        "</script>\n"
    )
    # The preamble takes the source verbatim, less any Markdown comment line,
    # which LaTeX would read as text.
    preamble = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("<!--")
    )
    tex = (
        "% Generated by tools/build_math_macros.py from latex-macros/macros.qmd.\n"
        f"{TEX_ACCENT_SAVE}"
        f"{preamble}\n"
        f"{TEX_ACCENT_RESTORE}"
        f"{TEX_BOLD_GREEK}"
    )
    for path, text in [(HTML_OUT, html), (TEX_OUT, tex)]:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            path.write_text(text, encoding="utf-8")
    print(f"build_math_macros: {len(macros)} macros")
    if delimited:
        print(
            "build_math_macros: PDF only (delimited \\def, which MathJax cannot "
            "define): " + ", ".join("\\" + d for d in delimited)
        )


if __name__ == "__main__":
    main()
