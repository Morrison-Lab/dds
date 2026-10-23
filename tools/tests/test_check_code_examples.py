#!/usr/bin/env python3
"""Self-test for tools/check_code_examples.py.

The checker's whole value is that a clean report means every snippet ran, so
the cases worth pinning are the ones where it could report clean having run
nothing. Each of those is exercised here against a throwaway chapter
directory; nothing touches the repository's own chapters.

Run with ``python3 tools/tests/test_check_code_examples.py``. Needs no pytest.
"""

from __future__ import annotations

import io
import contextlib
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import check_code_examples as cce  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


def run_checker(files: dict[str, str], *argv: str) -> tuple[int, str]:
    """Write a throwaway chapters dir, run main() over it, return (code, output)."""
    with tempfile.TemporaryDirectory() as tmp:
        chapters = Path(tmp) / "chapters"
        chapters.mkdir()
        for name, body in files.items():
            target = chapters / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cce.main(["--chapters", str(chapters), *argv])
        return code, buf.getvalue()


def block(lang: str, body: str) -> str:
    return f"```{lang}\n{body}\n```\n"


HAVE = {lang: shutil.which(spec["cmd"][0]) is not None
        for lang, spec in cce.LANGUAGES.items()}


print("extraction")

code, out = run_checker({"a.qmd": "# nothing here\n"})
check("a named directory with no snippets is a failure, not a clean run",
      code == 1)
check("...and says so", "examined nothing" in out)


def run_default(files: dict[str, str], *argv: str) -> tuple[int, str]:
    """Like run_checker, but with no --chapters: the default population.

    The module's CHAPTERS is pointed at a throwaway directory for the call,
    so the repository's own chapters are never read.
    """
    saved = cce.CHAPTERS
    with tempfile.TemporaryDirectory() as tmp:
        chapters = Path(tmp) / "chapters"
        chapters.mkdir()
        for name, body in files.items():
            (chapters / name).write_text(body, encoding="utf-8")
        cce.CHAPTERS = chapters
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cce.main(list(argv))
        finally:
            cce.CHAPTERS = saved
        return code, buf.getvalue()


code, out = run_default({"a.qmd": "# prose\n\n```{r}\n1 + 1\n```\n"})
check("the default run over chapters with no plain fence exits 0", code == 0, out)
check("...and says there is nothing to check",
      "no plain code fences found under chapters/: nothing to check" in out, out)
check("...not that it examined nothing", "examined nothing" not in out, out)

code, out = run_default({})
check("the default run over a directory with no .qmd file still fails",
      code == 1, out)

code, out = run_default({"a.qmd": "# prose\n"}, "--lang", "r")
check("a --lang filter matching nothing still fails", code == 1, out)

code, out = run_default({"a.qmd": "```python\nprint(1)\n"})
check("the default run still fails on a document defect with no snippets",
      code == 1, out)

crlf = block("python", "print(1)").replace("\n", "\r\n")

with tempfile.TemporaryDirectory() as tmp:
    p = Path(tmp) / "c.qmd"

    p.write_text(crlf, encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("a CRLF fence is extracted", len(blocks) == 1, repr(blocks))
    # Not a regression test for a bug that existed: read_text already strips
    # carriage returns, so the original regex handled a CRLF file too. It
    # guards against a future switch to read_bytes or newline="".
    check("...with no carriage returns in the body",
          blocks and "\r" not in blocks[0][2], repr(blocks))

    p.write_text("- item\n\n  ```python\n  print(1)\n  ```\n", encoding="utf-8")
    blocks, _ = cce.extract(p)
    check("an indented fence is extracted", len(blocks) == 1, repr(blocks))
    check("...with its indentation stripped",
          blocks and blocks[0][2].startswith("print(1)"), repr(blocks))

    p.write_text('```python filename="a.py"\nprint(1)\n```\n', encoding="utf-8")
    blocks, _ = cce.extract(p)
    check("an attributed info string is extracted", len(blocks) == 1, repr(blocks))

    p.write_text("````python\nx = '''\n```\n'''\nprint(len(x))\n````\n", encoding="utf-8")
    blocks, _ = cce.extract(p)
    check("an inner ``` does not truncate a ```` block",
          len(blocks) == 1 and "print(len(x))" in blocks[0][2], repr(blocks))

    p.write_text("```python\nprint(1)\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("an unterminated fence is reported, not dropped",
          not blocks and len(problems) == 1, repr((blocks, problems)))

    p.write_text("```text\nnot code\n```\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("an unrecognised language is ignored quietly",
          not blocks and not problems, repr((blocks, problems)))

    # Both cases above name a recognised language, so narrowing the report
    # back to `if lang:` left the suite green while turning exit 1 into exit
    # 0 on a malformed document.
    p.write_text("```text\ndangling\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("an unterminated fence is reported whatever the language",
          not blocks and len(problems) == 1
          and "unterminated unrecognised fence" in problems[0][1],
          repr((blocks, problems)))

    # Executable chunks share a session and read page-relative data, so run
    # alone they fail for no fault of their own; the render runs them.
    p.write_text("```{r}\nx <- read.csv('data/a.csv')\n```\n"
                 "```{python setup, echo=FALSE}\nprint(x)\n```\n"
                 "```{r, include=FALSE}\nx\n```\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("executable chunks are left to the render",
          not blocks and not problems, repr((blocks, problems)))

    p.write_text("```{.python}\nprint(1)\n```\n```{#lst-a .r}\n1\n```\n"
                 "```python\nprint(1)\n```\n", encoding="utf-8")
    blocks, _ = cce.extract(p)
    check("highlighted-only and plain fences are still run",
          [b[1] for b in blocks] == ["python", "r", "python"], repr(blocks))

    check("a braced bare language is an executable chunk",
          cce.is_executable_chunk("{python}") and cce.is_executable_chunk("{r label}"))
    check("a dotted or anchored language is not",
          not cce.is_executable_chunk("{.python}")
          and not cce.is_executable_chunk("{#lst-x .r}")
          and not cce.is_executable_chunk("python"))

    # A shorter inner run, or a different fence character, cannot close the
    # outer fence, so neither is evidence that anything was swallowed.
    p.write_text("````text\n```python\nprint(1)\n```\n````\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("a quoted fence inside a longer fence is well-formed",
          not blocks and not problems, repr((blocks, problems)))

    p.write_text("~~~text\n```python\nprint(1)\n```\n~~~\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("a backtick fence inside a tilde fence is well-formed",
          not blocks and not problems, repr((blocks, problems)))

    p.write_text("```text\n```python\nprint(1)\n```\n", encoding="utf-8")
    blocks, problems = cce.extract(p)
    check("a nested fence that could have closed the outer one is reported",
          any("nested" in why for _, why in problems), repr((blocks, problems)))

    # A STRICTLY longer inner run is the case the equal-run test above cannot
    # see: narrowing the guard's `>=` to `==` left the whole suite green while
    # turning exit 1 into exit 0 here. Measured against pandoc 3.1.3, the
    # inner block's four-backtick closer ends the outer block early, the line
    # after it escapes as a paragraph, and the outer closer opens a new
    # unterminated block.
    p.write_text(
        "```text\n````python\nprint(1)\n````\nafter\n```\n", encoding="utf-8"
    )
    blocks, problems = cce.extract(p)
    check("a LONGER nested fence is reported too",
          any("nested" in why for _, why in problems), repr((blocks, problems)))

print("language filter")

# --lang was exercised by no case at all, so inverting its comparison from
# != to ==, or deleting the filter outright, left the whole suite green while
# making a filtered run report clean over the snippets it had skipped -- a
# pass over a population never examined, which is the one outcome every guard
# in this file exists to make unreachable.
#
# A tally alone cannot see that: over two passing snippets, running the wrong
# one still reports "ran 1 snippet(s)". The case needs an asymmetry, so the
# unselected language fails.
if HAVE["python"]:
    two_langs = {
        "a.qmd": block("python", "print(1)") + "\n" + block("r", 'stop("boom")')
    }
    code, out = run_checker(two_langs, "--lang", "python")
    check("--lang runs the language asked for", code == 0, out)
    check("...and only that one", "ran 1 snippet(s)" in out, out)

if HAVE["r"]:
    code, out = run_checker(
        {"a.qmd": block("python", "print(1)") + "\n" + block("r", 'stop("boom")')},
        "--lang", "r",
    )
    check("--lang r selects the failing R snippet, not the passing Python one",
          code == 1, out)
    check("...and runs exactly one", "ran 1 snippet(s)" in out, out)
    # Without this clause a filtered run ends on a sentence identical to a
    # whole-corpus one, so a broken block in the excluded language reads as
    # absent rather than as unexamined.
    check("...and says how many the filter held back",
          "1 excluded by --lang r" in out, out)

print("verdicts")

code, out = run_checker({"a.qmd": "```python\nprint(1)\n"})
check("an unterminated fence fails the run", code == 1, out)

# The two vacuous passes the module header calls unreachable. Neither was
# pinned before: a mutation turning each of these verdicts into "pass" left the
# whole suite green, so the checker could report passes over code it never ran
# -- the outcome the header says is worse than having no checker at all.
if HAVE["python"]:
    original_timeout = cce.TIMEOUT_SECONDS
    cce.TIMEOUT_SECONDS = 1
    try:
        code, out = run_checker(
            {"a.qmd": block("python", "import time\ntime.sleep(30)")}
        )
    finally:
        cce.TIMEOUT_SECONDS = original_timeout
    check("a snippet that times out fails the run", code == 1, out)
    check("...and is counted as a failure, not a pass",
          "1 failed" in out, out)

# The missing-interpreter branch cannot be reached by naming a language this
# machine lacks, because every such case is gated on HAVE[lang] -- which is
# false in exactly the environment where the branch fires. Pointing a present
# language at an absent binary reaches it on any machine.
if HAVE["python"]:
    original_cmd = cce.LANGUAGES["python"]["cmd"]
    cce.LANGUAGES["python"]["cmd"] = ["python3-does-not-exist"]
    try:
        code, out = run_checker({"a.qmd": block("python", "print(1)")})
    finally:
        cce.LANGUAGES["python"]["cmd"] = original_cmd
    check("a missing interpreter skips rather than passing",
          "1 skipped" in out, out)
    check("...and makes the run non-zero without --allow-skips",
          code == 1, out)

# Julia's `#=` opens a block comment, but not when a bare `#` has already
# started a line comment ahead of it. Reading it as an opener there swallowed
# every line below, so a good block was reported as having no executable line
# AND dropped from the run -- the `#===` banner failure, one language over.
check("a Julia line comment naming #= does not open a block comment",
      not cce.is_effectively_empty(
          "# Julia opens a block comment with #=\nprintln(1)\n", "julia"))
check("...while a real Julia block comment still reads as empty",
      cce.is_effectively_empty("#= hidden()\nmore()\n=#\n", "julia"))

if HAVE["python"]:
    code, out = run_checker(
        {"a.qmd": block("python", "print(1)") + "\n```text\ndangling\n"}
    )
    check("a malformed document fails even with every snippet passing",
          code == 1, out)
    check("...and the summary line says so",
          "1 document defect(s)" in out, out)

# Every block runs in a fresh interpreter, so a helper two blocks both need is
# carried by both. That duplication is load-bearing and cannot be merged away;
# what it costs is a silent failure, since editing one copy leaves both blocks
# running and printing while the chapter reports two estimators as one.
if HAVE["python"]:
    def pair(first, second):
        return block("python", first) + "\n" + block("python", second)

    same = "def ridge(X):\n    return X\nprint(ridge(1))"
    code, out = run_checker({"a.qmd": pair(same, same)})
    check("two identical copies of a helper are not a defect", code == 0, out)

    drifted = "def ridge(X):\n    return X + 1\nprint(ridge(1))"
    code, out = run_checker({"a.qmd": pair(same, drifted)})
    check("a helper redefined differently in a second block fails the run",
          code == 1, out)
    check("...and the report names the function and the first copy's line",
          "`ridge` differs from the copy at line" in out, out)

    # Exactly, not modulo comments: a comment is where a deliberate divergence
    # would be explained, so a pair that differs only there is still drift.
    commented = "def ridge(X):\n    return X  # same code\nprint(ridge(1))"
    code, out = run_checker({"a.qmd": pair(same, commented)})
    check("...including when only a comment differs", code == 1, out)

    # Same name, different language: R's `ridge` is not Python's, and reporting
    # it would fire on every parallel-implementation section in the repository.
    # Gated on HAVE["r"] as well, like every other mixed-language case here: the
    # chapter carries an R block, so without Rscript that block comes back a
    # skip, and a skip alone makes the run non-zero unless --allow-skips is
    # passed. Ungated, this asserted `code == 0` on a machine that has only
    # Python and failed there for a reason that has nothing to do with
    # redefinition -- which is exactly the machine this suite is pitched at,
    # since CLAUDE.md documents it as the check needing no interpreter beyond
    # Python.
    if HAVE["r"]:
        code, out = run_checker(
            {"a.qmd": block("python", same) + "\n"
                      + block("r", "ridge <- function(X) X + 1\nprint(ridge(1))")}
        )
        check("the same name in another language is not a redefinition",
              code == 0, out)

    # The population is one chapter. Two chapters that each define a helper
    # are independent documents, and a reader meets only one of them.
    code, out = run_checker(
        {"a.qmd": block("python", same), "b.qmd": block("python", drifted)}
    )
    check("a helper redefined in another chapter is not a defect", code == 0, out)

# No real R or Julia snippet reaches the guard's language test, because none of
# them parses as Python AND carries a top-level `def`; `ast.parse` refuses them
# first. So the test above passes with the language filter deleted, and pinning
# the filter means calling the function directly with a block the extractor
# would never produce. The filter is kept for what it says rather than what it
# catches -- `ast` refusing R is an accident of two grammars, not a guarantee.
check("a non-python block is not parsed as python",
      cce.redefinitions([
          (1, "python", "def ridge(X):\n    return X\n"),
          (9, "r", "def ridge(X):\n    return X + 1\n"),
      ]) == [])
check("...while two python blocks in that same shape are reported",
      len(cce.redefinitions([
          (1, "python", "def ridge(X):\n    return X\n"),
          (9, "python", "def ridge(X):\n    return X + 1\n"),
      ])) == 1)

if HAVE["python"]:
    code, out = run_checker({"a.qmd": block("python", "print(1)")})
    check("a working snippet passes", code == 0, out)

    code, out = run_checker({"a.qmd": block("python", "raise SystemExit(3)")})
    check("a failing snippet fails", code == 1, out)

    code, out = run_checker({"a.qmd": block("python", "import nummpy_typo_xyz")})
    check("a missing package is a skip", "skip" in out, out)
    check("...and a skip fails the run by default", code == 1, out)

    code, out = run_checker(
        {"a.qmd": block("python", "import nummpy_typo_xyz")}, "--allow-skips"
    )
    check("...unless --allow-skips is given", code == 0, out)

    printer = block(
        "python",
        'print("ModuleNotFoundError: No module named \'x\'")\nraise ValueError("real")',
    )
    code, out = run_checker({"a.qmd": printer})
    skipped = any(line.startswith("skip ") for line in out.splitlines())
    check("a snippet that PRINTS a missing-package message still fails",
          code == 1 and not skipped, out)

    pair = (
        block("python", "open('leftover.txt', 'w').write('x')")
        + "\n"
        + block("python", "import os; assert not os.path.exists('leftover.txt')")
    )
    code, out = run_checker({"a.qmd": pair})
    check("each snippet gets its own working directory", code == 0, out)
else:
    print("  (python3 not on PATH: verdict cases skipped)")

print()
print("routes to a vacuous pass")

# A stray fence in prose steals the closer of the block after it. The parse is
# correct and the document is not, and nothing else would notice.
code, out = run_checker(
    {"a.qmd": block("python", 'print("runs")') + "\n```\n\n"
     + block("python", "raise SystemExit(3)")}
)
check("a block swallowed by a stray fence is reported", code == 1)
check("...and names it as malformed", "malformed" in out)

# Every interpreter accepts an empty file, so without this an emptied block is
# indistinguishable in the tally from one that ran.
code, out = run_checker({"a.qmd": "```python\n```\n"})
check("an empty fence is reported, not counted as a pass", code == 1)
check("...and says it has no executable line", "no executable line" in out)

code, out = run_checker({"a.qmd": block("python", "# print(1)")})
check("a comment-only fence is reported too", code == 1)

# Julia alone also has a BLOCK comment, whose interior lines begin with
# neither `#` nor whitespace -- so the line test above reads a block
# commented out that way as executable. It runs, exits 0, and is tallied as a
# pass over a no-op: the outcome this guard exists to prevent, one language
# down. Confirmed against julia 1.12.7: the file below exits 0 and prints
# nothing.
code, out = run_checker({"a.qmd": block("julia", '#=\nprintln(1)\nx = 1\n=#')})
check("a Julia block-commented fence is reported too", code == 1, out)
check("...and says it has no executable line", "no executable line" in out, out)

# Julia nests these, so the scan counts depth rather than stopping at the
# first `=#`. Closing one level too early leaks the tail (` c =#` here) and
# the block reads as executable again.
code, out = run_checker({"a.qmd": block("julia", "#= a #= b =# c =#")})
check("a NESTED Julia block comment is reported too", code == 1, out)

# The converse: real code after a block comment must still run, or the guard
# would reject every Julia snippet carrying a header comment.
check("a Julia block comment above real code is not empty",
      not cce.is_effectively_empty("#= note =#\nprintln(1)\n", "julia"))
check("...nor is code on the closing line",
      not cce.is_effectively_empty("#=\na\n=# println(2)\n", "julia"))

# glob is not recursive; rglob is.
if HAVE["python"]:
    code, out = run_checker(
        {"ok.qmd": block("python", 'print("runs")'),
         "sub/a.qmd": block("python", "raise SystemExit(3)")}
    )
    check("a chapter in a subdirectory is examined", code == 1)
    check("...and both snippets are counted", "ran 2 snippet(s)" in out)

# An unrecognised tag makes a block invisible; a known alias must resolve.
check("py resolves to python", cce.language_of("py") == "python")
check("jl resolves to julia", cce.language_of("jl") == "julia")
check("Rscript resolves to r", cce.language_of("Rscript") == "r")
check("python3 resolves to python", cce.language_of("python3") == "python")
check("an unrelated tag is still None", cce.language_of("bash") is None)

if HAVE["python"]:
    code, out = run_checker({"a.qmd": block("py", "raise SystemExit(3)")})
    # `code == 1` alone passes with ALIASES emptied, because the run then
    # exits 1 via "no snippets found" rather than via the block failing. The
    # failure tally is what discriminates.
    check("an aliased block is run, not skipped over",
          code == 1 and "1 failed" in out, out)
    check("...and counted", "ran 1 snippet(s)" in out)

    code, out = run_checker({"a.qmd": block("{.python}", "raise SystemExit(3)")})
    check("a Pandoc-attribute fence is run, not skipped over",
          code == 1 and "1 failed" in out, out)

    # A cross-referenceable block is a Pandoc listing, whose id comes FIRST:
    # reading only the leading token returns None, so anchoring a block --
    # the fix for a forward reference in the prose -- would silently drop it
    # from the run. Found by trying to use the feature, not by review.
    code, out = run_checker(
        {"a.qmd": block("{#lst-demo .python}", "raise SystemExit(3)")}
    )
    check("a listing with an id is run, not skipped over",
          code == 1 and "1 failed" in out, out)

# An attribute's VALUE must not resolve through its own right-hand side, or a
# `filename="setup.r"` on a text block would run that block under Rscript.
check("a language named only in an attribute value is not a language",
      cce.language_of('{#lst-x lang="python"}') is None)
check("...nor is one inside a filename", cce.language_of('{.text filename="r"}') is None)
# Pandoc accepts spaces around the `=`, so the value stands alone as its own
# token. The first cut of this guard skipped any token CONTAINING `=`, which
# never fires and left these three resolving -- a text block would have been
# run under Rscript. A guard that cannot fire reads as protection while
# protecting nothing.
check("...nor after a spaced `=`", cce.language_of("{.text filename = r}") is None)
check("...nor after a trailing `=`", cce.language_of("{#a k= julia}") is None)
check("...nor before a leading `=`", cce.language_of("{#a k =julia}") is None)
check("a class still resolves past a spaced attribute",
      cce.language_of("{.python k = v}") == "python")
check("an id alone resolves to nothing", cce.language_of("{#lst-x}") is None)
check("a listing id plus a class resolves to the class",
      cce.language_of("{#lst-a .r}") == "r")
check("an attribute after the class does not displace it",
      cce.language_of('{.julia code-fold="true"}') == "julia")

# A caption is prose, and Quarto will not resolve a listing crossref without
# one (measured against Quarto 1.10.18: the reference renders `?@lst-name`
# and the render warns). So an anchored block always carries a caption, and a
# caption may name a language it is not written in. A whitespace split leaves
# that word standing alone, which is how a token loop resolved both of these
# to r -- the second of them a Python block, which would then have been run
# under Rscript and failed as a syntax error in the wrong language.
check("a language named in a caption is not the block's language",
      cce.language_of('{#lst-x lst-cap="Comparing R and Python"}') is None)
check("...and does not displace a class that follows it",
      cce.language_of('{#lst-y lst-cap="How R does it" .python}') == "python")
check("...nor one that precedes it",
      cce.language_of('{#lst-z .julia lst-cap="Why R is slower here"}') == "julia")
check("a quoted value may hold the delimiter of the other quote",
      cce.language_of("""{#lst-w .python lst-cap="R's own version"}""") == "python")
check("the real anchored block in model-selection.qmd resolves to python",
      cce.language_of(
          '{#lst-winners-curse .python '
          'lst-cap="The replication behind the table above"}') == "python")

print("missing packages")

# The pattern must key on the interpreter's own fatal error. A warning the
# snippet emitted lands on the same stream and used to be read as one.
forged = (
    "import sys\n"
    "sys.stderr.write(\"UserWarning: ... No module named 'foo'\\n\")\n"
    "raise ValueError('a genuine bug')\n"
)
if HAVE["python"]:
    code, out = run_checker({"a.qmd": block("python", forged)}, "--allow-skips")
    check("a snippet's own stderr warning does not forge a skip", code == 1)
    check("...and it is reported as a failure", "1 failed" in out)

for lang, source in (
    ("python", "import definitely_not_a_package_xyz"),
    ("r", "library(definitelyNotAPackageXyz)"),
    ("julia", "using DefinitelyNotAPackageXyz"),
):
    if not HAVE[lang]:
        continue
    code, out = run_checker({"a.qmd": block(lang, source)}, "--allow-skips")
    check(f"a genuinely missing {lang} package is still a skip", code == 0, out)

# Julia says LoadError when the failure comes from a file rather than -e,
# which is how the checker always invokes it.
check(
    "Julia's LoadError wrapper is recognised",
    cce.is_missing_package(
        "ERROR: LoadError: ArgumentError: Package Foo not found in current path.\n"
    ),
)
check(
    "R's wrapped continuation line is recognised",
    cce.is_missing_package(
        "Error in library(foo) : \n  there is no package called 'foo'\n"
    ),
)
check(
    "a bare mention with no error line is not",
    not cce.is_missing_package("note: there is no package called 'foo'\n"),
)


# --- The Julia block-comment rule is Julia's alone -------------------------
#
# `#=` opens nothing in Python or R, where `#=== setup ===` is an ordinary
# end-of-line comment. Blanking those languages too swallowed every line after
# such a banner, so the block was reported as having no executable line AND
# dropped from the run set -- the failure this guard exists to catch, arriving
# by its opposite. Both directions are pinned: the banner must stay executable,
# and Julia's real block comment must still be caught.
# Written per language rather than shared: `is_effectively_empty` is a pure
# text scan and never parses, so one Python snippet would exercise both arms --
# but it would show a reader checking the R case a block of Python.
_BANNER = {
    "python": "#=== setup ===\nimport numpy as np\nprint(np.arange(3))\n",
    "r": "#=== setup ===\nx <- seq_len(3)\nprint(x)\n",
}
for _lang, _src in _BANNER.items():
    check(f"a `#===` banner comment leaves a {_lang} block executable",
          not cce.is_effectively_empty(_src, _lang))
check("...while the same text in Julia is a block comment and IS empty",
      cce.is_effectively_empty("#=== setup ===\nimport numpy as np\n", "julia"))

_AGREE = "blocks run standalone, so the copies must agree"

# --- Two definitions inside ONE block are not divergent copies -------------
#
# A block runs standalone, so rebinding a name in it is ordinary Python. The
# old message also cited the finding's own line as the other copy, telling an
# editor to compare line 10 against line 10.
check("two defs in one block are not reported",
      cce.redefinitions([
          (10, "python", "def f():\n    return 1\ndef f():\n    return 2\n"),
      ]) == [])

# That case alone passes whether the second definition is SKIPPED or KEPT, so
# it pins the symptom and not the mechanism. These three pin the mechanism.
# Skipping registered the block's dead FIRST copy as canonical, so a later
# block matching that dead text passed while the live definitions differed.
# Asserted as the whole tuple rather than as a count. A length of one is
# satisfied by a finding that names the wrong lines, which is the failure this
# message exists to prevent: both numbers must be the DEFINITION's line rather
# than its fence's, so 21 (the second block's `def`, not its fence at 20) and
# 13 (the first block's LIVE second `def`, not its fence at 10).
check("a later block matching the DEAD first copy is still reported",
      cce.redefinitions([
          (10, "python", "def f():\n    return 1\ndef f():\n    return 2\n"),
          (20, "python", "def f():\n    return 1\n"),
      ]) == [(21, "python `f` differs from the copy at line 13; " + _AGREE)])
check("...and one matching the LIVE last copy is not",
      cce.redefinitions([
          (10, "python", "def f():\n    return 1\ndef f():\n    return 2\n"),
          (20, "python", "def f():\n    return 2\n"),
      ]) == [])
# Comparing definition by definition re-compared each copy against a `seen`
# entry it never updated, so one divergence was reported once per copy.
check("a block defining a name twice reports one divergence, not two",
      cce.redefinitions([
          (10, "python", "def g():\n    return 0\n"),
          (20, "python", "def g():\n    return 1\ndef g():\n    return 2\n"),
      ]) == [(23, "python `g` differs from the copy at line 11; " + _AGREE)])

# --- A lambda bound to a name is a function definition ---------------------
#
# `model-selection.qmd` writes its `truth` this way and duplicates it across
# two blocks, so the commonest divergence route in this corpus was invisible.
check("a lambda that drifts between blocks is reported",
      len(cce.redefinitions([
          (10, "python", "truth = lambda x: 1 - 2 * x\n"),
          (20, "python", "truth = lambda x: 1 + 2 * x\n"),
      ])) == 1)
check("...and an identical one is not",
      cce.redefinitions([
          (10, "python", "truth = lambda x: 1 - 2 * x\n"),
          (20, "python", "truth = lambda x: 1 - 2 * x\n"),
      ]) == [])

# A decorator changes what the block computes, and `node.lineno` points at the
# `def`, so comparing from there read an undecorated copy as byte-identical.
check("a decorator difference is reported",
      len(cce.redefinitions([
          (10, "python", "import functools\n@functools.cache\ndef f(x):\n    return x\n"),
          (20, "python", "import functools\ndef f(x):\n    return x\n"),
      ])) == 1)

# `AsyncFunctionDef` is the other half of the isinstance tuple. Without this
# the suite stays green when it is dropped.
check("an async def that drifts between blocks is reported",
      len(cce.redefinitions([
          (10, "python", "async def fetch():\n    return 1\n"),
          (20, "python", "async def fetch():\n    return 2\n"),
      ])) == 1)

# The constant route is a documented NON-guarantee, not an oversight: blocks
# legitimately set different sample sizes and seeds. Pinned so that widening
# the guard to module-level names becomes a deliberate decision.
check("a module-level constant is deliberately NOT compared",
      cce.redefinitions([
          (10, "python", "LAM = 0.01\n"),
          (20, "python", "LAM = 0.10\n"),
      ]) == [])


if FAILURES:
    print("%d check(s) failed: %s" % (len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("all checks passed")
