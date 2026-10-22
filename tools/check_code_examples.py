#!/usr/bin/env python3
"""Run every fenced code example in the chapters and report the ones that fail.

Code whose output a page shows is an executable ``{r}``, ``{python}`` or
``{julia}`` chunk, which the render runs. Some examples are instead plain
``python`` / ``r`` / ``julia`` fences, shown but not executed -- a parallel
implementation printed beside the one that ran, for instance. A green
``quarto render`` says nothing about whether those work: a snippet can be
syntactically broken, call a function that does not exist, or silently compute
the wrong thing, and the site will build and publish it regardless.

This script closes that gap. Executable chunks are left to the render, since
they share a session with the chunks before them and cannot run alone (see
:func:`is_executable_chunk`). It extracts each other block, writes it to a
file, runs
it under the matching interpreter, and prints a verdict per block plus the
full output of each one that failed.

A checker that reports success over code it never ran is worse than no checker,
because it is indistinguishable from one that ran everything. Every guard below
exists only to make that outcome unreachable:

* finding no snippets is a failure, not a clean run, whenever the caller
  narrowed the search (``--chapters`` or ``--lang``) or no ``.qmd`` file was
  found at all, since then the silence may be the caller's mistake. The one
  exception is the default run over the repository's own chapters: every
  code example there has been an executed chunk since 2026-09-28, so finding
  no plain fence in a populated ``chapters/`` is the expected result, and the
  run says "nothing to check" and exits 0;
* a skipped snippet makes the run non-zero unless ``--allow-skips`` says the
  environment is known to be partial, so a misspelt package name cannot hide
  behind the missing-package rule;
* an unterminated fence is reported, whatever language it claimed, because the
  recovery that follows one cannot be trusted to find what came after it;
* a recognised-language fence nested inside an unrecognised one is reported
  when the *inner block's closer* would end the outer block early -- same
  fence character, run at least as long. Note that the inner opener itself
  never closes anything: CommonMark requires a closing fence carry nothing
  but spaces after it, and an opener by definition carries an info string.
  What steals the outer block is the bare run that closes the inner one,
  which matches the inner opener's length. So a shorter inner run, or a
  different fence character, ends nothing -- which is how a code fence is
  quoted inside a code block, and is well-formed;
* a block tagged with a known alias (``py``, ``jl``, ``Rscript``) or with
  Pandoc's attribute spelling (``{.python}``) is *run* rather than passed
  over, since an unrecognised tag would otherwise make a block invisible with
  nothing said;
* a block with no executable line is reported rather than tallied as a pass,
  because every interpreter accepts an empty file -- including a Julia block
  whose whole body sits inside a ``#= ... =#`` comment, whose interior lines
  begin with neither ``#`` nor whitespace and so read as executable;
* chapters are found recursively, so a file in a subdirectory cannot be missed;
* each snippet runs in its own directory, so it cannot read what an earlier one
  left behind;
* the missing-package patterns are anchored to each interpreter's own fatal
  error, so a warning the snippet itself emitted cannot be mistaken for one;
* the closing summary line carries the count of document problems, so the
  tail of a failed run does not read as a clean one.

That last guard is the weakest, and knowing why matters. A snippet can still
forge a skip by writing an interpreter's exact fatal-error line to stderr
deliberately. What it can no longer do is forge one *by accident*, which was
the realistic case: a ``warnings.warn`` mentioning a missing optional backend
used to classify a genuinely broken example as a skip.

Usage:

    python3 tools/check_code_examples.py                 # every chapter
    python3 tools/check_code_examples.py --lang r        # one language
    python3 tools/check_code_examples.py --show          # print passing output
    python3 tools/check_code_examples.py --allow-skips   # partial toolchain
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CHAPTERS = Path(__file__).resolve().parent.parent / "chapters"

# Each language maps to the file extension its interpreter expects and the
# command that runs a file. `Rscript` rather than `R`, because `R` alone waits
# for input on stdin.
LANGUAGES = {
    "python": {"ext": ".py", "cmd": ["python3"]},
    "r": {"ext": ".R", "cmd": ["Rscript"]},
    "julia": {"ext": ".jl", "cmd": ["julia"]},
}

# Spellings that mean one of the three above. Without these a block tagged
# `py` is not recognised, and an unrecognised tag makes a block invisible
# rather than noisy -- the same hazard the missing-package rule exists to
# close, one level up. `jl` is the extension this script itself writes.
ALIASES = {"py": "python", "python3": "python", "jl": "julia", "rscript": "r"}

# CommonMark: a fence opens on three or more backticks or tildes, indented by
# at most three spaces, and closes on a run of the same character at least as
# long. Matching the run length matters -- a block containing a ``` line inside
# a string would otherwise be truncated there and run as broken source.
FENCE_OPEN = re.compile(r"^(?P<indent> {0,3})(?P<run>`{3,}|~{3,})(?P<info>.*)$")

# Julia's first run of a session compiles as it goes, so it needs a longer
# budget than the other two. The value is a ceiling, not an expectation.
TIMEOUT_SECONDS = 600


def language_of(info: str) -> str | None:
    """Return the language a fence's info string names, or None.

    Quarto allows attributes after the language (``python filename="a.py"``)
    and brackets an executable block (``{python}``); both name the same
    language as the bare word does. Pandoc's attribute spelling ``{.python}``
    is a third form for the same language, and the one a contributor here is
    most likely to reach for, since it marks a block as highlighted rather
    than executed -- which is what every checked block here is. Left
    unresolved it made such a block invisible to the checker, with nothing
    said. Known aliases resolve to their canonical language rather than to
    None.

    The language is not always the FIRST token. A cross-referenceable block
    is a Pandoc listing, whose id leads: ``{#lst-name .python}``. Reading
    only the first token returns None there, so giving a block an anchor --
    the fix for a forward reference in the prose -- would silently drop it
    from the run.

    Scanning every token instead is not enough either, because an attribute
    VALUE is prose. Quarto will not resolve a listing crossref without an
    ``lst-cap`` (measured against Quarto 1.10.18: the reference renders as
    ``?@lst-name`` and the render warns), so an anchored block carries a
    caption, and a caption may say ``"Comparing R and Python"``. Split on
    whitespace, that caption contributes a bare ``R``. Two real spellings
    went wrong before this was a scanner rather than a token loop:
    ``{#lst-x lst-cap="Comparing R and Python"}``, which names no language
    at all, resolved to r; and ``{#lst-y lst-cap="How R does it" .python}``
    resolved to r as well, so a Python block would have been run under
    Rscript. The value is therefore consumed with its key and never tested.

    An escaped quote inside a value is not handled; Pandoc accepts one, and
    no caption in this repository needs it.
    """
    stripped = info.strip()
    if not stripped:
        return None
    if stripped.startswith("{") and stripped.endswith("}"):
        stripped = stripped[1:-1]

    i, n = 0, len(stripped)
    while i < n:
        if stripped[i] in " \t,":
            i += 1
            continue
        start = i
        while i < n and stripped[i] not in " \t,=":
            i += 1
        word = stripped[start:i]
        # Look past whitespace for an `=`: Pandoc accepts spaces around it, so
        # `filename = r` is a key and a value rather than two bare words.
        j = i
        while j < n and stripped[j] in " \t":
            j += 1
        if j < n and stripped[j] == "=":
            i = j + 1
            while i < n and stripped[i] in " \t":
                i += 1
            if i < n and stripped[i] in "\"'":
                quote = stripped[i]
                i += 1
                while i < n and stripped[i] != quote:
                    i += 1
                i += 1
            else:
                while i < n and stripped[i] not in " \t,":
                    i += 1
            continue
        bare = word.lstrip(".").lower()
        lowered = ALIASES.get(bare, bare)
        if lowered in LANGUAGES:
            return lowered
    return None


def is_executable_chunk(info: str) -> bool:
    """True when a fence is a Quarto chunk that the render itself executes.

    Quarto executes a block whose info string is braced and *begins* with a
    bare engine language: ``{r}``, ``{python}``, ``{julia}``,
    ``{r setup, include=FALSE}``. Those chunks share a session with the
    chunks before them and read data relative to their page, so running one
    alone in an empty directory fails for reasons that say nothing about the
    code. The render already runs them, and CI renders with R, Python and
    Julia installed, so a broken chunk fails the build. This checker covers
    what the render does not run: plain fences, and Pandoc's highlighted-only
    ``{.python}`` or ``{#lst-x .r}``, whose leading ``.`` or ``#`` is what
    keeps them from executing.
    """
    stripped = info.strip()
    if not stripped.startswith("{"):
        return False
    return re.split(r"[\s,}]", stripped[1:], maxsplit=1)[0].lower() in LANGUAGES


def strip_julia_block_comments(source: str) -> str:
    """Blank the interior of Julia's ``#= ... =#`` comments.

    The line test below keys on ``#``, which all three languages take to
    end-of-line. Julia alone also has a *block* comment, whose interior lines
    begin with neither ``#`` nor whitespace -- so a block commented out that
    way reads as executable, runs, exits 0, and is tallied as a pass over a
    no-op. That is the outcome :func:`is_effectively_empty` exists to prevent,
    one language down.

    Julia nests these, so the scan counts depth rather than stopping at the
    first ``=#``. Lines are blanked rather than deleted so a line number is
    never needed from this; only emptiness is.
    """
    out: list[str] = []
    depth = 0
    for line in source.splitlines():
        rest, kept = line, []
        while rest:
            opens = rest.find("#=")
            closes = rest.find("=#") if depth else -1
            # Inside a comment, an opener and a closer are both live, so take
            # whichever comes first. Scanning only for "=#" at depth closes a
            # nested comment one level too early and leaks its tail.
            if closes != -1 and (opens == -1 or closes < opens):
                depth -= 1
                rest = rest[closes + 2:]
                continue
            if not depth:
                # Outside a block comment a bare ``#`` runs to end of line, so
                # a ``#=`` sitting after one opens nothing. Without this the
                # scan read the ``#=`` in ``# Julia opens one with #=`` as an
                # opener and swallowed every line below it, so a good block
                # was reported as having no executable line *and* dropped from
                # the run -- the same failure the ``#===`` banner rule fixed
                # for Python and R, arriving one language over.
                hash_at = rest.find("#")
                if hash_at != -1 and hash_at != opens:
                    kept.append(rest)
                    break
            if opens == -1:
                if not depth:
                    kept.append(rest)
                break
            if not depth:
                kept.append(rest[:opens])
            depth += 1
            rest = rest[opens + 2:]
        out.append("".join(kept))
    return "\n".join(out)


def is_effectively_empty(source: str, lang: str) -> bool:
    """True when a block has no line an interpreter would act on.

    All three languages take ``#`` to end-of-line as a comment, so one line
    test serves for all three. Julia *also* has a block comment, whose
    interior lines begin with neither ``#`` nor whitespace, so those are
    blanked first -- but only for Julia. ``#=`` opens nothing in Python or R,
    where a banner comment such as ``#=== setup ===`` is an ordinary
    end-of-line comment; blanking those languages too swallowed every line
    after such a banner, so a perfectly good block was reported as having no
    executable line *and* dropped from the run set. That is the failure this
    function exists to catch, arriving by its opposite.

    Every interpreter accepts an empty file and exits 0, so without this a
    block that was accidentally emptied or commented out is indistinguishable
    in the tally from one that ran.
    """
    if lang == "julia":
        source = strip_julia_block_comments(source)
    return not any(
        line.strip() and not line.strip().startswith("#")
        for line in source.splitlines()
    )


def is_missing_package(stderr: str) -> bool:
    """True when the interpreter itself reported a package it could not find.

    Anchored to each interpreter's own fatal-error wording rather than matched
    loosely, so that a warning the snippet emitted -- which lands on this same
    stream -- is not read as a statement about the environment.
    """
    if re.search(r"^ModuleNotFoundError: No module named", stderr, re.M):
        return True
    # Julia wraps this in `LoadError` when the failure comes from a file
    # rather than from -e, which is how this script always invokes it.
    if re.search(
        r"^ERROR: [^\n]*ArgumentError: Package .+ not found in current path",
        stderr,
        re.M,
    ):
        return True
    # R wraps this message onto an indented continuation line, so the phrase
    # itself is not at a line start; require the error line as well.
    return bool(
        re.search(
            r"^Error\b[^\n]*(?:\n[^\n]*){0,2}there is no package called",
            stderr,
            re.M,
        )
    )


def extract(path: Path) -> tuple[list[tuple[int, str, str]], list[tuple[int, str]]]:
    """Return the fenced blocks in a file, and everything wrong with its fences.

    Blocks come back as (line number, language, source). Problems come back as
    (line number, description) and are failures rather than silent omissions:
    a block the extractor cannot read is exactly the case where reporting
    nothing looks identical to reporting success.
    """
    # read_text applies universal-newline translation, so a CRLF file arrives
    # here already split on "\n" with no carriage returns left in it. Nothing
    # further is needed, and adding it would be dead code.
    lines = path.read_text(encoding="utf-8").split("\n")
    blocks: list[tuple[int, str, str]] = []
    problems: list[tuple[int, str]] = []

    i = 0
    while i < len(lines):
        opener = FENCE_OPEN.match(lines[i])
        if not opener:
            i += 1
            continue
        run, indent = opener.group("run"), opener.group("indent")
        lang = language_of(opener.group("info"))
        closer = re.compile(r"^ {0,3}" + re.escape(run[0]) + "{%d,}\\s*$" % len(run))
        start, body = i + 1, []
        j = start
        while j < len(lines) and not closer.match(lines[j]):
            body.append(lines[j][len(indent):] if lines[j].startswith(indent) else lines[j])
            j += 1
        if j >= len(lines):
            # Reported whatever the language, because the scan resumes inside
            # the block and cannot be trusted to read what follows correctly.
            problems.append((i + 1, f"unterminated {lang or 'unrecognised'} fence"))
            i = start
            continue
        source = "\n".join(body) + "\n"
        if lang and is_executable_chunk(opener.group("info")):
            pass  # the render runs it; see is_executable_chunk
        elif lang:
            if is_effectively_empty(source, lang):
                problems.append((i + 1, f"{lang} fence has no executable line"))
            else:
                blocks.append((i + 1, lang, source))
        else:
            # A stray fence in prose swallows the block after it: its closer
            # is taken as this one's. The parse is right and the document is
            # wrong, and nothing else would report it.
            #
            # The inner *opener* is never the thief, and reading it as one
            # is the easy mistake here. CommonMark requires a closing fence
            # be followed only by spaces or tabs, and an opener carries an
            # info string by construction -- `language_of` returns None
            # without one -- so such a line cannot close anything. Measured
            # against pandoc 3.1.3: a ```text block containing a bare
            # ```python line swallows that line as content.
            #
            # What ends the outer block early is the bare run that closes the
            # inner block, and a well-formed inner block closes on a run of
            # its opener's length. So comparing the inner opener's length is
            # a proxy for its closer's, which is the length that decides it.
            # A shorter inner run, or a different fence character, closes
            # nothing -- exactly how one quotes a code fence inside a code
            # block, and well-formed. Reporting those said "the document is
            # malformed" about a correct one.
            for offset, line in enumerate(body):
                nested = FENCE_OPEN.match(line)
                if not nested or not language_of(nested.group("info")):
                    continue
                inner = nested.group("run")
                if inner[0] == run[0] and len(inner) >= len(run):
                    problems.append(
                        (
                            start + offset + 1,
                            "a recognised-language fence is nested inside an "
                            "unrecognised one; the document is malformed",
                        )
                    )
        i = j + 1

    return blocks, problems + redefinitions(blocks)


def _top_level_definition(node: ast.AST) -> tuple[str, int] | None:
    """The name a top-level node binds to a function, and the line it starts on.

    ``None`` for anything else. Two forms count, because both are function
    definitions and a chapter uses each: a ``def`` (or ``async def``), and a
    lambda assigned to a bare name, which is how `model-selection.qmd` writes
    its `truth`.

    The line is the *decorator's* where one is present. ``node.lineno`` points
    at the ``def`` and excludes ``decorator_list``, so comparing from there
    made an undecorated copy read byte-identical to a decorated one -- the
    decorator being exactly the kind of difference that changes what the block
    computes.
    """
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        start = node.lineno
        for dec in node.decorator_list:
            start = min(start, dec.lineno)
        return node.name, start
    if (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Lambda)
    ):
        return node.targets[0].id, node.lineno
    return None


def redefinitions(blocks: list[tuple[int, str, str]]) -> list[tuple[int, str]]:
    """Report a Python function one chapter defines twice, differently.

    Every block runs in a fresh interpreter, so a block that needs a helper
    must carry it. That makes some duplication load-bearing rather than
    careless: `model-selection.qmd` defines `design`, `ridge` and `cv_error`
    in both of its Python blocks because each has to run on its own, and
    merging them would break the second.

    What the duplication costs is a silent way to go wrong. Change `ridge` in
    one block and not the other and both still run, both still print, and the
    chapter now reports two sets of figures from two different estimators
    with nothing to say so. Prose can warn an editor about that; it cannot
    catch one who did not read the warning.

    So the copies are required to agree exactly, comments included. Exactly,
    rather than modulo comments, because a comment is where the divergence
    would be explained if it were deliberate -- and because the looser rule
    needs a tokenizer, which is a second thing to get wrong.

    **What "the copies" covers, stated as a rule rather than as a list.**
    Exactly two forms are compared, and both must sit directly in the module
    body: a ``def`` or ``async def``, including its decorators, and a
    single-target assignment of a lambda to a bare name
    (``truth = lambda x: ...``). Nothing else is compared, and a clean run
    rules nothing else out.

    The rule is put positively because a list of what is missed falls silently
    behind the code -- an earlier version of this docstring named three routes
    and called that the set, while at least three more were already live. Some
    consequences, each confirmed against this checker on 2026-09-23: a ``def``
    nested inside a top-level ``if``, ``try`` or ``with`` is invisible, since
    only ``tree.body`` is walked; so is an annotated binding
    (``truth: Callable = lambda x: ...``, an ``AnnAssign``) and a chained one
    (``a = truth = lambda x: ...``, which has two targets); so is a class and
    every method in it, the class body never being walked at all; and so is
    any binding built by a call rather than written out.
    A module-level constant (``LAM = 0.01`` against ``LAM = 0.10``) is left
    alone deliberately rather than overlooked -- blocks legitimately set
    different sample sizes and seeds, so requiring every module-level name to
    agree would report a difference the chapter intends. The rest are a real
    gap; see https://github.com/Morrison-Lab/lds/issues/33.

    Only Python is checked, because `ast` settles what a top-level definition
    is and a regular expression over R or Julia does not. No R or Julia
    helper is currently duplicated across blocks in any chapter (measured
    2026-09-23), so that gap too is unexercised rather than tolerated; it is
    a separate question from the one above, because it is about the languages
    this function never examines rather than the definitions it examines and
    does not see, and it is
    https://github.com/Morrison-Lab/lds/issues/32.
    """
    seen: dict[str, tuple[int, str]] = {}
    problems: list[tuple[int, str]] = []
    for line, lang, source in blocks:
        if lang != "python":
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            # A block that does not parse is the interpreter's to report, with
            # its own message and its own line number. Saying so here as well
            # would double-report it, and saying nothing loses nothing.
            continue
        lines = source.split("\n")

        # The definitions this block LEAVES IN PLACE, last binding wins. A
        # block that defines a name twice runs standalone, so rebinding it is
        # ordinary Python and says nothing about any other block -- but the
        # live binding is the last one, and that is the only one another
        # block's copy has to agree with. Collecting the block first, rather
        # than comparing definition by definition, is what keeps the dead
        # first copy out of `seen` and stops a block that defines a name twice
        # reporting the same divergence once per copy.
        #
        # `line` is the fence opener, so the block's own first source line is
        # `line + 1` and an `ast` lineno of L sits at `line + L`. Both the
        # position reported and the one remembered are the definition's, not
        # the fence's: two blocks in one chapter can define the same name
        # identically except for a byte, and a message naming the fence sends
        # the reader to the top of a block to hunt for which line moved.
        live: dict[str, tuple[int, str]] = {}
        for node in tree.body:
            named = _top_level_definition(node)
            if named is None:
                continue
            name, start = named
            live[name] = (line + start,
                          "\n".join(lines[start - 1:node.end_lineno]))

        for name, (where, text) in live.items():
            first, previous = seen.get(name, (None, None))
            if previous is None:
                seen[name] = (where, text)
            elif previous != text:
                problems.append(
                    (
                        where,
                        f"python `{name}` differs from the copy at line "
                        f"{first}; blocks run standalone, so the copies must "
                        f"agree",
                    )
                )
    return problems


def run(lang: str, source: str, workdir: Path) -> tuple[str, str]:
    """Run one snippet. Returns (verdict, output), verdict in pass/fail/skip."""
    spec = LANGUAGES[lang]
    if shutil.which(spec["cmd"][0]) is None:
        return "skip", f"{spec['cmd'][0]} is not installed"

    script = workdir / f"snippet{spec['ext']}"
    script.write_text(source, encoding="utf-8")
    try:
        proc = subprocess.run(
            [*spec["cmd"], str(script)],
            capture_output=True,
            text=True,
            cwd=workdir,
            timeout=TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "fail", f"timed out after {TIMEOUT_SECONDS}s"

    output = (proc.stdout + proc.stderr).strip()
    if proc.returncode == 0:
        return "pass", output
    # Only the interpreter's own fatal error can say the environment is at
    # fault. A snippet that warns about a missing module and then fails for an
    # unrelated reason is a failure.
    if is_missing_package(proc.stderr):
        return "skip", output
    return "fail", output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=sorted(LANGUAGES), help="only this language")
    parser.add_argument("--show", action="store_true", help="print output of passes too")
    parser.add_argument(
        "--allow-skips",
        action="store_true",
        help="exit 0 even when snippets were skipped for a missing interpreter "
        "or package; without it a skip is a failure, because a misspelt "
        "package name looks exactly like an absent one",
    )
    parser.add_argument(
        "--chapters",
        type=Path,
        default=None,
        help="directory to scan (default: the repository's chapters/); "
        "naming one makes finding no snippets there a failure",
    )
    args = parser.parse_args(argv)
    # Read at call time rather than bound as the argparse default, so the
    # self-test can point the default population at a throwaway directory.
    explicit_chapters = args.chapters is not None
    if not explicit_chapters:
        args.chapters = CHAPTERS

    # rglob, not glob: a chapter in a subdirectory would otherwise go
    # unexamined with nothing said about it.
    sources = sorted(args.chapters.rglob("*.qmd"))
    # One parse per file. Two would read the file twice, and the two passes
    # could disagree about its contents if it changed in between.
    parsed = {path: extract(path) for path in sources}

    tallies = {"pass": 0, "fail": 0, "skip": 0}
    failures: list[tuple[str, str]] = []
    defects: list[str] = []
    # Snippets `--lang` held back. Counted so the summary can name them: a run
    # over one language of three otherwise ends on a sentence identical to a
    # whole-corpus run, and a broken block in a filtered-out language is not
    # merely unreported but unreachable, since the interpreter that would have
    # reported it never runs.
    excluded = 0

    with tempfile.TemporaryDirectory() as tmp:
        for index, (path, line, lang, source) in enumerate(
            (p, ln, lg, src) for p in sources for ln, lg, src in parsed[p][0]
        ):
            if args.lang and lang != args.lang:
                excluded += 1
                continue
            # Its own directory, so a snippet cannot read files or sources
            # left behind by an earlier one.
            workdir = Path(tmp) / f"s{index:04d}"
            workdir.mkdir()
            verdict, output = run(lang, source, workdir)
            tallies[verdict] += 1
            where = f"{path.relative_to(args.chapters.parent)}:{line} ({lang})"
            if verdict == "fail":
                failures.append((where, output))
                print(f"FAIL {where}")
            elif verdict == "skip":
                print(f"skip {where}: {output.splitlines()[0]}")
            elif args.show:
                print(f"pass {where}\n{output}\n")

    for path in sources:
        for line, why in parsed[path][1]:
            where = f"{path.relative_to(args.chapters.parent)}:{line}"
            defects.append(f"{where}: {why}")
            print(f"FAIL {where}: {why}")

    total = sum(tallies.values())
    # The defect count belongs on this line, not only in the FAIL lines
    # above it: a reader who checks the tail of the log is the case this
    # module is written against, and a run that failed on a malformed
    # document -- or on two copies of a helper that have drifted apart --
    # otherwise ends on a sentence indistinguishable from a clean one.
    summary = (
        f"\nran {total} snippet(s): "
        f"{tallies['pass']} passed, {tallies['fail']} failed, {tallies['skip']} skipped"
    )
    if defects:
        summary += f", {len(defects)} document defect(s)"
    if excluded:
        summary += f" ({excluded} excluded by --lang {args.lang})"
    print(summary)

    for where, output in failures:
        print(f"\n--- {where}\n{output}")

    if total == 0:
        # The default run over a populated chapters/ with no plain fence in
        # it is the normal state of this repository, not a vacuous pass: the
        # files were all read, and none held anything to run. Anything the
        # caller narrowed -- a named directory, or a --lang that excluded
        # everything -- still fails, because there the silence is more likely
        # a wrong path or filter than an empty corpus. So does a directory
        # with no .qmd file at all, which reads the same as a wrong path.
        if not (explicit_chapters or args.lang or excluded or defects) and sources:
            print(
                f"\nno plain code fences found under "
                f"{args.chapters.name}/: nothing to check"
            )
            return 0
        print("\nno snippets found: the checker examined nothing")
        return 1
    if defects:
        return 1
    if tallies["skip"] and not args.allow_skips:
        print("\nsnippets were skipped; pass --allow-skips if that is expected")
        return 1
    return 1 if tallies["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
