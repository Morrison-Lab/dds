#!/usr/bin/env python3
"""Check the rendered site in ``_site/`` rather than the sources that produced it.

These classes of defect survive a green ``quarto render``. Most are invisible
in a source diff, being properties of what was written to ``_site/`` rather
than of any source file; the deck-naming check is the exception, reading an
``output-file`` a diff does show:

* a missing output -- a page, slide deck, or handout its source promised but
  that was never written, or that a later render removed;
* a page overwritten by another format's render, so the website serves a slide
  deck where the chapter should be (this happens when a page does not name a
  distinct ``output-file`` per format);
* a stylesheet or script a page references but that is not in ``_site`` --
  rendering one format at a time prunes the supporting files of the formats not
  being rendered, so the last render wins and every other format is left
  without its assets. Render every format in one ``quarto render``;
* a slide deck that is really one scrollable slide, because its chapter has no
  headings for reveal.js to split on;
* a generated chapter that was never added to the render list, so it is never
  built and never checked;
* a deck whose ``output-file`` does not end in ``-slides.html``, which silently
  costs it the "Webpage version" back-link -- the filter in
  ``_extensions/d-morrison/revealjs-html-links`` derives the page URL from that
  suffix, and returns without adding a link when it is absent;
* a deck carrying no back-link at all, because the filter was not registered
  -- detected from the absence of the filter's injected script, since the link
  itself is built in the browser and never appears in the rendered HTML;
* a deck slide whose section id does not exist on the page the back-link
  actually targets, so the link points at a fragment the browser cannot
  resolve -- or, where a deck is named after something other than its own
  source, at a page that does not exist at all. A source whose own page
  failed to render is reported by the missing-page check instead, so that
  second branch fires only on such a name mismatch;
* a broken image or a link to a page that was never rendered;
* an unlinked ``<code>data/<name>.csv</code>`` reference in prose (excluding
  local save-path instructions after ``save it as``), where a published dataset
  file under ``data/`` is named in inline code without a hyperlink;
* a Pandoc example list. The site uses none on purpose, so one only appears
  when a cross-reference written ``@label.``, ``@label)`` or ``(@label)`` starts a line
  that Pandoc can read as the start of a list -- a continuation line inside a
  list item, or a line after a blank one: Pandoc reads it as an example-list
  marker and swallows the reference. A source-side grep cannot stand in for
  this, since whether such a line starts a list depends on its context.

Usage::

    python3 tools/check_site.py [--site _site] [--config _quarto-website.yml]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"')
ASSET_RE = re.compile(r'<(?:link[^>]+href|script[^>]+src)="([^"]+)"')
HREF_RE = re.compile(r'<a[^>]+href="([^"]+)"')
RENDER_ENTRY_RE = re.compile(r"^\s{4}- (\S+\.qmd)\s*$")
OUTPUT_FILE_RE = re.compile(r"^\s*output-file:\s*(\S+)\s*$", re.MULTILINE)
# A reveal.js deck wraps its slides in `<div class="reveal">`; a website page
# has no such element. Note the marker is on that div and NOT on <body>, whose
# class list looks similar enough to write a matcher that never fires.
REVEAL_RE = re.compile(r'<div[^>]*\bclass="[^"]*\breveal\b[^"]*"')
# Each slide reveal.js splits at level 2. A deck with one of these is a whole
# chapter on a single scrollable slide, which renders and looks fine to every
# other check here.
SLIDE_RE = re.compile(r'<section[^>]*\bclass="[^"]*\bslide level2\b')
# Attribute order inside a tag is not fixed, so read the tag's attributes once
# and pick `id` and `class` out of them rather than assuming either comes first.
SECTION_TAG_RE = re.compile(r"<section\b([^>]*)>")
# Anchored on a tag boundary rather than `\b`, which also matches between the
# `-` and the `i` of any `data-*-id` attribute -- `data-anchor-id` is the one
# Quarto is known to emit. A `\b` form would then admit a value that is not an
# element id, and would read the wrong attribute where such a name precedes
# `id` inside a <section> tag. Like the <script> strip below, this is a guard
# rather than a fix for a live defect, and the measurement says why rather
# than saying the attribute is absent: a chapter's HTML page carries
# `data-anchor-id` on each of its content headings, and both forms still
# harvest the identical set of ids from it, because Quarto sets each
# `data-anchor-id` to the enclosing section's own id -- so the loose form
# reads more occurrences and no new value. Re-derive it on the page in front
# of you by harvesting under both forms and diffing the two sets. What the
# anchoring guards against is a `data-*-id` whose value is NOT an element id.
# A deck carries none of these at all, which is where `slide_section_ids`
# looks.
ID_ATTR_RE = re.compile(r'(?:^|\s)id="([^"]+)"')
# A <script> element's own `id` is markup and is kept; its BODY is not, and
# is stripped before ids are harvested. The construct that motivates the
# strip is an HTML fragment built inside a JS string --
# `el.innerHTML = '<div id="x">'` -- whose `id="` reads to a regex exactly
# like a real element id, and counting one as a page id would let a coincidental
# collision mask a genuinely dead back-link fragment. Neither a
# `getElementById("x")` call nor a JSON `"id":` key can match ID_ATTR_RE, so
# neither is a reason for this. In a local render of this site no body-only
# id occurs anywhere: the strip is a guard against a construct Quarto does
# not currently emit, not a fix for a live defect.
# The capture group restores the start tag, so the `<script id="quarto-*">`
# elements Quarto does emit keep their ids. Dropping the start tag instead
# deletes those ids -- `quarto-html-after-body`, `quarto-html-before-body`
# and `quarto-search-options` are the ones seen on <script> elements here.
# Which pages lose which id depends on what the render produced, so no count
# is quoted: re-derive it by harvesting ids from `_site` under both forms of
# this pattern and diffing the two sets.
SCRIPT_BLOCK_RE = re.compile(r"(<script\b[^>]*>).*?</script\s*>", re.DOTALL | re.IGNORECASE)
CLASS_ATTR_RE = re.compile(r'\bclass="([^"]*)"')
# Pandoc writes an example list as `<ol class="example" type="1">`, with
# `start=` before `class=` on a later list. `example` must be a whole class
# token, so a class such as `example-list` does not match. Case and quote
# style are tolerated, as SCRIPT_BLOCK_RE tolerates case, although Pandoc
# writes lowercase, double-quoted attributes.
EXAMPLE_LIST_RE = re.compile(
    r"""<ol\b[^>]*\sclass=(["'])(?:[^"']*\s)?example(?:\s[^"']*)?\1""", re.IGNORECASE
)
# Quarto names a deck's opening section `title-slide` and the HTML page's own
# title block `title-block-header`, so this one id is absent from every page by
# design. The back-link filter skips it for the same reason.
TITLE_SLIDE_ID = "title-slide"
# The filter builds its back-link in JavaScript, so the rendered page carries
# no `id="revealjs-html-anchor"` attribute to look for -- the anchor does not
# exist until the browser runs the script. What IS static is the script itself,
# so this greps the identifier out of that source text. It therefore reports
# whether the filter injected its script, which is exactly the failure a
# dropped `filters:` entry produces; it says nothing about what the browser
# then renders. Coupled to the identifier in
# `_extensions/d-morrison/revealjs-html-links/revealjs-html-links.lua`, which
# names this checker in a comment of its own.
FILTER_MARKER = "revealjs-html-anchor"

EXTERNAL_PREFIXES = ("http://", "https://", "mailto:", "javascript:", "data:", "#")


def render_targets(config: Path) -> list[str]:
    """Read the render list out of the website profile, without a YAML parser.

    The list is a flat sequence of four-space-indented entries under
    ``project.render``, so a line match is enough and keeps this script
    dependency-free.
    """
    targets = []
    in_render = False
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.strip() == "render:":
            in_render = True
            continue
        if in_render:
            match = RENDER_ENTRY_RE.match(line)
            if match:
                targets.append(match.group(1))
            elif line.strip() and not line.startswith(" " * 4):
                break
    return targets


def declared_outputs(source: Path) -> list[str]:
    """Return the ``output-file`` names a page's front matter promises."""
    text = source.read_text(encoding="utf-8")
    _, _, rest = text.partition("---\n")
    front_matter, _, _ = rest.partition("\n---")
    return OUTPUT_FILE_RE.findall(front_matter)


def count_example_lists(page_text: str) -> int:
    """Count the Pandoc example lists in a rendered page.

    Script bodies are stripped first, as for ids, so markup built inside a
    JavaScript string is not counted.
    """
    return len(EXAMPLE_LIST_RE.findall(SCRIPT_BLOCK_RE.sub(r"\1", page_text)))


PRE_BLOCK_RE = re.compile(r"<pre\b[^>]*>.*?</pre\s*>", re.DOTALL | re.IGNORECASE)
ANCHOR_BLOCK_RE = re.compile(r"<a\b[^>]*>.*?</a\s*>", re.DOTALL | re.IGNORECASE)
SAVE_AS_CSV_RE = re.compile(
    r"\bsave\s+it\s+as\s+<code\b[^>]*>data/[A-Za-z0-9_.-]+\.csv</code\s*>",
    re.IGNORECASE,
)
UNLINKED_DATA_CSV_RE = re.compile(
    r"<code\b[^>]*>(data/[A-Za-z0-9_.-]+\.csv)</code\s*>",
    re.IGNORECASE,
)


def unlinked_data_csvs(page_text: str) -> list[str]:
    """Return unlinked inline ``<code>data/<name>.csv</code>`` references in a page.

    Strips ``<script>``, ``<pre>`` and ``<a>`` blocks first, and skips local
    save-path instructions immediately following ``save it as``.
    """
    stripped = SCRIPT_BLOCK_RE.sub(r"\1", page_text)
    stripped = PRE_BLOCK_RE.sub("", stripped)
    stripped = ANCHOR_BLOCK_RE.sub("", stripped)
    stripped = SAVE_AS_CSV_RE.sub("", stripped)
    return UNLINKED_DATA_CSV_RE.findall(stripped)


def slide_section_ids(deck_text: str) -> set[str]:
    """Return the ids of a deck's <section> elements carrying a `slide` class.

    That is a narrower set than the filter's own reachable one: the filter's
    sibling walk tests only that a <section> has an id other than the title
    slide's, with no class test. The two coincide for every deck Quarto
    currently produces, whose only id-bearing classless <section> is the title
    slide, which both exclude.
    """
    ids = set()
    for attrs in SECTION_TAG_RE.findall(deck_text):
        classes = CLASS_ATTR_RE.search(attrs)
        if not classes or "slide" not in classes.group(1).split():
            continue
        found = ID_ATTR_RE.search(attrs)
        if found and found.group(1) != TITLE_SLIDE_ID:
            ids.add(found.group(1))
    return ids


def check_deck_output(
    produced: Path,
    produced_text: str,
    name: str,
    out_dir: Path,
) -> list[str]:
    """Report every defect a declared `revealjs` output can carry.

    Six causes, each silent on its own. Four are back-link failures: a deck
    whose name the filter cannot parse, a filter that never ran, a link target
    that was never rendered, and a section id the target page does not carry.
    The other two are not about the back-link at all, and are here because they
    share this function's inputs rather than its subject -- an output named
    `-slides.html` that is not a deck, and a deck the whole chapter fits on one
    slide of. The module docstring lists that last one as its own class of
    defect, which is why this function is named for the output rather than for
    the link.

    `out_dir` is the rendered directory the deck and its page share, so the
    link target is derived the same way the filter derives it.
    """
    problems: list[str] = []
    is_deck = bool(REVEAL_RE.search(produced_text))
    if is_deck and not name.endswith("-slides.html"):
        problems.append(
            f"{produced}: a deck whose output-file does not end in `-slides.html`, "
            "so the revealjs-html-links filter cannot derive its page URL and the "
            "deck silently gets no `Webpage version` link"
        )
    if name.endswith("-slides.html") and not is_deck:
        problems.append(f"{produced}: expected a reveal.js deck, got a plain page")
    elif name.endswith("-slides.html"):
        if len(SLIDE_RE.findall(produced_text)) < 2:
            problems.append(
                f"{produced}: the whole chapter is on one slide -- its source has no "
                "headings to split at, so it should not declare a revealjs output"
            )
        # The name check above tests one CAUSE of a missing back-link.
        # This tests a different one: with the filter unregistered, no
        # deck gets a link at all and every other check here stays
        # green. Measured against a real render: a deck built with the
        # filter registered carries this marker twice, and the same
        # deck built without it carries it zero times.
        if FILTER_MARKER not in produced_text:
            problems.append(
                f"{produced}: a deck with no `Webpage version` back-link -- the "
                "revealjs-html-links filter did not run, so check that "
                "`_quarto-website.yml` still lists it under `format: revealjs`"
            )
        # The filter derives its target from the DECK's own filename,
        # not from the source's, and the two coincide only while every
        # chapter names its deck `<source stem>-slides.html`. Derive it
        # the same way, so a deck whose stem differs is reported rather
        # than checked against a page it never links to.
        link_name = name[: -len("-slides.html")] + ".html"
        link_target = out_dir / link_name
        if not link_target.exists():
            problems.append(
                f"{produced}: its `Webpage version` link points at {link_name}, "
                f"which was never rendered ({link_target} is missing)"
            )
            return problems
        # Every id the back-link can name has to exist on the page it
        # links to. A dead fragment is silently ignored by the browser,
        # so nothing but this check reports one.
        page_ids = set(ID_ATTR_RE.findall(SCRIPT_BLOCK_RE.sub(
            r"\1", link_target.read_text(encoding="utf-8", errors="replace")
        )))
        dead = slide_section_ids(produced_text) - page_ids
        if dead:
            shown_ids = sorted(dead)[:5]
            shown = ", ".join(shown_ids)
            if len(dead) > len(shown_ids):
                shown += ", ..."
            problems.append(
                f"{produced}: {len(dead)} slide section id(s) do not exist on "
                f"{link_name}, so the `Webpage version` link resolves to nothing "
                f"there ({shown})"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", default=Path("_site"), type=Path)
    parser.add_argument("--config", default=Path("_quarto-website.yml"), type=Path)
    args = parser.parse_args()

    if not args.site.is_dir():
        print(f"error: {args.site} does not exist; run `quarto render` first", file=sys.stderr)
        return 1

    targets = render_targets(args.config)
    if not targets:
        print(f"error: no render targets found in {args.config}", file=sys.stderr)
        return 1

    # A preview built without the PDF handouts (tools/skip-pdf-in-previews.py)
    # ships this marker, so its missing handouts are intended.
    no_pdf = (args.site / "preview-no-pdf.txt").exists()

    problems: list[str] = []
    pages: list[Path] = []
    outputs = 0

    for target in targets:
        source = Path(target)
        page = args.site / source.with_suffix(".html")
        outputs += 1
        if not page.exists():
            problems.append(f"{target}: no rendered page at {page}")
            continue
        pages.append(page)
        if REVEAL_RE.search(page.read_text(encoding="utf-8", errors="replace")):
            problems.append(
                f"{page}: website page was overwritten by a reveal.js deck "
                "-- give the page a distinct `output-file` for each format"
            )

        for name in declared_outputs(source):
            if no_pdf and name.endswith(".pdf"):
                continue
            outputs += 1
            produced = args.site / source.parent / name
            if not produced.exists():
                problems.append(f"{target}: declares output-file {name}, but {produced} is missing")
                continue
            if not produced.suffix == ".html":
                continue
            pages.append(produced)
            problems += check_deck_output(
                produced,
                produced.read_text(encoding="utf-8", errors="replace"),
                name,
                args.site / source.parent,
            )

    # A chapter the generator wrote but nobody listed is never rendered, so every
    # check above skips it in silence and the run still reports success.
    chapters_dir = args.config.parent / "chapters"
    if chapters_dir.is_dir():
        listed = {Path(t).name for t in targets}
        for qmd in sorted(chapters_dir.glob("*.qmd")):
            if qmd.name not in listed:
                problems.append(
                    f"{qmd}: exists but is not in {args.config}'s render list, so it is "
                    "never built and never checked"
                )

    images = links = assets = 0
    missing_assets: set[str] = set()
    for page in pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        example_lists = count_example_lists(text)
        if example_lists:
            problems.append(
                f"{page}: {example_lists} Pandoc example list(s) -- a cross-reference "
                "written `@label.`, `@label)` or `(@label)` at the start of a line was read as an "
                "example-list marker; reflow or reword it so the reference no longer "
                "starts a line"
            )
        unlinked_csvs = unlinked_data_csvs(text)
        if unlinked_csvs:
            problems.append(
                f"{page}: unlinked data CSV reference(s) in prose "
                f"({', '.join(sorted(set(unlinked_csvs)))}) -- wrap each in a "
                "markdown link to the file"
            )
        for src in ASSET_RE.findall(text):
            if src.startswith(EXTERNAL_PREFIXES):
                continue
            assets += 1
            if not (page.parent / src).resolve().exists():
                # Reported once per asset rather than once per page: a pruned
                # support directory is referenced by every page, and forty
                # copies of one fact is noise.
                missing_assets.add(src)
        for src in IMG_RE.findall(text):
            if src.startswith(EXTERNAL_PREFIXES):
                continue
            images += 1
            if not (page.parent / src).resolve().exists():
                problems.append(f"{page}: missing image {src}")
        for href in HREF_RE.findall(text):
            if href.startswith(EXTERNAL_PREFIXES):
                continue
            target_path = href.split("#", 1)[0]
            if not target_path:
                continue
            links += 1
            if target_path.endswith(".qmd"):
                problems.append(f"{page}: unresolved source link {href}")
            elif not (page.parent / target_path).resolve().exists():
                problems.append(f"{page}: broken link {href}")

    for src in sorted(missing_assets):
        problems.append(
            f"missing supporting file {src} -- render every format in one "
            "`quarto render`, so that no format's render prunes another's assets"
        )

    print(
        f"checked {len(pages)} rendered page(s) and {outputs} declared output(s) "
        f"from {len(targets)} source(s); {assets} stylesheet/script reference(s); "
        f"{images} image reference(s); {links} internal link(s)"
    )
    if problems:
        print(f"\n{len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print("no problems found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
