#!/usr/bin/env python3
"""Leave the PDF handouts out of a preview render, or render them in their own pass.

The LaTeX handouts take about half of the preview build's render time, and
only the push to `main` has to produce them.
Quarto cannot drop a declared format from a profile, and every chapter declares
`pdf` in its own front matter as well as the project file, so this `pre-render`
script removes the `pdf:` blocks from the working copy before Quarto reads them.

On a full site render in CI (`publish.yml` on `main`, or a PR carrying the
`preview:pdf` label), rendering all three formats (`html`, `revealjs` and `pdf`,
101 document renders across 34 chapters) inside a single Deno process exhausts
the 16 GB runner's memory and kills the runner (`exit code 143`).
Splitting `html` and `revealjs` via `--to` prunes the other format's
`_site/site_libs/` assets, whereas the PDF handouts are self-contained files
under `_site/chapters/`.
When PDF is not skipped in a CI full render, this script therefore runs
`quarto render --to pdf` in a separate child process first (which exits and
frees its Deno heap), and then removes the `pdf:` blocks so the outer
`quarto render` builds `html` and `revealjs` together without `pdf`.

Set `LDS_SKIP_PDF=0` to force the handouts on, or `LDS_SKIP_PDF=1` to force
them off (for testing).
Set `LDS_SPLIT_PDF=1` to force the two-pass split locally, or `LDS_SPLIT_PDF=0`
to disable it.
Rewriting `pdf:` blocks modifies the working copy, so run either override only
in a throwaway checkout.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABEL = "preview:pdf"
# Written beside the sources when the handouts are left out.
# The project's `resources:` list copies it into the rendered site, which is how
# tools/check_site.py learns that a missing handout is intended.
MARKER = "preview-no-pdf.txt"
# A `pdf:` key at the two-space indent of a `format:` map, and the lines that
# belong to it: blank lines, and anything indented deeper (comments included).
PDF_BLOCK = re.compile(r"^  pdf:[^\n]*\n(?:(?:[ \t]*\n)|(?:    [^\n]*\n))*", re.M)
# The `pre-render:` list in `_quarto-website.yml`, stripped during the inner
# `--to pdf` pass so the pre-render scripts do not run a second time.
PRE_RENDER_BLOCK = re.compile(
    r"^  pre-render:[^\n]*\n(?:(?:[ \t]*\n)|(?:    [^\n]*\n))*", re.M
)


def skip_requested() -> bool:
    forced = os.environ.get("LDS_SKIP_PDF")
    if forced in ("0", "1"):
        return forced == "1"
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return False
    path = os.environ.get("GITHUB_EVENT_PATH")
    if not path or not Path(path).is_file():
        return False
    event = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = {x["name"] for x in event.get("pull_request", {}).get("labels", [])}
    return LABEL not in labels


def split_requested() -> bool:
    if os.environ.get("LDS_IN_PDF_PASS") == "1":
        return False
    forced = os.environ.get("LDS_SPLIT_PDF")
    if forced in ("0", "1"):
        return forced == "1"
    return (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("QUARTO_PROJECT_RENDER_ALL") == "1"
    )


def strip_pdf(text: str, front_matter_only: bool) -> str:
    if not front_matter_only:
        return PDF_BLOCK.sub("", text)
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end == -1:
        return text
    head, rest = text[: end + 1], text[end + 1 :]
    return PDF_BLOCK.sub("", head) + rest


def render_pdf_pass(project: Path) -> None:
    """Render all PDF handouts in a child `quarto render --to pdf` process.

    Quarto cleans the project output directory (`_site/`) before running
    `project.pre-render` scripts (`src/command/render/project.ts`), so PDFs
    written to `_site/chapters/*-handout.pdf` during `pre-render` survive when
    the outer `quarto render` proceeds to build `html` and `revealjs`
    (verified end-to-end with Quarto 1.10.18 and `tools/check_site.py`).
    """
    orig = project.read_text(encoding="utf-8")
    no_pre = PRE_RENDER_BLOCK.sub("", orig, count=1)
    if no_pre == orig:
        raise RuntimeError(f"found no pre-render block to strip in {project}")
    st = project.stat()
    project.write_text(no_pre, encoding="utf-8")
    os.utime(project, (st.st_atime, st.st_mtime))
    try:
        bin_dir = os.environ.get("QUARTO_BIN_PATH")
        quarto = (
            str(Path(bin_dir) / "quarto")
            if bin_dir and (Path(bin_dir) / "quarto").is_file()
            else "quarto"
        )
        cmd = [quarto, "render"]
        profile = os.environ.get("QUARTO_PROFILE")
        if profile:
            cmd.extend(["--profile", profile])
        out_dir = os.environ.get("QUARTO_PROJECT_OUTPUT_DIR")
        if out_dir:
            cmd.extend(["--output-dir", out_dir])
        cmd.extend(["--to", "pdf"])
        env = os.environ.copy()
        env["LDS_IN_PDF_PASS"] = "1"
        print(
            "skip-pdf-in-previews: rendering PDF handouts in a separate pass",
            file=sys.stderr,
        )
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    finally:
        project.write_text(orig, encoding="utf-8")
        os.utime(project, (st.st_atime, st.st_mtime))


def update_freeze_hash(chapter: Path, old: str, new: str) -> None:
    """Keep a restored HTML freeze entry valid when `pdf:` is stripped from `chapter`.

    Quarto's `freeze: auto` computes `"hash"` in
    `_freeze/chapters/<stem>/execute-results/<fmt>.json` as the MD5 digest of
    the raw `.qmd` source file (`md5HashSync(Deno.readTextFileSync(file))` in
    Quarto's `src/command/render/freeze.ts`) before include expansion.
    """
    freeze_html = (
        ROOT / "_freeze" / "chapters" / chapter.stem / "execute-results" / "html.json"
    )
    if not freeze_html.is_file():
        return
    old_hash = hashlib.md5(old.encode("utf-8")).hexdigest()
    new_hash = hashlib.md5(new.encode("utf-8")).hexdigest()
    text = freeze_html.read_text(encoding="utf-8")
    old_token = f'"hash": "{old_hash}"'
    if old_token in text:
        freeze_html.write_text(
            text.replace(old_token, f'"hash": "{new_hash}"', 1),
            encoding="utf-8",
        )


def main() -> int:
    skip = skip_requested()
    split = not skip and split_requested()
    if not skip and not split:
        return 0
    project = ROOT / "_quarto-website.yml"
    if split:
        render_pdf_pass(project)
    changed = []
    shared = ROOT / "_quarto.yml"
    targets = (
        ([(shared, False)] if shared.is_file() else [])
        + [(project, False)]
        + [(p, True) for p in sorted((ROOT / "chapters").glob("*.qmd"))]
    )
    for path, front in targets:
        old = path.read_text(encoding="utf-8")
        new = strip_pdf(old, front)
        if new != old:
            st = path.stat()
            path.write_text(new, encoding="utf-8")
            # Preserve the timestamp set by tools/set-mtimes-from-git.py so
            # Quarto's `date: last-modified` shows the commit date, not today.
            os.utime(path, (st.st_atime, st.st_mtime))
            if front:
                update_freeze_hash(path, old, new)
            changed.append(path.name)
    if not changed:
        # Nothing to strip: a handout that is then missing is a real failure,
        # so do not mark the preview as intentionally PDF-free.
        print(
            "skip-pdf-in-previews: warning: found no pdf format to remove",
            file=sys.stderr,
        )
        return 0
    # Drop any `.quarto/idx/` files written before `pdf:` was stripped (such as
    # during `render_pdf_pass`), so Quarto's `readInputTargetIndex()` does not
    # load stale `projectFormats` entries into `inputTargetIndexCache`.
    shutil.rmtree(ROOT / ".quarto" / "idx", ignore_errors=True)
    if skip:
        (ROOT / MARKER).write_text(
            "Rendered without the PDF handouts (tools/skip-pdf-in-previews.py).\n",
            encoding="utf-8",
        )
    print(
        f"skip-pdf-in-previews: removed the pdf format from {len(changed)} files",
        file=sys.stderr,
    )
    # A chapter whose pdf entry has another shape (`format: [html, pdf]`, a
    # different indent) would still render a handout, silently.
    for path, front in targets:
        text = path.read_text(encoding="utf-8")
        if front:
            text = text.partition("\n---\n")[0] if text.startswith("---\n") else ""
        if re.search(r"^\s*(pdf:|format:.*\bpdf\b)", text, re.M):
            print(
                f"skip-pdf-in-previews: warning: {path.name} still mentions pdf",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
