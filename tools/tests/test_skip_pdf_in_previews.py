#!/usr/bin/env python3
"""Self-test for tools/skip-pdf-in-previews.py.

The script silently does nothing if its pattern stops matching, so the cases
pinned here are the two shapes it must strip (a chapter's front matter and the
project file), the neighbours it must leave alone, and the gate that decides
whether a run strips at all.

Run with ``python3 tools/tests/test_skip_pdf_in_previews.py``. Needs no pytest.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "skip-pdf-in-previews.py"
spec = importlib.util.spec_from_file_location("skip_pdf", SCRIPT)
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


CHAPTER = (
    "---\ntitle: T\nformat:\n  html:\n    css: webex.css\n"
    "  revealjs:\n    output-file: t-slides.html\n"
    "  pdf:\n    output-file: t-handout.pdf\n"
    "resources:\n  - data/a.csv\n---\n\nThe pdf: in prose stays.\n  pdf:\n    kept: yes\n"
)
got = sp.strip_pdf(CHAPTER, front_matter_only=True)
check("strips the pdf block from front matter", "t-handout.pdf" not in got, got)
check("keeps the html and revealjs blocks", "css: webex.css" in got and "t-slides.html" in got, got)
check("keeps the key after the block", "resources:\n  - data/a.csv" in got, got)
check("leaves the body alone", "kept: yes" in got and "The pdf: in prose stays." in got, got)

PROJECT = "format:\n  html:\n    toc: true\n  # a comment about pdf\n  pdf:\n    # inside\n    toc: false\n\n    documentclass: article\n"
got = sp.strip_pdf(PROJECT, front_matter_only=False)
check("strips the whole pdf block from the project file", "documentclass" not in got and "toc: false" not in got, got)
check("keeps html and a comment above the block", "toc: true" in got and "a comment about pdf" in got, got)

check("a file with no front matter is unchanged", sp.strip_pdf("no front matter\n  pdf:\n    x: 1\n", True) == "no front matter\n  pdf:\n    x: 1\n")


def skip_requested(env: dict[str, str], labels: list[str] | None = None) -> bool:
    """Run skip_requested() under a clean environment and a fake PR event."""
    keys = ["LDS_SKIP_PDF", "GITHUB_EVENT_NAME", "GITHUB_EVENT_PATH"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    with tempfile.TemporaryDirectory() as tmp:
        if labels is not None:
            event = Path(tmp) / "event.json"
            event.write_text(json.dumps({"pull_request": {"labels": [{"name": n} for n in labels]}}))
            env = {**env, "GITHUB_EVENT_PATH": str(event)}
        try:
            os.environ.update(env)
            return sp.skip_requested()
        finally:
            for k in keys:
                os.environ.pop(k, None)
                if saved[k] is not None:
                    os.environ[k] = saved[k]


PR = {"GITHUB_EVENT_NAME": "pull_request"}
check("a PR without the label skips", skip_requested(PR, []) is True)
check("a PR with another label skips", skip_requested(PR, ["clear freezer"]) is True)
check("a PR with preview:pdf renders the handouts", skip_requested(PR, ["preview:pdf"]) is False)
check("a push renders the handouts", skip_requested({"GITHUB_EVENT_NAME": "push"}, []) is False)
check("a local run renders the handouts", skip_requested({}) is False)
check("LDS_SKIP_PDF=1 forces a skip", skip_requested({"LDS_SKIP_PDF": "1"}) is True)
check("LDS_SKIP_PDF=0 beats the PR event", skip_requested({**PR, "LDS_SKIP_PDF": "0"}, []) is False)


def split_requested(env: dict[str, str]) -> bool:
    """Run split_requested() under a clean environment."""
    keys = ["LDS_IN_PDF_PASS", "LDS_SPLIT_PDF", "GITHUB_ACTIONS", "QUARTO_PROJECT_RENDER_ALL"]
    saved = {k: os.environ.pop(k, None) for k in keys}
    try:
        os.environ.update(env)
        return sp.split_requested()
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]


CI_FULL = {"GITHUB_ACTIONS": "true", "QUARTO_PROJECT_RENDER_ALL": "1"}
check("CI full project render splits PDF into its own pass", split_requested(CI_FULL) is True)
check("CI single-file render does not split", split_requested({"GITHUB_ACTIONS": "true"}) is False)
check("local full render does not split by default", split_requested({"QUARTO_PROJECT_RENDER_ALL": "1"}) is False)
check("LDS_SPLIT_PDF=1 forces a split locally", split_requested({"LDS_SPLIT_PDF": "1"}) is True)
check("LDS_SPLIT_PDF=0 disables the split in CI", split_requested({**CI_FULL, "LDS_SPLIT_PDF": "0"}) is False)
check(
    "LDS_IN_PDF_PASS=1 recursion guard blocks a nested split",
    split_requested({**CI_FULL, "LDS_SPLIT_PDF": "1", "LDS_IN_PDF_PASS": "1"}) is False,
)

WEBSITE_YML = (
    "project:\n  type: website\n  pre-render:\n    - tools/a.py\n"
    "    # comment inside pre-render\n\n    - tools/skip-pdf-in-previews.py\n"
    "  resources:\n    - 404.html\n"
)
stripped_pre = sp.PRE_RENDER_BLOCK.sub("", WEBSITE_YML, count=1)
check(
    "PRE_RENDER_BLOCK strips pre-render and stops at the next 2-space key",
    "pre-render:" not in stripped_pre
    and "tools/a.py" not in stripped_pre
    and "  resources:\n    - 404.html\n" in stripped_pre,
    stripped_pre,
)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    no_pre_yml = tmp_path / "no-pre.yml"
    no_pre_yml.write_text("project:\n  type: website\n", encoding="utf-8")
    try:
        sp.render_pdf_pass(no_pre_yml)
        check("render_pdf_pass raises RuntimeError when pre-render block is missing", False)
    except RuntimeError:
        check("render_pdf_pass raises RuntimeError when pre-render block is missing", True)

    proj_yml = tmp_path / "_quarto-website.yml"
    proj_yml.write_text(WEBSITE_YML, encoding="utf-8")
    fixed_mtime = 1_700_000_000
    os.utime(proj_yml, (fixed_mtime, fixed_mtime))
    recorded: dict[str, object] = {}
    orig_run = sp.subprocess.run

    def fake_run(cmd: list[str], cwd: Path, env: dict[str, str], check: bool) -> None:
        recorded["cmd"] = cmd
        recorded["env_guard"] = env.get("LDS_IN_PDF_PASS")
        recorded["during_text"] = proj_yml.read_text(encoding="utf-8")

    sp.subprocess.run = fake_run
    try:
        sp.render_pdf_pass(proj_yml)
    finally:
        sp.subprocess.run = orig_run

    check(
        "render_pdf_pass strips pre-render while child quarto render --to pdf runs",
        "pre-render:" not in str(recorded.get("during_text", ""))
        and recorded.get("env_guard") == "1"
        and recorded.get("cmd", [])[-2:] == ["--to", "pdf"],
        str(recorded),
    )
    check(
        "render_pdf_pass restores _quarto-website.yml content and mtime afterwards",
        proj_yml.read_text(encoding="utf-8") == WEBSITE_YML
        and int(proj_yml.stat().st_mtime) == fixed_mtime,
    )

    orig_root = sp.ROOT
    sp.ROOT = tmp_path
    try:
        chapters_dir = tmp_path / "chapters"
        chapters_dir.mkdir(parents=True)
        chapter_path = chapters_dir / "sample.qmd"
        chapter_path.write_text(CHAPTER, encoding="utf-8")
        os.utime(chapter_path, (fixed_mtime, fixed_mtime))
        proj_yml.write_text(WEBSITE_YML + PROJECT, encoding="utf-8")
        os.utime(proj_yml, (fixed_mtime, fixed_mtime))

        freeze_dir = tmp_path / "_freeze" / "chapters" / "sample" / "execute-results"
        freeze_dir.mkdir(parents=True)
        old_src = CHAPTER
        new_src = sp.strip_pdf(CHAPTER, front_matter_only=True)
        old_md5 = sp.hashlib.md5(old_src.encode("utf-8")).hexdigest()
        new_md5 = sp.hashlib.md5(new_src.encode("utf-8")).hexdigest()
        html_json = freeze_dir / "html.json"
        tex_json = freeze_dir / "tex.json"
        html_json.write_text(f'{{"hash": "{old_md5}", "result": []}}\n', encoding="utf-8")
        tex_json.write_text(f'{{"hash": "{old_md5}", "result": []}}\n', encoding="utf-8")

        shared_yml = tmp_path / "_quarto.yml"
        shared_yml.write_text(
            "profile:\n  default: website\nformat:\n  html: default\n  revealjs: default\n  pdf: default\n",
            encoding="utf-8",
        )
        os.utime(shared_yml, (fixed_mtime, fixed_mtime))
        stale_idx = tmp_path / ".quarto" / "idx" / "chapters"
        stale_idx.mkdir(parents=True)
        (stale_idx / "sample.qmd.json").write_text("{}", encoding="utf-8")

        # Verify that during the inner PDF pass, chapter_path still has its
        # unstripped content (matching tex.json), and after main() finishes in
        # split mode, html.json has md5(new), tex.json still has md5(old), and
        # no preview-no-pdf.txt marker is written.
        pdf_pass_chapter_hash: list[str] = []

        def fake_pdf_pass_run(
            cmd: list[str], cwd: Path, env: dict[str, str], check: bool
        ) -> None:
            cur = chapter_path.read_text(encoding="utf-8")
            pdf_pass_chapter_hash.append(sp.hashlib.md5(cur.encode("utf-8")).hexdigest())

        sp.subprocess.run = fake_pdf_pass_run
        saved_env = {
            k: os.environ.pop(k, None)
            for k in ["LDS_SKIP_PDF", "LDS_SPLIT_PDF", "LDS_IN_PDF_PASS"]
        }
        try:
            os.environ["LDS_SKIP_PDF"] = "0"
            os.environ["LDS_SPLIT_PDF"] = "1"
            rc = sp.main()
        finally:
            sp.subprocess.run = orig_run
            for k, v in saved_env.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

        check(
            "main() in split mode runs PDF pass before stripping chapter (matching tex.json)",
            rc == 0 and pdf_pass_chapter_hash == [old_md5],
            str(pdf_pass_chapter_hash),
        )
        check(
            "main() in split mode rewrites md5(old) to md5(new) in html.json and keeps tex.json",
            f'"hash": "{new_md5}"' in html_json.read_text(encoding="utf-8")
            and f'"hash": "{old_md5}"' in tex_json.read_text(encoding="utf-8"),
        )
        check(
            "main() in split mode does not write preview-no-pdf.txt and preserves chapter mtime",
            not (tmp_path / sp.MARKER).exists()
            and int(chapter_path.stat().st_mtime) == fixed_mtime,
        )
        check(
            "main() strips pdf: from _quarto.yml and clears stale .quarto/idx",
            "pdf:" not in shared_yml.read_text(encoding="utf-8")
            and "html: default" in shared_yml.read_text(encoding="utf-8")
            and int(shared_yml.stat().st_mtime) == fixed_mtime
            and not (tmp_path / ".quarto" / "idx").exists(),
        )
    finally:
        sp.ROOT = orig_root

if FAILURES:
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1)
print("\nall passed")
