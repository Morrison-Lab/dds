#!/usr/bin/env python3
"""Self-test for tools/reticulate_logging_patch.py.

Runs reticulate's own output-capture context, from the installed R package,
with a handler shaped like torch's _StderrHandler (a logging.Handler whose
`stream` is a read-only property returning sys.stderr) registered on a
logger. The cases pinned here: unpatched, the capture raises, as it does
in a render; patched, it captures and restores without raising; an
ordinary StreamHandler is still redirected; a read-only handler that does
not follow sys.stderr still raises, rather than escaping the capture; and
running the patch twice does not wrap it twice.

Needs R with reticulate installed (the render needs both anyway); it stops
with an error if it cannot find them. Run with
``python3 tools/tests/test_reticulate_logging_patch.py``. Needs no pytest.
"""

from __future__ import annotations

import io
import logging
import runpy
import subprocess
import sys
from pathlib import Path

PATCH = Path(__file__).resolve().parent.parent / "reticulate_logging_patch.py"

rpytools_dir = subprocess.run(
    ["Rscript", "-e", 'cat(system.file("python", package = "reticulate", mustWork = TRUE))'],
    capture_output=True, text=True, check=True,
).stdout.strip()
sys.path.insert(0, rpytools_dir)
import rpytools.output as output  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


class FollowsStderr(logging.Handler):
    """Shaped like torch._logging._internal._StderrHandler in torch 2.14."""

    @property
    def stream(self):
        return sys.stderr


class FixedReadOnly(logging.Handler):
    """A read-only stream that does not follow sys.stderr."""

    def __init__(self, target):
        super().__init__()
        self._target = target

    @property
    def stream(self):
        return self._target


def capture(handler):
    """Run one chunk's capture with `handler` registered; return (error, inside)."""
    logger = logging.getLogger("test_reticulate_logging_patch")
    logger.handlers = [handler]
    inside = None
    try:
        with output.OutputCaptureContext(capture_stdout=True, capture_stderr=True):
            inside = handler.stream
        return None, inside
    except AttributeError as err:
        return err, inside
    finally:
        logger.handlers = []
        sys.stdout, sys.stderr = real_stdout, real_stderr


real_stdout, real_stderr = sys.stdout, sys.stderr

err, _ = capture(FollowsStderr())
check("unpatched, the capture raises (the bug is reproduced)", err is not None)

runpy.run_path(str(PATCH))
patched = output._setStream

handler = FollowsStderr()
err, inside = capture(handler)
check("patched, the capture runs without raising", err is None, str(err))
check("patched, the handler writes to the capture buffer inside", inside is not real_stderr)
check("patched, the handler writes to the real stderr after", handler.stream is real_stderr)

ordinary = logging.StreamHandler(sys.stderr)
err, inside = capture(ordinary)
# reticulate 1.47.0 does not point an ordinary handler back afterwards,
# patched or not, so only the redirection inside is pinned here.
check(
    "patched, an ordinary handler is still redirected",
    err is None and inside is not real_stderr,
)

err, _ = capture(FixedReadOnly(sys.stderr))
check("patched, a read-only handler that does not follow sys.stderr still raises", err is not None)

runpy.run_path(str(PATCH))
check("running the patch twice does not wrap it twice", output._setStream is patched)

if FAILURES:
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1)
print("\nall passed")
