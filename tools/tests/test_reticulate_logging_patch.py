#!/usr/bin/env python3
"""Self-test for tools/reticulate_logging_patch.py.

The patch replaces reticulate's `rpytools.output._setStream`. The cases
pinned here: a handler whose `stream` is a read-only property (torch's
_StderrHandler) is left alone instead of raising, an ordinary StreamHandler
is still redirected, and running the patch twice does not wrap it twice.
A stand-in `rpytools.output` copies reticulate 1.47.0's `_setStream`, so the
test needs neither R nor torch.

Run with ``python3 tools/tests/test_reticulate_logging_patch.py``. Needs no pytest.
"""

from __future__ import annotations

import io
import logging
import runpy
import sys
import types
from pathlib import Path

PATCH = Path(__file__).resolve().parent.parent / "reticulate_logging_patch.py"

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


def _set_stream_1_47(handler, stream):
    """reticulate 1.47.0's rpytools.output._setStream, verbatim."""
    setStream = getattr(handler, "setStream", None)
    if setStream is not None:
        return setStream(stream)
    old_stream = handler.stream
    handler.stream = stream
    return old_stream


output = types.ModuleType("rpytools.output")
output._setStream = _set_stream_1_47
package = types.ModuleType("rpytools")
package.output = output
sys.modules["rpytools"] = package
sys.modules["rpytools.output"] = output


class ReadOnlyStderrHandler(logging.StreamHandler):
    """Shaped like torch._logging._internal._StderrHandler."""

    def __init__(self):
        logging.Handler.__init__(self)

    @property
    def stream(self):
        return sys.stderr


read_only = ReadOnlyStderrHandler()
try:
    output._setStream(read_only, io.StringIO())
    unpatched_raises = False
except AttributeError:
    unpatched_raises = True
check("unpatched, a read-only stream raises (the bug is reproduced)", unpatched_raises)

runpy.run_path(str(PATCH))
patched = output._setStream

try:
    returned = output._setStream(read_only, io.StringIO())
    check("patched, a read-only stream is left alone", returned is sys.stderr, repr(returned))
except AttributeError as err:
    check("patched, a read-only stream is left alone", False, str(err))

old, new = io.StringIO(), io.StringIO()
ordinary = logging.StreamHandler(old)
returned = output._setStream(ordinary, new)
check("patched, an ordinary handler is still redirected", ordinary.stream is new and returned is old)

runpy.run_path(str(PATCH))
check("running the patch twice does not wrap it twice", output._setStream is patched)

if FAILURES:
    print(f"\n{len(FAILURES)} failure(s)")
    sys.exit(1)
print("\nall passed")
