#!/usr/bin/env python3
"""Self-test for the PDF accent fix in tools/build_math_macros.py.

The macros file defines the transpose as ``\\def\\'{^{\\top}}``, which replaces
LaTeX's text accent. The generated preamble must save the accent before that
definition and restore a mode-aware ``\\'`` after it. Run with
``python3 tools/tests/test_build_math_macros.py``. Needs no pytest.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
FAILURES: list[str] = []


def check(name: str, condition: bool) -> None:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    if not condition:
        FAILURES.append(name)


subprocess.run([sys.executable, str(ROOT / "tools" / "build_math_macros.py")], check=True)
tex = (ROOT / "_math-macros.tex").read_text(encoding="utf-8")
save = tex.find("\\let\\ddsacute\\'")
define = tex.find("\\def\\'{^{\\top}}")
restore = tex.find("\\DeclareRobustCommand{\\'}")
check("accent is saved", save >= 0)
check("transpose definition present", define >= 0)
check("mode-aware \\' is declared", restore >= 0)
check("save precedes the macros, which precede the restore", 0 <= save < define < restore)
check("restore falls back to the saved accent", "\\expandafter\\ddsacute\\fi" in tex[restore:])

if FAILURES:
    sys.exit(f"{len(FAILURES)} failure(s)")
print("all passed")
