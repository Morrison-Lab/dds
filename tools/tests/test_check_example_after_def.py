#!/usr/bin/env python3
"""Self-test for tools/check_example_after_def.py.

The check passes vacuously if it never finds a definition, or if it counts
anything as an example.  So these cases pin each outcome on a small fake
chapter: a followed definition, an unfollowed one, a theorem whose proof sits
between it and its example, a run of definitions sharing one example, a
theorem that does not share, an exercise (accepted unless --strict), and an
example reached through an include.

Run with ``python3 tools/tests/test_check_example_after_def.py``.  Needs no pytest.
"""

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import check_example_after_def as ced  # noqa: E402

FAILURES = []


def check(name, condition, detail=""):
    if condition:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name}{': ' + detail if detail else ''}")
        FAILURES.append(name)


CHAPTER = """---
title: Fake
---

::: {#def-followed}
A.
:::

{{< slidebreak >}}

::: {#exm-a}
An example.
:::

::: {#def-bare}
B.
:::

Plain text.

::: {#thm-proved}
C.
:::

::: proof
Because.
:::

{{< include _subfiles/_exm-c.qmd >}}

::: {#def-run-one}
D.
:::

::: {#def-run-two}
E.
:::

::: {#exm-run}
Shared.
:::

::: {#def-callout-between}
H.
:::

::: {.callout-note}
Synonyms.
:::

::: {#exm-callout}
After a callout.
:::

::: {#def-rem-between}
J.
:::

::: {#rem-after-def}
Commentary on the definition.
:::

::: {#exm-rem}
After a remark.
:::

::: {#def-comment-between}
I.
:::

<!--
A multi-line comment
that is not content.
-->

::: {#exm-comment}
After a comment.
:::

::: {#def-chunk-between}
K.
:::

```{r}
#| label: fig-chunk
plot(1)
```

::: {#exm-chunk}
After a figure chunk.
:::

::: {#def-div-between}
L.
:::

::: {#fig-div}
A figure div.
:::

::: {#exm-div}
After a figure div.
:::

::: {#def-code-between}
M.
:::

```{r}
#| label: tbl-not-a-figure
1 + 1
```

::: {#exm-code}
After plain code.
:::

::: {#thm-alone}
F.
:::

::: {#thm-after-def}
G.
:::

::: {#exr-apply}
Apply it.
:::
"""

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    (root / "chapters" / "_subfiles").mkdir(parents=True)
    (root / "chapters" / "fake.qmd").write_text(CHAPTER)
    (root / "chapters" / "_subfiles" / "_exm-c.qmd").write_text("::: {#exm-c}\nC example.\n:::\n")
    got = {r[0]: r[3] for r in ced.audit(root / "chapters" / "fake.qmd")}
    strict = {r[0]: r[3] for r in ced.audit(root / "chapters" / "fake.qmd", strict=True)}

    check("finds every definition and result div", len(got) == 13, str(got))
    check("a definition followed by an example passes", got["def-followed"])
    check("a definition followed by text fails", not got["def-bare"])
    check("a proof between theorem and example is skipped", got["thm-proved"])
    # thm-proved is followed by a proof, then an include that expands to exm-c.
    nxt = {r[0]: r[4] for r in ced.audit(root / "chapters" / "fake.qmd")}
    check("an example reached through an include counts", nxt["thm-proved"] == "exm-c", nxt["thm-proved"])
    check("a run of definitions shares the example after it", got["def-run-one"] and got["def-run-two"])
    check("a callout between definition and example is skipped", got["def-callout-between"])
    check("a remark between definition and example is skipped", got["def-rem-between"])
    check("a figure chunk between definition and example is skipped", got["def-chunk-between"])
    check("a figure div between definition and example is skipped", got["def-div-between"])
    check("other code between definition and example is not skipped", not got["def-code-between"])
    check("a multi-line comment between definition and example is skipped", got["def-comment-between"])
    check("a theorem does not share the example before it", not got["thm-alone"])
    check("a theorem followed by an exercise passes", got["thm-after-def"])
    check("--strict refuses an exercise", not strict["thm-after-def"])

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warn_only = ced.main(["--root", str(root)])
        failing = ced.main(["--root", str(root), "--fail"])
    check("warns without failing by default", warn_only == 0)
    check("--fail exits 1 when a div lacks an example", failing == 1)

if FAILURES:
    print(f"{len(FAILURES)} failed")
    sys.exit(1)
print("all passed")
