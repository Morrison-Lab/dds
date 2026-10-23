---
description: Spell-check changed files with typos
allowed-tools:
  - Bash(typos *)
---

Run `typos` on the changed files (install it with `pip install typos` if it is missing).
CI runs the same checker on the lines a PR adds.

Report misspellings grouped by file.
Add correctly spelled terms it misreads to `_typos.toml` rather than disabling the check.
Fix only misspellings your change introduced.
If there are none, say so.
