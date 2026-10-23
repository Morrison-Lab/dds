# Render toolchain in a cloud session

**Install whatever is missing.**
R and Python are always available here;
never skip a check, or design content, tools or CI, around their absence.

In a cloud session, the SessionStart hook `.claude/hooks/render-toolchain.sh` installs the toolchain in the background:
the R packages named in the two render workflows' `r-packages` (plus `lintr`),
TinyTeX (with `luacolor` and `lua-ul`), `rsvg-convert` and `typos`.
It tells the session where its log and done-marker are (`$TMPDIR/dds-render-toolchain/`)
and which exports a render needs.
Render only once the marker exists;
a log ending in `== FAILED` means a step needs fixing by hand, not skipping.

`quarto install tinytex` fails in a cloud session with a 403, because it reads the latest release from `api.github.com`.
The hook downloads the release asset by tag instead, from a URL that names the release.
