# Render toolchain in a cloud session

Moved out of `CLAUDE.md` unchanged.

**Install whatever is missing.**
R and Python are always available here;
never skip a check, or design content, tools or CI, around their absence.

In a cloud session, the SessionStart hook `.claude/hooks/render-toolchain.sh` starts that installation in the background:
the R packages named in the two render workflows' `r-packages` (plus `lintr`),
JuliaCall's RCall and Suppressor, TinyTeX (with `luacolor` and `lua-ul`), `rsvg-convert` and `typos`.
`quarto install tinytex` fails in a cloud session with a 403, because it reads the latest release from `api.github.com`,
and so does the TinyTeX installer script (`github.com/.../releases/latest`).
What works (measured 2026-10-12): download the release asset by tag
(`https://github.com/rstudio/tinytex-releases/releases/download/v2026.03/TinyTeX-1-v2026.03.tar.gz`) and unpack it into `~`,
then point `tlmgr` at MIT's CTAN mirror, `https://mirrors.mit.edu/CTAN/systems/texlive/tlnet` (note the capital `CTAN`; `ctan` is a 404),
and run that mirror's `update-tlmgr-latest.sh -- --update` once, because the tarball is TeX Live 2025 and `tlmgr` refuses to install across releases.
After that `tlmgr install` works and `lualatex` runs.
The hook does all of this, so a thread never needs to defer a PDF check to CI.
It tells the session where its log and done-marker are (`$TMPDIR/lds-render-toolchain/`)
and which exports a render needs.
Render only once the marker exists;
a log ending in `== FAILED` means a step needs fixing by hand, not skipping.
A fresh container lacks all of them (measured 2026-10-04).
JuliaCall cannot install its own dependencies there,
because the agent proxy refuses the Julia package server (`pkg.julialang.org`);
`JULIA_PKG_SERVER="" julia -e 'import Pkg; Pkg.add(["RCall", "Suppressor"])'`
reads the registry from GitHub instead, and is what the hook runs.

