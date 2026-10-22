#!/bin/bash
# SessionStart hook: install the render toolchain in cloud sessions.
#
# A fresh cloud container has R, Quarto and Julia but not what a render of
# this site needs, and each gap fails differently (measured 2026-10-04):
# - no knitr or rmarkdown, so any page with a code chunk fails at once;
# - no RCall or Suppressor for JuliaCall, and the Julia package server
#   (pkg.julialang.org) is refused by the agent proxy, so `Pkg.add` fails
#   until JULIA_PKG_SERVER is empty and Pkg reads the registry from GitHub;
# - no rsvg-convert, so a PDF handout with an SVG figure fails;
# - no TinyTeX (the installers are refused), or one that is not on PATH;
# - no typos or lintr for the pre-commit checklist.
#
# The R packages are read from the `r-packages` inputs of the two render
# workflows, so this hook never drifts from CI.
# Installing takes a few minutes, so it runs detached and the hook returns at
# once; every firing (startup, resume, clear, compact) tells the session where
# the log and the done-marker are, and which exports a render needs.
#
# Local sessions are left alone.

set -u

[ "${CLAUDE_CODE_REMOTE:-}" = "true" ] || exit 0

root="${CLAUDE_PROJECT_DIR:-$(pwd)}"
state="${TMPDIR:-/tmp}/dds-render-toolchain"
mkdir -p "$state"
log="$state/install.log"
done_marker="$state/done"
lock="$state/lock"

tinytex="$HOME/.TinyTeX/bin/x86_64-linux"
# Always put TinyTeX on PATH: on a fresh container the install below creates the
# directory after this message is written, and a harmless missing PATH entry
# beats a first-firing message that sends the render to a TeX-less shell.
exports="export R_LD_LIBRARY_PATH=\"\$(R RHOME)/lib\" PATH=\"$tinytex:\$PATH\""

say() {
  jq -n --arg m "$1" '{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": $m}}'
}

if [ -e "$done_marker" ]; then
  say "The render toolchain is installed (log: $log). Before rendering, run: $exports"
  exit 0
fi

# `quarto install tinytex` and the TinyTeX installer script fail here: both ask
# api.github.com or github.com/.../releases/latest, which the proxy refuses
# (403), and the TeX Live mirrors tlmgr defaults to are unreachable. What works
# (measured 2026-10-12): the release asset by tag, then tlmgr pointed at MIT's
# CTAN mirror. A TinyTeX tarball is a year behind the mirror's TeX Live, and
# tlmgr refuses a cross-release install, so update-tlmgr-latest runs first.
install_tinytex() {
  local tmp tag year month i
  tmp="$(mktemp -d)"
  trap 'rm -rf "${tmp:-}"' RETURN
  year="$(date -u +%Y)"
  month="$((10#$(date -u +%m)))"
  # Newest tag that exists, going back up to 18 months; tags are vYYYY.MM.
  for i in $(seq 0 17); do
    tag="$(printf 'v%d.%02d' "$year" "$month")"
    if curl -fsIL "https://github.com/rstudio/tinytex-releases/releases/download/$tag/TinyTeX-1-$tag.tar.gz" >/dev/null; then
      break
    fi
    tag=""
    month=$((month - 1))
    if [ "$month" -eq 0 ]; then month=12; year=$((year - 1)); fi
  done
  [ -n "$tag" ] || { echo "== no TinyTeX release tag found"; return 1; }
  echo "== TinyTeX $tag"
  curl -fsSL -o "$tmp/tinytex.tar.gz" \
    "https://github.com/rstudio/tinytex-releases/releases/download/$tag/TinyTeX-1-$tag.tar.gz" || return 1
  tar -xzf "$tmp/tinytex.tar.gz" -C "$HOME" || return 1
  local bin="$HOME/.TinyTeX/bin/x86_64-linux"
  local ctan="https://mirrors.mit.edu/CTAN/systems/texlive/tlnet"
  "$bin/tlmgr" option repository "$ctan" || return 1
  curl -fsSL -o "$tmp/update-tlmgr-latest.sh" "$ctan/update-tlmgr-latest.sh" || return 1
  PATH="$bin:$PATH" sh "$tmp/update-tlmgr-latest.sh" -- --update || return 1
  # The extras preview.yml installs for the handouts.
  "$bin/tlmgr" install luacolor lua-ul || return 1
}

install() {
  status=0

  mapfile -t r_pkgs < <(
    awk '/r-packages: \|/{f=1; next} f && /^ *any::/{sub(/^ *any::/, ""); print; next} {f=0}' \
      "$root/.github/workflows/preview.yml" "$root/.github/workflows/publish.yml" | sort -u
  )
  # An empty or knitr-less list means the workflows moved or changed shape;
  # installing only lintr would then report success with nothing usable.
  if ! printf '%s\n' "${r_pkgs[@]}" | grep -qx knitr; then
    echo "== no r-packages (with knitr) found in $root/.github/workflows/{preview,publish}.yml"
    status=1
  fi
  r_pkgs+=(lintr)
  echo "== R packages: ${r_pkgs[*]}"
  # Posit Package Manager serves prebuilt Linux binaries, which needs an R user agent.
  # The R code is single-quoted on purpose: its $ belongs to R, not the shell.
  # shellcheck disable=SC2016
  Rscript -e '
    codename <- tryCatch(system("lsb_release -cs", intern = TRUE), error = function(e) "noble")
    options(
      repos = c(CRAN = sprintf("https://packagemanager.posit.co/cran/__linux__/%s/latest", codename)),
      HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(),
        paste(getRversion(), R.version$platform, R.version$arch, R.version$os))
    )
    pkgs <- commandArgs(trailingOnly = TRUE)
    missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing)) install.packages(missing, Ncpus = 4)
    still <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
    if (length(still)) stop("not installed: ", paste(still, collapse = ", "))
  ' "${r_pkgs[@]}" || status=1

  if command -v julia >/dev/null; then
    echo "== Julia: RCall and Suppressor for JuliaCall"
    # RCall's build loads R, so give it the same library path a render uses.
    R_LD_LIBRARY_PATH="$(R RHOME)/lib" JULIA_PKG_SERVER="" julia -e '
      import Pkg
      need = filter(p -> Base.find_package(p) === nothing, ["RCall", "Suppressor"])
      isempty(need) || Pkg.add(need)
    ' || status=1
  fi

  if [ ! -d "$HOME/.TinyTeX/bin/x86_64-linux" ] && ! command -v tlmgr >/dev/null; then
    echo "== TinyTeX"
    install_tinytex || status=1
  fi

  if ! command -v rsvg-convert >/dev/null; then
    echo "== rsvg-convert"
    { apt-get install -y -qq librsvg2-bin ||
      { apt-get update -qq && apt-get install -y -qq librsvg2-bin; }; } || status=1
  fi

  if ! command -v typos >/dev/null; then
    echo "== typos"
    if command -v uv >/dev/null; then
      uv tool install typos || status=1
    else
      pip install -q typos || status=1
    fi
  fi

  if [ "$status" -eq 0 ]; then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$done_marker"
    echo "== done"
  else
    echo "== FAILED: see the steps above"
  fi
}

# One install at a time: SessionStart also fires on resume, clear and compact,
# and a second run would race the first on R's, Julia's and apt's locks.
# The installer holds an flock on $lock for as long as it runs; the kernel
# drops it when the process exits or dies, so no lock is ever left stale.
# If a firing slips in between the probe and the installer taking the lock,
# its installer loses the lock, says so in the log, and exits.
if flock -n "$lock" true; then
  # The installer script is single-quoted: its $ belong to the inner shell.
  # shellcheck disable=SC2016
  env root="$root" done_marker="$done_marker" lock="$lock" \
    setsid nohup bash -c "$(declare -f install_tinytex install)"'
      exec 9>> "$lock"
      if ! flock -n 9; then
        echo "== another install holds $lock; this one exits"
        exit 0
      fi
      echo "== $(date -u +%Y-%m-%dT%H:%M:%SZ) install started"
      install' \
    >> "$log" 2>&1 < /dev/null &
  started="is installing in the background"
else
  started="is already being installed by an earlier firing of this hook"
fi

say "The render toolchain (R packages from the CI workflows, JuliaCall's Julia packages, TinyTeX with luacolor and lua-ul, rsvg-convert, typos) $started; log: $log. Do not render until $done_marker exists: before then, a page with code fails for want of knitr, and a Julia chunk sends JuliaCall to the blocked package server. If the log ends with '== FAILED', fix that step rather than skipping a check. Before rendering, run: $exports"
