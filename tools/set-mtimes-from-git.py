"""Set each tracked source file's modification time to its last commit time.

Quarto's `date: last-modified` reads a file's modification time, and a fresh
checkout (every CI render) stamps every file with the checkout time, so
without this each page would claim it changed on the day the site was built.
Run as a Quarto `pre-render` script. Files with uncommitted changes keep
their own time, so a local render still shows today's date for them.
"""

import os
import subprocess
import sys


def git(*args):
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def warn(message):
    print(f"set-mtimes-from-git: {message}", file=sys.stderr)


def unshallow():
    if git("rev-parse", "--is-shallow-repository").strip() != "true":
        return
    try:
        git("fetch", "--quiet", "--unshallow")
    except subprocess.CalledProcessError as err:
        warn(
            "shallow clone and `git fetch --unshallow` failed, so dates are "
            f"the newest fetched commit's: {err.stderr.strip()}"
        )


def last_commit_times():
    times = {}
    stamp = None
    log = git("log", "--format=%x00%ct", "--name-only", "--no-renames", "HEAD")
    for line in log.splitlines():
        if line.startswith("\0"):
            stamp = int(line[1:])
        elif line and line not in times:
            times[line] = stamp
    return times


def main():
    try:
        git("rev-parse", "--git-dir")
    except (subprocess.CalledProcessError, FileNotFoundError):
        warn("not a git checkout; leaving modification times alone")
        return
    root = git("rev-parse", "--show-toplevel").strip()
    os.chdir(root)
    unshallow()
    dirty = {
        line[3:]
        for line in git("status", "--porcelain", "--no-renames").splitlines()
    }
    times = last_commit_times()
    tracked = git("ls-files", "*.qmd").splitlines()
    set_count = 0
    for path in tracked:
        if path in dirty or path not in times or not os.path.exists(path):
            continue
        os.utime(path, (times[path], times[path]))
        set_count += 1
    print(f"set-mtimes-from-git: set {set_count} of {len(tracked)} .qmd file(s)")


main()
