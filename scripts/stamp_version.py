"""Write the build stamp into core/version.py, then put it back.

Run by macs.spec at package time. Only the BUILD_DATE and GIT_SHA lines are
rewritten -- VERSION stays hand-edited, so a build can never silently bump the
version number.

The stamp is reverted once the build finishes (`restore()`), because otherwise
every package would leave core/version.py modified in the working tree and the
next build's `git status --porcelain` check would report it as dirty.
"""

from __future__ import annotations

import datetime
import re
import subprocess
from pathlib import Path

_VERSION_PY = Path(__file__).resolve().parent.parent / "core" / "version.py"

_original: str | None = None


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=_VERSION_PY.parent.parent,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def stamp() -> dict:
    """Bake build date and git SHA into core/version.py. Returns what was set."""
    global _original

    # Read and write bytes, never text. `write_text` translates a newline to
    # the platform newline, so on Windows a file checked out with LF came back
    # as CRLF -- identical content, different bytes. git then reported
    # version.py as modified after every package build, and the *next* build
    # stamped a "+" onto a tree whose only change was the previous build's own
    # restore.
    raw = _VERSION_PY.read_bytes()
    if _original is None:
        _original = raw
    text = raw.decode("utf-8")

    sha = _git("rev-parse", "--short", "HEAD") or "unknown"
    if _git("status", "--porcelain"):
        sha += "+"                       # built from a dirty tree
    date = datetime.datetime.now().replace(microsecond=0).isoformat()

    # No end-of-line anchor. The file is read as bytes so the restore stays
    # exact, which means its line endings survive; on a CRLF checkout the line
    # ends with a carriage return that a trailing quote in the pattern can
    # never reach, so the anchored version silently matched nothing and the
    # build shipped unstamped. That is why the substitution count is checked
    # rather than assumed. A dot never crosses a newline, so dropping the
    # anchor still confines the match to one line.
    text, n_date = re.subn(r'^BUILD_DATE = ".*"', f'BUILD_DATE = "{date}"',
                           text, count=1, flags=re.M)
    text, n_sha = re.subn(r'^GIT_SHA = ".*"', f'GIT_SHA = "{sha}"',
                          text, count=1, flags=re.M)
    if not n_date or not n_sha:
        raise RuntimeError(
            "stamp_version: could not find BUILD_DATE/GIT_SHA in %s "
            "(matched date=%d sha=%d) -- refusing to produce an unstamped build"
            % (_VERSION_PY, n_date, n_sha))
    _VERSION_PY.write_bytes(text.encode("utf-8"))
    return {"build_date": date, "git_sha": sha}


def restore() -> None:
    """Undo stamp(), leaving the working tree as it was."""
    global _original
    if _original is not None:
        # The exact bytes read before stamping, so the file is byte-for-byte
        # what it was and the tree stays clean.
        _VERSION_PY.write_bytes(_original)
        _original = None


if __name__ == "__main__":
    print(stamp())
