"""Version identity for MACS.

`VERSION` is the source of truth and is edited by hand. The build stamp and git
SHA are filled in by `scripts/stamp_version.py`, which the PyInstaller spec runs
before freezing -- a frozen build therefore says exactly which commit produced
it, while a dev run says "dev" rather than inheriting a stale stamp from the
last package.
"""

from __future__ import annotations

VERSION = "1.5.6"
PRODUCT_NAME = "MACS"
PRODUCT_TAGLINE_ZH = "多通道主动匀场平台"
PRODUCT_TAGLINE_EN = "Multi-channel Active Coil Shimming"

#: Overwritten in place by scripts/stamp_version.py at package time.
BUILD_DATE = ""
GIT_SHA = ""


def _dev_sha() -> str:
    """Git SHA of the working tree, for runs that were never packaged."""
    import subprocess
    from pathlib import Path

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parent.parent,
                capture_output=True, text=True, timeout=5)
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "+"
            return sha
    except Exception:
        pass
    return "unknown"


def build_info() -> dict:
    """Everything the log line, title bar and JSON report need."""
    stamped = bool(BUILD_DATE)
    return {
        "version": VERSION,
        "build_date": BUILD_DATE or "dev",
        "git_sha": GIT_SHA or _dev_sha(),
        "packaged": stamped,
    }


def version_string(short: bool = False) -> str:
    """`v1.2.0` or `v1.2.0 (build 2026-08-06T10:15, abc1234)`."""
    info = build_info()
    if short:
        return f"v{info['version']}"
    date = info["build_date"]
    if date != "dev" and "T" in date:
        date = date[:16]          # to the minute; seconds are noise here
    return f"v{info['version']} (build {date}, {info['git_sha']})"


def banner(lang: str = "zh") -> str:
    """First line of the log."""
    tag = PRODUCT_TAGLINE_ZH if lang == "zh" else PRODUCT_TAGLINE_EN
    return f"{PRODUCT_NAME} {version_string()} — {tag}"
