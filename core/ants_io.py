"""ANTsPy file I/O that survives non-ASCII paths.

ANTsPy passes the filename straight down to ITK's C++ layer, which opens it
through the narrow Win32 API and therefore decodes it in the active ANSI code
page. A subject folder named `扫描数据_ScanData` then fails with

    RuntimeError: Could not create ImageIO object for file ...

This is the same defect fixed in cpp_bet (see cpp_bet/include/fs_path.h), but
ANTsPy ships as a prebuilt wheel, so it cannot be fixed at the source. Instead
the file is staged through a scratch directory whose path is pure ASCII, and
only Python -- which handles Unicode paths correctly -- touches the real
location.

Staging costs one copy of a few MB per registration; the alternative is telling
users their data folder may not have a Chinese name.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
import uuid
from pathlib import Path


def _is_ascii(s: str) -> bool:
    return all(ord(c) < 128 for c in s)


def _short_path(p: str) -> str | None:
    """Windows 8.3 form of an existing path, which is always ASCII."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        need = fn(p, None, 0)
        if not need:
            return None
        buf = ctypes.create_unicode_buffer(need)
        if not fn(p, buf, need):
            return None
        return buf.value
    except Exception:
        return None


_scratch_base: str | None = None


def _ascii_scratch_base() -> str:
    """A writable directory whose absolute path contains only ASCII."""
    global _scratch_base
    if _scratch_base is not None:
        return _scratch_base

    candidates = [os.environ.get("MACS_SCRATCH"), tempfile.gettempdir()]
    for cand in candidates:
        if not cand or not os.path.isdir(cand):
            continue
        cand = os.path.abspath(cand)
        if _is_ascii(cand):
            _scratch_base = cand
            return _scratch_base
        # A Chinese *user name* makes %TEMP% itself non-ASCII; 8.3 is ASCII by
        # construction, so it rescues that case when the volume still has 8.3
        # name generation enabled.
        short = _short_path(cand)
        if short and _is_ascii(short):
            _scratch_base = short
            return _scratch_base

    raise RuntimeError(
        "找不到纯 ASCII 的临时目录，ANTs 配准无法进行。\n"
        "请设置环境变量 MACS_SCRATCH 指向一个英文路径，例如 C:\\macs_tmp。")


def _suffix(path: str) -> str:
    """Keep .nii.gz intact -- ITK selects its reader from the extension."""
    low = path.lower()
    for s in (".nii.gz", ".mha", ".mhd", ".nrrd", ".nii"):
        if low.endswith(s):
            return s
    return Path(path).suffix or ".nii.gz"


def _staged_name(path: str) -> str:
    return os.path.join(_ascii_scratch_base(),
                        f"macs_{uuid.uuid4().hex}{_suffix(path)}")


def image_read(path):
    """ants.image_read that works when `path` contains non-ASCII characters."""
    import ants

    path = str(path)
    if _is_ascii(path):
        return ants.image_read(path)

    tmp = _staged_name(path)
    try:
        shutil.copyfile(path, tmp)
        return ants.image_read(tmp)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def image_write(img, path) -> None:
    """ants.image_write that works when `path` contains non-ASCII characters."""
    import ants

    path = str(path)
    if _is_ascii(path):
        ants.image_write(img, path)
        return

    tmp = _staged_name(path)
    try:
        ants.image_write(img, tmp)
        _commit(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _commit(tmp: str, dest: str) -> None:
    # os.replace, not shutil.move: replace is atomic on the same volume and
    # overwrites an existing file instead of failing.
    try:
        os.replace(tmp, dest)
    except OSError:
        shutil.copyfile(tmp, dest)
        os.remove(tmp)


# ─────────────────────────────────────────────────────────────────────────
#  For ANTs running in a *separate* interpreter
# ─────────────────────────────────────────────────────────────────────────
# `_flirt_antspy_subprocess` builds a Python script with the paths baked into
# the source and runs it under a different interpreter, which may not have this
# package importable. Staging therefore has to happen here, in the parent, so
# the generated script only ever sees ASCII.

@contextlib.contextmanager
def ascii_inputs(*paths):
    """ASCII-safe readable copies of existing files; originals untouched."""
    staged: list[str] = []
    usable: list[str] = []
    try:
        for p in paths:
            p = str(p)
            if _is_ascii(p):
                usable.append(p)
                continue
            tmp = _staged_name(p)
            shutil.copyfile(p, tmp)
            staged.append(tmp)
            usable.append(tmp)
        yield usable
    finally:
        for tmp in staged:
            try:
                os.remove(tmp)
            except OSError:
                pass


@contextlib.contextmanager
def ascii_output(path):
    """An ASCII-safe destination, moved onto `path` if the block succeeds.

    Yields "" for a falsy `path`, matching the callers that treat an empty
    output path as "don't write one".
    """
    if not path:
        yield ""
        return

    path = str(path)
    if _is_ascii(path):
        yield path
        return

    tmp = _staged_name(path)
    try:
        yield tmp
        if os.path.exists(tmp):
            _commit(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
