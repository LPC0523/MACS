"""Assemble a distributable release folder and ZIP from `dist/MACS/`.

The PyInstaller output is already self-contained -- it carries its own Python,
Qt, ANTsPy, ONNX Runtime and VC++ runtime, so a target laptop needs nothing
installed. What it is missing is the wrapping a person receiving it needs: a
launcher that works no matter where the folder is dropped, a readme in Chinese,
and the manuals. This script adds exactly that, then reports the size and
SHA-256 that go into CHANGELOG.md.

    python scripts/make_release.py            # folder + ZIP
    python scripts/make_release.py --no-zip   # folder only

Run `dist/MACS/MACS.exe --selftest` first; this script refuses to package a
build that has not passed it (pass --skip-selftest to override).
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent          # shimming_qt/
REPO_DIR = APP_DIR.parent                                  # shimming_app_latest/
DIST = APP_DIR / "dist" / "MACS"
RELEASE_ROOT = REPO_DIR / "release"
DOCS_SRC = REPO_DIR / "docs"

# Launcher: cd to its own folder first, so a desktop shortcut or a double-click
# from a different working directory still finds _internal/ next to the exe.
LAUNCHER = """@echo off
rem Launch from this file's own folder, so double-clicking works no matter
rem where the shortcut lives or what the working directory happens to be.
cd /d "%~dp0"
start "" "MACS.exe"
"""

README = """MACS {ver} — 多通道主动匀场平台
================================

安装
----
免安装。把整个 {name} 文件夹复制到本机硬盘，双击「启动 MACS.bat」即可。

不要只复制 MACS.exe —— 它需要同目录下的 _internal 文件夹才能运行。
也不要在压缩包里直接运行，先解压到硬盘。

解压位置不要太深。程序内部最长的文件路径有 115 个字符，而 Windows 默认
上限是 260，所以解压目标路径请控制在 140 字符以内 —— 桌面、D:\\MACS 这类
都没问题；解压到层层嵌套的网盘同步目录里可能会报「找不到路径的一部分」。

系统要求
--------
Windows 10 / 11，64 位。无需安装 Python、FSL、ANTs 或其他任何依赖，
也不需要联网和管理员权限。

第一次使用
----------
1. 打开软件，切到「系统自检」页，点「重新检查」，确认全绿。
2. 回到「一键匀场」，把 DICOM 文件夹拖进去。
3. 选线圈，点「开始匀场」。

详细说明见「文档\\用户手册.md」。

文件夹里有什么
--------------
MACS.exe              主程序
启动 MACS.bat         启动脚本（推荐用这个）
_internal\\            运行所需文件，不要删改
文档\\用户手册.md      操作手册
文档\\CHANGELOG.md     版本变更记录
文档\\协议_电流表.md   动态匀场电流表协议（供固件对接）

可以替换的文件
--------------
下列文件是普通文件，更新时直接替换即可，不需要重新打包程序：

_internal\\config\\coils\\*.json     线圈配置
_internal\\coils\\                  线圈场图数据
_internal\\atlases\\                MNI 图谱与前额叶模板
_internal\\model\\bet_micro.onnx    脑提取模型

用户自己导入的线圈保存在 C:\\Users\\<用户名>\\.macs\\ 下，卸载本文件夹不会丢失。

已知限制
--------
- 首次启动约需 10 秒（加载运行库）。
- 数据文件夹路径可以含中文。
- 启动时会带一个黑色命令行窗口，那是运行日志，出问题时请连它一起截图。
"""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tree_size(root: Path) -> tuple[int, int]:
    """(file count, total bytes) under `root`."""
    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def _read_version() -> str:
    sys.path.insert(0, str(APP_DIR))
    from core.version import VERSION
    return VERSION


def _run_selftest() -> None:
    exe = DIST / "MACS.exe"
    print(f"→ running {exe.name} --selftest …")
    proc = subprocess.run([str(exe), "--selftest"], capture_output=True,
                          text=True, encoding="utf-8", errors="replace",
                          timeout=600)
    tail = (proc.stdout or "").strip().splitlines()
    for line in tail:
        if line.startswith(("[FAIL]", "SELFTEST")):
            print("   " + line)
    if proc.returncode != 0 or "SELFTEST PASSED" not in (proc.stdout or ""):
        raise SystemExit(
            "selftest did not pass — refusing to package.\n"
            f"{proc.stdout}\n{proc.stderr}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-zip", action="store_true", help="skip the ZIP step")
    ap.add_argument("--skip-selftest", action="store_true")
    args = ap.parse_args()

    if not (DIST / "MACS.exe").is_file():
        raise SystemExit(f"no build at {DIST} — run build.bat first")

    if not args.skip_selftest:
        _run_selftest()

    version = _read_version()
    name = f"MACS_v{version}"
    out = RELEASE_ROOT / name

    if out.exists():
        print(f"→ replacing existing {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"→ copying {DIST} → {out}")
    shutil.copytree(DIST / "_internal", out / "_internal")
    shutil.copy2(DIST / "MACS.exe", out / "MACS.exe")

    (out / "启动 MACS.bat").write_text(LAUNCHER, encoding="utf-8", newline="\r\n")
    (out / "自述.txt").write_text(README.format(ver=f"v{version}", name=name),
                                  encoding="utf-8", newline="\r\n")

    docs_out = out / "文档"
    docs_out.mkdir()
    for src, dst in ((DOCS_SRC / "用户手册.md", "用户手册.md"),
                     (DOCS_SRC / "协议_电流表.md", "协议_电流表.md"),
                     (APP_DIR / "CHANGELOG.md", "CHANGELOG.md")):
        if not src.is_file():
            raise SystemExit(f"missing document: {src}")
        shutil.copy2(src, docs_out / dst)

    n_files, n_bytes = _tree_size(out)
    exe = out / "MACS.exe"
    print(f"\n=== {name} ===")
    print(f"folder      : {out}")
    print(f"             {n_files:,} files, {n_bytes / 1024**2:,.1f} MB")
    print(f"MACS.exe    : {exe.stat().st_size:,} bytes "
          f"({exe.stat().st_size / 1024**2:.2f} MB)")
    print(f"exe SHA-256 : {_sha256(exe)}")

    if not args.no_zip:
        zip_path = RELEASE_ROOT / f"{name}.zip"
        print(f"\n→ zipping → {zip_path} (this takes a few minutes)")
        # ZIP_DEFLATED, not stored: the payload is mostly DLLs and NIfTI, which
        # compress well, and every recipient's Windows can open a deflate ZIP
        # without installing anything.
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for p in sorted(out.rglob("*")):
                if p.is_file():
                    zf.write(p, p.relative_to(RELEASE_ROOT))
        print(f"zip         : {zip_path.stat().st_size:,} bytes "
              f"({zip_path.stat().st_size / 1024**2:.1f} MB)")
        print(f"zip SHA-256 : {_sha256(zip_path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
