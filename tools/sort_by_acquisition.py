"""Split a console export into one folder per acquisition.

    D:\\python.exe tools/sort_by_acquisition.py <export_dir> [-o <out_dir>] [--move]

A shim session exports as one flat pile: every repeat of a protocol carries the
same SeriesDescription and is told apart only by SeriesNumber. The app converts
such a folder correctly -- names collide, so it prefixes them -- but it then has
24 complete field maps to choose from and takes the first by name, which is the
earliest, not the one you want. Sorting first means pointing the app at exactly
the acquisition you mean.

Grouping follows the console's numbering: acquisition N is series N01 (Echo1),
N02 (Echo2), N03 (phase). Verified on a 5T export -- 24 acquisitions, each
holding exactly those three, transmit frequency constant within each and
differing between them.

Copies by default. `--move` relocates instead, which is faster and does not
double the disk, but leaves the export changed; the copy is the safe default
because a mis-grouped move is not undoable.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _series_info(series_dir: Path):
    """(SeriesNumber, SeriesDescription, ImagingFrequency Hz) from one slice."""
    import pydicom

    for f in sorted(series_dir.iterdir()):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True,
                                 specific_tags=["SeriesNumber", "SeriesDescription",
                                                "ImagingFrequency", "Modality"])
        except Exception:
            continue
        sn = getattr(ds, "SeriesNumber", None)
        if sn is None:
            continue
        f0 = getattr(ds, "ImagingFrequency", None)
        return (int(sn), str(getattr(ds, "SeriesDescription", "") or ""),
                round(float(f0) * 1e6) if f0 else None)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="按采集编号拆分 DICOM 导出目录")
    ap.add_argument("export_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="输出目录，默认 <export_dir>_sorted")
    ap.add_argument("--move", action="store_true", help="移动而非复制")
    ap.add_argument("--dry-run", action="store_true", help="只打印分组，不动文件")
    a = ap.parse_args(argv)

    src = a.export_dir.resolve()
    if not src.is_dir():
        print(f"不是目录：{src}")
        return 2
    out = (a.out or src.parent / f"{src.name}_sorted").resolve()

    # A "series" is any directory holding DICOMs directly.
    series_dirs = sorted({p.parent for p in src.rglob("*.dcm") if p.is_file()})
    if not series_dirs:
        print(f"没有找到 .dcm 文件：{src}")
        return 2

    groups = defaultdict(list)
    skipped = []
    for d in series_dirs:
        info = _series_info(d)
        if info is None:
            skipped.append((d, "读不到 SeriesNumber"))
            continue
        sn, desc, f0 = info
        # Same rule the converter uses to break filename collisions.
        key = f"a{sn // 100:02d}" if sn >= 100 else f"s{sn}"
        groups[key].append((sn, desc, f0, d))

    print(f"{len(series_dirs)} 个序列 → {len(groups)} 个采集\n")
    print("%-6s %-5s %-38s %-13s %s" % ("采集", "序列", "描述", "f0 (Hz)", "组内一致"))
    for key in sorted(groups):
        items = sorted(groups[key])
        f0s = {f for _, _, f, _ in items if f is not None}
        # One acquisition has one transmit frequency. More than one here means
        # the grouping put unrelated series together, and every downstream
        # number would inherit that -- so it is called out, not tidied away.
        # ASCII markers: this prints to a GBK console, which cannot encode the
        # tick and warning glyphs used in the GUI log.
        flag = "OK" if len(f0s) <= 1 else f"[!] {len(f0s)} 个不同的 f0"
        for i, (sn, desc, f0, _) in enumerate(items):
            print("%-6s %-5d %-38s %-13s %s" % (
                key if i == 0 else "", sn, desc[:38],
                f0 if f0 is not None else "-", flag if i == 0 else ""))

    if a.dry_run:
        print("\n--dry-run：没有动任何文件")
        return 0

    if out.exists() and any(out.iterdir()):
        print(f"\n输出目录非空，先清空或换一个：{out}")
        return 2

    for key in sorted(groups):
        for _, _, _, d in sorted(groups[key]):
            dst = out / key / d.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if a.move:
                shutil.move(str(d), str(dst))
            else:
                shutil.copytree(d, dst)
    print(f"\n{'移动' if a.move else '复制'}完成 → {out}")
    for d, why in skipped:
        print(f"  跳过 {d.name}：{why}")
    print("\n把其中一个采集目录拖进 MACS 即可，例如："
          f"\n  {out / sorted(groups)[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
