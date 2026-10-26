"""Headless check of the one-shot pipeline — no GUI required.

Run after touching anything under core/: it exercises coil discovery, ROI
resolution, the full DICOM->currents path, and the input guard, and prints a
pass/fail summary.

    python tests/test_pipeline.py
    python tests/test_pipeline.py --dicom D:/data/S28 --roi pfc
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent
sys.path.insert(0, str(APP))

failures: list[str] = []


def check(cond: bool, what: str) -> None:
    print(("  [OK]   " if cond else "  [FAIL] ") + what)
    if not cond:
        failures.append(what)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", default=str(APP / "test_dic"))
    ap.add_argument("--roi", default="whole_brain")
    args = ap.parse_args()

    print("1. coil registry")
    from core.coil_registry import list_coils, get_default, discover_report
    coils = list_coils()
    check(len(coils) > 0, f"{len(coils)} usable coil(s): "
                          f"{[c.name for c in coils]}")
    d = get_default()
    check(d is not None, f"default coil = {d}")
    if d:
        check(len(d.channel_dirs()) == d.channel_count,
              f"{d.channel_count} channel folders on disk")
    if not coils:
        print(discover_report())

    print("2. ROI registry")
    from core.roi_registry import list_rois
    rois = list_rois()
    check(len(rois) >= 3, f"{[r.key for r in rois]}")

    print("3. input guard")
    from core.pipeline import looks_like_dicom_dir
    ok, _ = looks_like_dicom_dir(args.dicom)
    check(ok, f"accepts a DICOM tree: {args.dicom}")
    bad_ok, why = looks_like_dicom_dir(str(APP / "atlases"))
    check(not bad_ok, "rejects a NIfTI folder")
    check(bool(why) and "DICOM" in why, f"and says why: {why[:46]}...")
    missing_ok, _ = looks_like_dicom_dir(str(APP / "definitely_not_here"))
    check(not missing_ok, "rejects a missing path")

    if not ok:
        print("\nno DICOM input available, stopping before the full run")
        return 1 if failures else 0

    print("4. full pipeline")
    from core.pipeline import run_pipeline, PipelineError
    out = Path(tempfile.mkdtemp(prefix="macs_pipe_"))
    try:
        r = run_pipeline(args.dicom, output_dir=out, roi=args.roi,
                         progress_cb=lambda m: None)
        check(r.currents is not None and len(r.currents) > 0,
              f"{len(r.currents)} currents, "
              f"max |I| = {max(abs(r.currents)):.2f} A")
        check(r.roi_mask.any(), f"ROI {r.roi_name}: {int(r.roi_mask.sum()):,} voxels")
        s = r.stats
        check(s["roi_rmse_after"] < s["roi_rmse_before"],
              f"ROI RMSE {s['roi_rmse_before']:.1f} -> {s['roi_rmse_after']:.1f} Hz")
        std = s["roi_std_before"]
        check(5 < std < 300,
              f"input field std {std:.1f} Hz is physically plausible")
        check(r.total_seconds < 120,
              f"took {r.total_seconds:.1f}s "
              f"({', '.join(f'{k} {v:.1f}s' for k, v in r.seconds.items())})")
    except PipelineError as e:
        check(False, f"pipeline raised: {e.message} [{e.technical}]")
    finally:
        shutil.rmtree(out, ignore_errors=True)

    print()
    if failures:
        print(f"FAILED — {len(failures)} check(s): {failures}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
