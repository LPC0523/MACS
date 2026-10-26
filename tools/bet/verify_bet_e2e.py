"""End-to-end check of the BET that a packaged MACS build will actually run.

Loads the bundled _cpp_bet + model out of dist/MACS/_internal (not the dev
tree), runs it on real scans, and scores the masks against HD-BET. Run this
after any rebuild or after swapping the model file.

    python verify_bet_e2e.py
    python verify_bet_e2e.py --dist path/to/dist/MACS
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

HERE = Path(__file__).resolve().parent
# tools/bet/ -> tools/ -> shimming_qt/ -> shimming_app_latest/
PROJECT = HERE.parent.parent
REPO = PROJECT.parent


def dice(a: np.ndarray, b: np.ndarray) -> float:
    den = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / den) if den else 1.0


def cases() -> list[tuple[str, Path, Path]]:
    """(name, magnitude, HD-BET reference mask) triples that exist on disk."""
    ref = REPO / "scratch_ref"
    out = []
    for name, mag, hd in [
        ("S20", REPO / "test_nii/S20_nii/b0map_tra_2nd_Echo2_2.nii.gz",
         ref / "s20_hdbet_bet.nii.gz"),
        ("S17", REPO / "test_nii/S17_earshim_nii/b0map_tra_PG_2_Echo2_2.nii.gz",
         ref / "s17_hdbet_bet.nii.gz"),
    ]:
        if mag.is_file() and hd.is_file():
            out.append((name, mag, hd))

    # Held-out students' test split, if the training dataset is present.
    ds = REPO / "cpp_bet/training/dataset"
    sp = ds / "splits.json"
    if sp.is_file():
        for c in json.loads(sp.read_text())["test"]:
            img, lab = ds / "images" / f"{c}.nii.gz", ds / "labels" / f"{c}.nii.gz"
            if img.is_file() and lab.is_file():
                out.append((c[:34], img, lab))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=str(PROJECT / "dist" / "MACS"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-dice", type=float, default=0.95)
    args = ap.parse_args()

    internal = Path(args.dist) / "_internal"
    if not internal.is_dir():
        print(f"[e2e] no packaged build at {internal}")
        return 1

    sys.path.insert(0, str(internal))
    import os
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(internal))
    import _cpp_bet

    models = sorted((internal / "model").glob("*.onnx"))
    if len(models) != 1:
        print(f"[e2e] expected exactly one bundled model, found "
              f"{[m.name for m in models]}")
        return 1
    model = models[0]

    be = _cpp_bet.BrainExtractor()
    if not be.initialize(str(model), args.threads):
        print(f"[e2e] initialize failed: {be.last_error()}")
        return 1
    be.set_mode("quality")

    print(f"[e2e] model    : {model.name} ({model.stat().st_size/1e6:.2f} MB)")
    print(f"[e2e] geometry : {be.geometry_info()}")
    print()

    tmp = PROJECT / "_e2e_tmp"
    tmp.mkdir(exist_ok=True)
    rows, failures = [], []

    print(f"{'case':36s} {'Dice':>8s} {'time':>7s} {'pred L':>7s} {'ref L':>7s}")
    print("-" * 70)
    for name, mag, ref_mask in cases():
        out = tmp / f"{name}.nii.gz"
        t0 = time.time()
        rc = be.extract(str(mag), str(out))
        wall = time.time() - t0
        if rc != 0:
            print(f"{name:36s}  FAILED rc={rc} {be.last_error()}")
            failures.append(name)
            continue
        got = np.asanyarray(nib.load(str(out)).dataobj) > 0.5
        ref = np.asanyarray(nib.load(str(ref_mask)).dataobj) > 0.5
        zoom = float(np.prod(nib.load(str(mag)).header.get_zooms()[:3]))
        d = dice(got, ref)
        rows.append(d)
        flag = "" if d >= args.min_dice else "   <-- BELOW THRESHOLD"
        print(f"{name:36s} {d:8.4f} {wall:6.2f}s {got.sum()*zoom/1e6:7.2f} "
              f"{ref.sum()*zoom/1e6:7.2f}{flag}")
        if d < args.min_dice:
            failures.append(name)
        out.unlink(missing_ok=True)

    tmp.rmdir()
    if not rows:
        print("[e2e] no cases ran")
        return 1

    arr = np.array(rows)
    print("-" * 70)
    print(f"{'mean':36s} {arr.mean():8.4f}")
    print(f"{'worst':36s} {arr.min():8.4f}")
    print()
    if failures:
        print(f"[e2e] FAILED — {len(failures)} case(s) below "
              f"Dice {args.min_dice}: {failures}")
        return 1
    print(f"[e2e] PASSED — {len(rows)} cases, all >= Dice {args.min_dice}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
