"""Phase A -- is there a trustworthy native -> MNI forward transform?

    D:\\python.exe tools/research/pfc_modes/phaseA_transform_audit.py

`mni2native.mat` is a 4x4 written by `core/fsl_replace`, whose `_apply_affine`
uses it as

    vox_M = inv(src_affine) @ world_M @ ref_affine

to pull `src` onto `ref`'s grid. With src=MNI and ref=native that makes the
stored matrix map **native world coordinates to MNI world coordinates** -- the
opposite of what its filename suggests. Getting that backwards would silently
warp every field map into nonsense, so the direction is established here by
end-to-end test rather than by reading the code.

The test: carry the MNI PFC atlas into each subject's native grid using the
matrix in each direction, and score both against the `*_pfc_mask.nii.gz` that
ANTs actually produced. The direction that reproduces it is the direction.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (            # noqa: E402
    OUT_DIR, Progress, cohort, write_json)
from core.paths import find_atlases                       # noqa: E402

#: Dice below this against the stored ANTs mask means the transform cannot be
#: used to carry fields between spaces for this subject.
DICE_MIN = 0.80


def apply_world(src_data, src_affine, ref_shape, ref_affine, world_M, order=0):
    """Resample `src` onto `ref`'s grid, given ref_world -> src_world = world_M."""
    from scipy.ndimage import affine_transform

    vox_M = np.linalg.inv(src_affine) @ world_M @ ref_affine
    return affine_transform(np.asarray(src_data, dtype=np.float32),
                            vox_M[:3, :3], offset=vox_M[:3, 3],
                            output_shape=tuple(ref_shape), order=order,
                            mode="constant", cval=0.0)


def dice(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def main():
    P = Progress("Phase A — native -> MNI transform audit")
    entries, folds = cohort()
    mni_path, pfc_path = find_atlases()
    if not mni_path or not pfc_path:
        raise SystemExit("atlases not found")
    P.log(f"atlas PFC: {pfc_path}")

    pfc_img = nib.load(str(pfc_path))
    pfc_mni = np.asarray(pfc_img.get_fdata()) > 0.5
    P.log(f"MNI PFC {pfc_img.shape}, {int(pfc_mni.sum()):,} voxels")

    rows = []
    for name, hz, brain_p, pfc_p in entries:
        sdir = Path(hz).parent
        mat = sdir / "mni2native.mat"
        row = {"subject": name, "fold": folds[name],
               "has_matrix": mat.is_file()}
        if not mat.is_file():
            rows.append(row)
            P.log(f"  {name:12s} no mni2native.mat")
            continue

        M = np.loadtxt(str(mat))
        ref = nib.load(str(hz))
        truth = np.asarray(nib.load(str(pfc_p)).get_fdata()) > 0.5
        if truth.shape != ref.shape:
            from core.data_context import resample_to_target
            t_img = nib.load(str(pfc_p))
            truth = resample_to_target(truth.astype(np.float32), t_img.affine,
                                       ref.affine, ref.shape, order=0) > 0.5

        # Direction 1: stored matrix is native_world -> MNI_world (pull-back).
        d1 = apply_world(pfc_mni.astype(np.float32), pfc_img.affine,
                         ref.shape, ref.affine, M) > 0.5
        # Direction 2: the filename's reading, MNI_world -> native_world.
        d2 = apply_world(pfc_mni.astype(np.float32), pfc_img.affine,
                         ref.shape, ref.affine, np.linalg.inv(M)) > 0.5

        row.update({
            "dice_as_stored": round(dice(d1, truth), 4),
            "dice_inverted": round(dice(d2, truth), 4),
            "truth_voxels": int(truth.sum()),
            "as_stored_voxels": int(d1.sum()),
            "inverted_voxels": int(d2.sum()),
            "det": round(float(np.linalg.det(M[:3, :3])), 5),
        })
        # Round trip through the matrix and back, in world millimetres.
        pts = np.array([[0, 0, 0], [50, 0, 0], [0, 50, 0], [0, 0, 50],
                        [-40, 60, 20]], dtype=float)
        hom = np.c_[pts, np.ones(len(pts))].T          # (4, n)
        back = (np.linalg.inv(M) @ (M @ hom)).T        # (n, 4)
        row["roundtrip_max_mm"] = round(
            float(np.max(np.abs(back[:, :3] - pts))), 9)
        rows.append(row)
        P.log(f"  {name:12s} dice as-stored {row['dice_as_stored']:.3f}  "
              f"inverted {row['dice_inverted']:.3f}  det {row['det']:+.3f}")

    have = [r for r in rows if r.get("dice_as_stored") is not None]
    if not have:
        raise SystemExit("no usable matrices")
    ds = np.array([r["dice_as_stored"] for r in have])
    di = np.array([r["dice_inverted"] for r in have])
    P.log("")
    P.log(f"as stored : mean dice {ds.mean():.3f}  min {ds.min():.3f}")
    P.log(f"inverted  : mean dice {di.mean():.3f}  min {di.min():.3f}")

    stored_wins = int((ds > di).sum())
    P.log(f"as-stored better on {stored_wins}/{len(have)} scans")
    direction = ("native_world_to_mni_world" if ds.mean() > di.mean()
                 else "mni_world_to_native_world")
    best = ds if ds.mean() > di.mean() else di
    P.log(f"=> stored matrix maps {direction}")

    usable = [have[i]["subject"] for i in range(len(have)) if best[i] >= DICE_MIN]
    failed = [have[i]["subject"] for i in range(len(have)) if best[i] < DICE_MIN]
    missing = [r["subject"] for r in rows if not r.get("has_matrix")]
    P.log(f"usable (dice >= {DICE_MIN}): {len(usable)}  "
          f"failed: {len(failed)} {failed}  missing matrix: {len(missing)} {missing}")

    verdict = {
        "direction_of_stored_matrix": direction,
        "dice_mean_as_stored": round(float(ds.mean()), 4),
        "dice_mean_inverted": round(float(di.mean()), 4),
        "dice_min_best_direction": round(float(best.min()), 4),
        "dice_threshold": DICE_MIN,
        "n_usable": len(usable), "usable": usable,
        "n_failed": len(failed), "failed": failed,
        "n_missing_matrix": len(missing), "missing_matrix": missing,
        "max_roundtrip_mm": round(
            float(max(r.get("roundtrip_max_mm", 0.0) for r in have)), 12),
        "viable": len(usable) >= 20,
    }
    P.log("VERDICT: " + ("forward transform usable"
                         if verdict["viable"] else "NOT enough usable transforms"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT_DIR / "transform_audit.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    write_json(OUT_DIR / "transform_audit.json",
               {"verdict": verdict, "per_subject": rows})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
