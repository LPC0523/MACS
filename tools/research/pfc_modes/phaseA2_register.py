"""Phase A2 -- build a native -> MNI transform that can be verified.

    D:\\python.exe tools/research/pfc_modes/phaseA2_register.py [--limit N]

Phase A showed the cached `mni2native.mat` cannot be used: neither reading of
its direction reproduces the `*_pfc_mask.nii.gz` that ships beside each subject
(best mean Dice 0.18). The matrices and the masks came from different
registration paths, so the stored 4x4 does not describe the mapping the masks
were made with.

This registers each subject afresh -- skull-stripped magnitude to the
skull-stripped MNI brain, ANTs affine, the same settings production uses -- and
keeps the transform. Direction is native -> MNI, which is what carrying a field
map into the template requires.

Every subject is checked before its transform is accepted: the MNI PFC atlas is
pulled back to native through the inverse and scored against the mask already on
disk. A transform that cannot reproduce a mask that was itself made by ANTs is
not one to warp field maps with.

Writes only under results/; subject directories are never modified.
"""

from __future__ import annotations

import argparse
import csv
import shutil
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
from core.data_context import resample_to_target          # noqa: E402

TX_DIR = OUT_DIR / "transforms"

# Both thresholds exist to catch a *failed* registration, not to rank good ones.
# What failure looks like here is known: the cached `mni2native.mat` reproduces
# the shipped PFC mask at Dice 0.08-0.18. What success looks like is also known
# now -- re-registering all 40 scans gives brain Dice 0.845-0.936 (mean 0.904,
# no outliers, no second mode) and PFC agreement 0.470-0.941. There is a wide
# empty gap between the two, and both thresholds are placed inside it.
#
# An earlier draft used 0.90 for the brain gate. That is not a failure boundary
# -- it falls on the mean of a smooth distribution and would have discarded 16
# scans for being ordinarily imperfect rather than wrong. Affine registration to
# a template does not exceed ~0.95 on brain overlap by construction, because an
# affine cannot match individual cortical shape.

#: Primary gate: the subject's brain, pushed into MNI, against the template
#: brain. 0.80 is below anything a working affine produces and far above a
#: broken one.
BRAIN_DICE_MIN = 0.80

#: Secondary gate. Agreement between this registration and the independent one
#: that produced the shipped `*_pfc_mask.nii.gz` -- two uncertain fits, not an
#: error against truth, so it is set only to exclude gross disagreement.
PFC_DICE_FLOOR = 0.35

#: Production's own registration settings (core/roi_registry._register_pfc), so
#: a transform accepted here is one the shipping pipeline would also produce.
REG_KW = dict(type_of_transform="Affine", aff_metric="mattes", aff_sampling=32,
              aff_iterations=(2100, 1200, 1200, 100),
              aff_shrink_factors=(6, 4, 2, 1),
              aff_smoothing_sigmas=(3, 2, 1, 0), verbose=False)


def dice(a, b):
    a = np.asarray(a, bool); b = np.asarray(b, bool)
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()

    P = Progress("Phase A2 — re-register native -> MNI, and verify each one")
    import ants
    from core import ants_io

    mni_path, pfc_path = find_atlases()
    entries, folds = cohort()
    want = [w.strip() for w in args.subjects.split(",") if w.strip()]
    if want:
        entries = [e for e in entries if e[0] in want]
    if args.limit:
        entries = entries[:args.limit]
    P.log(f"{len(entries)} scans; MNI brain {mni_path}")

    TX_DIR.mkdir(parents=True, exist_ok=True)
    fixed = ants_io.image_read(mni_path)
    pfc_atlas = ants_io.image_read(pfc_path)

    rows = []
    for name, hz, brain_p, pfc_p in entries:
        sdir = Path(hz).parent
        bet = sorted(sdir.glob("*_bet.nii.gz"))
        row = {"subject": name, "fold": folds[name]}
        if not bet:
            row["status"] = "no *_bet.nii.gz"
            rows.append(row); P.log(f"  {name:12s} {row['status']}"); continue

        out_dir = TX_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        fwd_kept = out_dir / "native_to_mni_0GenericAffine.mat"

        try:
            moving = ants_io.image_read(bet[0])
            if fwd_kept.is_file():
                P.log(f"  {name:12s} using cached transform")
                fwd = [str(fwd_kept)]
            else:
                reg = ants.registration(fixed=fixed, moving=moving, **REG_KW)
                # Affine registration returns a single .mat; keep our own copy
                # because ANTs writes into a temporary directory.
                src = Path(reg["fwdtransforms"][0])
                shutil.copyfile(src, fwd_kept)
                fwd = [str(fwd_kept)]

            # Verification: MNI PFC atlas -> native, through the inverse of the
            # transform we just built, scored against the mask on disk.
            back = ants.apply_transforms(
                fixed=moving, moving=pfc_atlas, transformlist=fwd,
                whichtoinvert=[True], interpolator="nearestNeighbor")
            back_np = back.numpy() > 0.5

            truth_img = nib.load(str(pfc_p))
            truth = np.asarray(truth_img.get_fdata()) > 0.5
            bet_img = nib.load(str(bet[0]))
            if truth.shape != bet_img.shape:
                truth = resample_to_target(truth.astype(np.float32),
                                           truth_img.affine, bet_img.affine,
                                           bet_img.shape, order=0) > 0.5
            d = dice(back_np, truth)

            # Primary gate: push the subject's own brain forward into MNI and
            # see whether it lands on the template's brain.
            fwd_brain = ants.apply_transforms(
                fixed=fixed, moving=moving, transformlist=fwd,
                interpolator="linear")
            brain_d = dice(fwd_brain.numpy() > 0, fixed.numpy() > 0)

            row.update({"status": "ok",
                        "brain_dice_in_mni": round(brain_d, 4),
                        "dice_vs_shipped_mask": round(d, 4),
                        "warped_voxels": int(back_np.sum()),
                        "shipped_voxels": int(truth.sum()),
                        "bet": bet[0].name,
                        "transform": str(fwd_kept.relative_to(OUT_DIR))})
            row["accepted"] = bool(brain_d >= BRAIN_DICE_MIN
                                   and d >= PFC_DICE_FLOOR)
            P.log(f"  {name:12s} brain dice {brain_d:.3f}  "
                  f"PFC agreement {d:.3f}  "
                  f"{'accepted' if row['accepted'] else 'REJECTED'}")
        except Exception as e:
            row["status"] = f"{type(e).__name__}: {e}"
            row["accepted"] = False
            P.log(f"  {name:12s} FAILED {row['status'][:70]}")
        rows.append(row)

    good = [r for r in rows if r.get("accepted")]
    ds = np.array([r["dice_vs_shipped_mask"] for r in rows
                   if "dice_vs_shipped_mask" in r])
    bs = np.array([r["brain_dice_in_mni"] for r in rows
                   if "brain_dice_in_mni" in r])
    P.log("")
    if bs.size:
        P.log(f"brain dice in MNI (gate): mean {bs.mean():.3f}  "
              f"min {bs.min():.3f}  max {bs.max():.3f}")
    if ds.size:
        P.log(f"PFC agreement with shipped mask: mean {ds.mean():.3f}  "
              f"min {ds.min():.3f}  max {ds.max():.3f}")
    P.log(f"accepted {len(good)}/{len(rows)} "
          f"(brain dice >= {BRAIN_DICE_MIN}, PFC agreement >= {PFC_DICE_FLOOR})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(OUT_DIR / "registration_qa.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    write_json(OUT_DIR / "registration_qa.json", {
        "n": len(rows), "n_accepted": len(good),
        "brain_dice_gate": BRAIN_DICE_MIN, "pfc_agreement_floor": PFC_DICE_FLOOR,
        "brain_dice_mean": round(float(bs.mean()), 4) if bs.size else None,
        "brain_dice_min": round(float(bs.min()), 4) if bs.size else None,
        "pfc_agreement_mean": round(float(ds.mean()), 4) if ds.size else None,
        "pfc_agreement_min": round(float(ds.min()), 4) if ds.size else None,
        "reg_settings": {k: str(v) for k, v in REG_KW.items()},
        "accepted": [r["subject"] for r in good],
        "rejected": [r["subject"] for r in rows if not r.get("accepted")],
        "per_subject": rows})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
