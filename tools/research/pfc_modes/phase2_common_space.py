"""Can the cohort share one space, and which one?

The PCA in this study looks for residual modes that a *fixed piece of hardware*
can correct. The array does not move with the anatomy: it sits in the bore, and
what varies between scans is where the head is relative to it. So the candidate
common space is the scanner's own world frame, not an anatomical template --
in MNI space the coil field itself would differ per subject, which is exactly
the quantity the study needs to hold fixed.

This script decides whether that frame is usable: do the scans overlap in world
coordinates, and do their PFC masks land close enough together for one shared
weight w_P to mean anything. If they do not, the brief says to skip the PCA and
run the hardware Slepian route alone -- so this reports a verdict, not a fix.

    D:\\python.exe tools/research/pfc_modes/phase2_common_space.py
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
    OUT_DIR, Progress, cohort, load_subject, write_json)


def world_stats(mask, affine):
    idx = np.argwhere(mask)
    if idx.size == 0:
        return None
    w = nib.affines.apply_affine(affine, idx)
    return {"centroid": w.mean(axis=0), "lo": w.min(axis=0), "hi": w.max(axis=0),
            "n": int(idx.shape[0])}


def main():
    P = Progress("Phase 2 — is there a usable common space?")
    entries, folds = cohort()
    P.log(f"cohort {len(entries)} scans")

    rows, brains, pfcs = [], [], []
    for e in entries:
        s = load_subject(e)
        wb = world_stats(s["brain"], s["affine"])
        wp = world_stats(s["pfc"], s["affine"]) if s["pfc"] is not None else None
        if wb is None or wp is None:
            P.log(f"  {s['name']}: missing mask, skipped")
            continue
        brains.append(wb); pfcs.append(wp)
        rows.append({
            "subject": s["name"], "fold": folds[s["name"]],
            "shape": "x".join(str(v) for v in s["shape"]),
            "voxel_mm": "x".join(
                f"{v:.2f}" for v in np.abs(np.diag(s["affine"])[:3])),
            "brain_voxels": wb["n"], "pfc_voxels": wp["n"],
            **{f"brain_centroid_{a}": round(float(wb["centroid"][i]), 1)
               for i, a in enumerate("xyz")},
            **{f"pfc_centroid_{a}": round(float(wp["centroid"][i]), 1)
               for i, a in enumerate("xyz")},
            **{f"brain_lo_{a}": round(float(wb["lo"][i]), 1) for i, a in enumerate("xyz")},
            **{f"brain_hi_{a}": round(float(wb["hi"][i]), 1) for i, a in enumerate("xyz")},
        })

    P.log(f"usable scans: {len(rows)}")

    bc = np.array([[r[f"brain_centroid_{a}"] for a in "xyz"] for r in rows])
    pc = np.array([[r[f"pfc_centroid_{a}"] for a in "xyz"] for r in rows])
    lo = np.array([[r[f"brain_lo_{a}"] for a in "xyz"] for r in rows])
    hi = np.array([[r[f"brain_hi_{a}"] for a in "xyz"] for r in rows])

    P.log("brain centroid spread in scanner world coordinates (mm)")
    for i, a in enumerate("xyz"):
        P.log(f"    {a}: mean {bc[:, i].mean():+7.1f}  sd {bc[:, i].std():5.1f}  "
              f"range [{bc[:, i].min():+.0f}, {bc[:, i].max():+.0f}]")
    P.log("PFC centroid spread (mm)")
    for i, a in enumerate("xyz"):
        P.log(f"    {a}: mean {pc[:, i].mean():+7.1f}  sd {pc[:, i].std():5.1f}  "
              f"range [{pc[:, i].min():+.0f}, {pc[:, i].max():+.0f}]")

    # The box every brain covers, versus the box any brain covers.
    inter_lo, inter_hi = lo.max(axis=0), hi.min(axis=0)
    union_lo, union_hi = lo.min(axis=0), hi.max(axis=0)
    inter_ok = bool(np.all(inter_hi > inter_lo))
    P.log(f"world bounding boxes: intersection "
          f"{np.round(inter_lo, 0)} .. {np.round(inter_hi, 0)}  "
          f"({'non-empty' if inter_ok else 'EMPTY'})")
    P.log(f"                      union        "
          f"{np.round(union_lo, 0)} .. {np.round(union_hi, 0)}")
    if inter_ok:
        P.log(f"  intersection extent {np.round(inter_hi - inter_lo, 0)} mm; "
              f"union extent {np.round(union_hi - union_lo, 0)} mm")

    voxel_sizes = sorted({r["voxel_mm"] for r in rows})
    shapes = sorted({r["shape"] for r in rows})
    P.log(f"distinct voxel sizes: {voxel_sizes}")
    P.log(f"distinct shapes     : {shapes}")

    # Centroid scatter is suggestive but not decisive. What settles it is
    # whether one shared w_P, defined once in the scanner frame, actually
    # covers each scan's PFC -- so resample every mask onto a common 2 mm grid
    # and measure the overlap directly.
    P.log("resampling every PFC mask onto a common 2 mm scanner-frame grid")
    step = 2.0
    g_lo = np.floor(union_lo / step) * step
    g_hi = np.ceil(union_hi / step) * step
    dims = np.maximum(((g_hi - g_lo) / step).astype(int) + 1, 1)
    common_affine = np.diag([step, step, step, 1.0])
    common_affine[:3, 3] = g_lo
    P.log(f"    grid {tuple(dims)} at {step} mm, origin {np.round(g_lo, 0)}")

    from core.data_context import resample_to_target

    stack = []
    for e in entries:
        s = load_subject(e)
        if s["pfc"] is None or not s["pfc"].any():
            continue
        m = resample_to_target(s["pfc"].astype(np.float32), s["affine"],
                               common_affine, tuple(dims), order=0) > 0.5
        stack.append((s["name"], m))
    P.log(f"    {len(stack)} masks on the common grid")

    M = np.stack([m for _, m in stack], axis=0)
    counts = M.sum(axis=0)
    n = M.shape[0]
    inter_vox = int((counts == n).sum())
    union_vox = int((counts > 0).sum())
    mean_size = float(M.reshape(n, -1).sum(axis=1).mean())

    # Consensus region: voxels in at least half the scans. This is the most
    # generous shared w_P anyone would actually build.
    consensus = counts >= (n + 1) // 2
    cov = np.array([float((m & consensus).sum()) / max(int(m.sum()), 1)
                    for _, m in stack])

    P.log(f"    all-{n} intersection {inter_vox:,} voxels "
          f"({100 * inter_vox / max(mean_size, 1):.1f}% of a typical PFC)")
    P.log(f"    union {union_vox:,} voxels "
          f"({union_vox / max(mean_size, 1):.1f}x a typical PFC)")
    P.log(f"    majority-consensus region {int(consensus.sum()):,} voxels")
    P.log(f"    per-scan coverage by consensus: mean {cov.mean():.2f}  "
          f"min {cov.min():.2f}  worst {stack[int(np.argmin(cov))][0]}")
    P.log(f"    scans with <50% coverage: "
          f"{[stack[i][0] for i in np.where(cov < 0.5)[0]]}")

    pfc_spread = float(np.max(pc.std(axis=0)))
    verdict = {
        "common_grid_mm": step,
        "common_grid_dims": [int(v) for v in dims],
        "pfc_all_intersection_voxels": inter_vox,
        "pfc_union_voxels": union_vox,
        "pfc_typical_voxels": round(mean_size, 1),
        "pfc_intersection_over_typical": round(inter_vox / max(mean_size, 1), 4),
        "consensus_voxels": int(consensus.sum()),
        "consensus_coverage_mean": round(float(cov.mean()), 4),
        "consensus_coverage_min": round(float(cov.min()), 4),
        "scans_below_half_coverage": [stack[i][0] for i in np.where(cov < 0.5)[0]],
        "n_scans": len(rows),
        "brain_centroid_sd_mm": np.round(bc.std(axis=0), 2).tolist(),
        "pfc_centroid_sd_mm": np.round(pc.std(axis=0), 2).tolist(),
        "pfc_centroid_range_mm": np.round(pc.max(axis=0) - pc.min(axis=0), 1).tolist(),
        "world_intersection_lo": np.round(inter_lo, 1).tolist(),
        "world_intersection_hi": np.round(inter_hi, 1).tolist(),
        "world_intersection_nonempty": inter_ok,
        "distinct_voxel_sizes": voxel_sizes,
        "distinct_shapes": shapes,
    }

    # A shared w_P is only meaningful if the PFC lands in roughly the same place
    # in the frame the coil is fixed in. 20 mm is the scale at which the array's
    # own field varies appreciably over the PFC.
    reasons = []
    if not inter_ok:
        reasons.append("scans do not share a common world bounding box")
    # The decisive test: a shared w_P must actually sit on each scan's PFC.
    # Thresholds fixed before looking: a consensus region has to cover most of
    # a typical PFC on average, and must not miss any single scan outright.
    if cov.mean() < 0.70:
        reasons.append(
            f"a majority-consensus PFC covers only {cov.mean():.0%} of a "
            "typical scan's PFC on average")
    if cov.min() < 0.40:
        reasons.append(
            f"worst scan ({stack[int(np.argmin(cov))][0]}) is only "
            f"{cov.min():.0%} covered by the consensus region")
    if inter_vox / max(mean_size, 1) < 0.10:
        reasons.append(
            f"the all-scan intersection is {inter_vox / max(mean_size, 1):.1%} "
            "of a typical PFC -- there is almost no tissue common to every scan")
    verdict["pca_in_scanner_frame_viable"] = not reasons
    verdict["reasons"] = reasons or ["scanner-frame PCA is viable"]

    P.log("VERDICT: " + ("scanner-frame PCA is viable"
                         if not reasons else "NOT viable — " + "; ".join(reasons)))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "common_space_audit.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_json(OUT_DIR / "common_space_verdict.json", verdict)
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
