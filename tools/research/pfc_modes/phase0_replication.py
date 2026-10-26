"""Phase 0 -- reproduce the S28 diagnosis on S20, independently.

    D:\\python.exe tools/research/pfc_modes/phase0_replication.py [--subject S20]

No parameter is tuned to make S20 agree with S28. The registration is checked
before any number is computed, and the run stops rather than reporting figures
from a mask that landed in the wrong place -- an ROI that is silently wrong
produces a study that is silently wrong.

Outputs under results/hybrid_pfc_residual_modes/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (           # noqa: E402
    SH8, SH9, SH5, IMAX, LAMBDA_REST, OUT_DIR, EXCLUDED_RADIAN, Progress,
    basis_with_a00, cohort, coil_maps_for, independent_energy, load_subject,
    protected_hashes, sh_bounds_production, sh_matrix, triplane_png,
    write_json)
from core.adaptive_joint import solve_adaptive_joint     # noqa: E402

UNIFORM = {"base": 1.0, "alpha": 0.0, "top_fraction": 0.0}


def qa_registration(s, P):
    """Everything that must be true before the ROI may be used."""
    qa = {
        "name": s["name"],
        "b0_shape": list(s["shape"]),
        "b0_affine": np.asarray(s["affine"]).round(6).tolist(),
        "brain_voxels": int(s["brain"].sum()),
        "brain_same_grid": s["brain_same_grid"],
        "pfc_path": None if s["pfc_path"] is None else str(s["pfc_path"]),
        "pfc_same_grid": s["pfc_same_grid"],
    }
    if s["pfc"] is None:
        qa["ok"] = False
        qa["reason"] = "no PFC mask found for this subject"
        return qa

    pfc, brain = s["pfc"], s["brain"]
    n_pfc = int(pfc.sum())
    qa["pfc_voxels"] = n_pfc
    qa["pfc_fraction_of_brain_pct"] = round(100.0 * n_pfc / max(int(brain.sum()), 1), 3)

    idx = np.argwhere(np.ones_like(brain, bool))
    import nibabel as nib
    W = nib.affines.apply_affine(s["affine"], idx).reshape((*brain.shape, 3))
    wb, wp = W[brain], W[pfc] if n_pfc else np.zeros((0, 3))
    qa["brain_centroid_mm"] = np.round(wb.mean(axis=0), 1).tolist()
    qa["brain_extent_mm"] = np.round(wb.max(axis=0) - wb.min(axis=0), 1).tolist()
    if n_pfc:
        qa["pfc_centroid_mm"] = np.round(wp.mean(axis=0), 1).tolist()
        qa["pfc_extent_mm"] = np.round(wp.max(axis=0) - wp.min(axis=0), 1).tolist()
        # RAS: +y anterior. The PFC must sit anterior to the brain centroid;
        # this is the check that catches a left-right or flipped registration
        # that still produces a plausible voxel count.
        qa["pfc_anterior_offset_mm"] = round(
            float(wp[:, 1].mean() - wb[:, 1].mean()), 1)

    reasons = []
    if n_pfc == 0:
        reasons.append("PFC mask does not intersect the brain mask")
    else:
        frac = qa["pfc_fraction_of_brain_pct"]
        if not (1.0 <= frac <= 25.0):
            reasons.append(f"PFC is {frac:.1f}% of the brain, outside 1-25%")
        if qa["pfc_anterior_offset_mm"] <= 5.0:
            reasons.append(
                f"PFC centroid is only {qa['pfc_anterior_offset_mm']:+.1f} mm "
                "anterior of the brain centroid -- expected clearly positive")
    qa["ok"] = not reasons
    qa["reason"] = "; ".join(reasons) if reasons else "checks passed"
    return qa


def collinearity_table(s, basis, maps, P):
    """Independent energy of the array vs each harmonic set, by region size."""
    from scipy.ndimage import binary_dilation

    brain, pfc = s["brain"], s["pfc"]
    regions = [("PFC", pfc)]
    for it in (3, 6, 12, 24):
        regions.append((f"PFC_dilated_x{it}",
                        binary_dilation(pfc, iterations=it) & brain))
    regions.append(("whole_brain", brain))

    rows = []
    for tag, reg in regions:
        n = int(reg.sum())
        C = maps.reshape(-1, maps.shape[-1])[reg.ravel(), :]
        row = {"region": tag, "voxels": n}
        for terms, label in ((SH8, "vs_SH8"), (SH9, "vs_SH9_withA00"),
                             (SH5, "vs_SH5_2ndorder")):
            row[label + "_pct"] = round(
                independent_energy(C, sh_matrix(basis, terms, reg)), 4)
        # How much of the SH8 residual is itself a uniform offset. Without this
        # split, "independent of the harmonics" reads as spatial structure when
        # it can be almost entirely DC -- which A00 would absorb, except that
        # A00 is not a variable the console accepts.
        A8 = sh_matrix(basis, SH8, reg)
        Q8, _ = np.linalg.qr(A8)
        R8 = C - Q8 @ (Q8.T @ C)
        denom = float(np.sum(C ** 2))
        dc = R8.mean(axis=0)
        row["vs_SH8_dc_pp"] = round(
            100.0 * float(np.sum(dc ** 2)) * R8.shape[0] / denom, 4)
        row["vs_SH8_nondc_pct"] = round(row["vs_SH8_pct"] - row["vs_SH8_dc_pp"], 4)
        rows.append(row)
        P.log(f"    {tag:20s} {n:8,}  SH8 {row['vs_SH8_pct']:7.3f}% "
              f"(DC {row['vs_SH8_dc_pp']:6.3f} + 结构 {row['vs_SH8_nondc_pct']:6.3f})  "
              f"SH9 {row['vs_SH9_withA00_pct']:7.3f}%  "
              f"SH5 {row['vs_SH5_2ndorder_pct']:7.3f}%")
    return rows


def spectrum(s, maps):
    """Normalised singular values of the array over the PFC, and effective rank."""
    C = maps.reshape(-1, maps.shape[-1])[s["pfc"].ravel(), :]
    sv = np.linalg.svd(C, compute_uv=False)
    sv_n = sv / sv[0]
    p = sv ** 2 / np.sum(sv ** 2)
    p = p[p > 0]
    return {
        "singular_values_normalised": np.round(sv_n, 5).tolist(),
        # Participation ratio and entropy rank: two standard readings, reported
        # together because they disagree when the spectrum has a long tail.
        "effective_rank_participation": round(
            float(np.sum(sv ** 2) ** 2 / np.sum(sv ** 4)), 3),
        "effective_rank_entropy": round(float(np.exp(-np.sum(p * np.log(p)))), 3),
        "n_above_1pct_of_first": int(np.sum(sv_n >= 0.01)),
    }


def _metrics(res, s):
    b, p = s["brain"], s["pfc"]
    rest = b & ~p
    out = {}
    for tag, m in (("pfc", p), ("rest", rest), ("whole", b)):
        v = res[m]
        out[f"{tag}_rmse"] = round(float(np.sqrt(np.mean(v ** 2))), 2)
        out[f"{tag}_std"] = round(float(np.std(v)), 2)
        out[f"{tag}_mean"] = round(float(np.mean(v)), 2)
    return out


def shim_configs(s, basis, maps, P):
    """no shim / SH only / coil only / joint whole-brain / joint region-weighted."""
    brain, pfc = s["brain"], s["pfc"]
    bounds = sh_bounds_production(SH8)
    basis_list = [basis[n] for n in SH8]
    zero = [np.zeros_like(s["b0"])]

    def run(use_sh, use_coil, weighted):
        kw = dict(imax=IMAX, lambda_current=0.0, lambda_sh=0.0,
                  lambda_pfc_low=0.0, centre=False)
        if weighted:
            kw.update(pfc_mask=pfc, lambda_rest=LAMBDA_REST)
        else:
            kw["weight_kwargs"] = UNIFORM
        bl = basis_list if use_sh else zero
        kw["sh_bounds"] = bounds if use_sh else [(0.0, 0.0)]
        return solve_adaptive_joint(s["b0"], brain, bl,
                                    maps if use_coil else None, **kw)

    rows = []
    base = {"config": "no_shim", "solver": "n/a"}
    base.update(_metrics(s["b0"], s))
    rows.append(base)

    for tag, sh, co, wt in (("SH_only", True, False, False),
                            ("coil_only", False, True, False),
                            ("SH+coil_wholebrain", True, True, False),
                            ("SH+coil_regionweighted", True, True, True)):
        r = run(sh, co, wt)
        row = {"config": tag}
        row.update(_metrics(r["residual"], s))
        cur = np.asarray(r["coil_currents"], dtype=float)
        row["current_rms"] = round(float(np.sqrt(np.mean(cur ** 2))), 3) if cur.size else 0.0
        row["current_peak"] = round(float(np.max(np.abs(cur))), 3) if cur.size else 0.0
        row["channels_at_limit"] = int(np.sum(np.abs(np.abs(cur) - IMAX) < 1e-3)) if cur.size else 0
        row["centre_frequency_hz"] = round(float(r["centre_frequency_hz"]), 2)
        row["solver"] = "ok" if r["success"] else f"FAILED: {r['message'][:60]}"
        if sh:
            row["scanner_delta"] = {n: round(float(v), 2)
                                    for n, v in zip(SH8, r["scanner_sh_delta"])}
        rows.append(row)
        P.log(f"    {tag:24s} PFC rmse {row['pfc_rmse']:7.2f}  "
              f"std {row['pfc_std']:6.2f} | whole rmse {row['whole_rmse']:7.2f} | "
              f"|I|peak {row['current_peak']:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S20")
    args = ap.parse_args()

    P = Progress(f"Phase 0 — independent replication on {args.subject}")
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    hashes_before = protected_hashes()

    entries, folds = cohort()
    entry = next((c for c in entries if c[0] == args.subject), None)
    if entry is None:
        raise SystemExit(
            f"{args.subject} is not in the usable cohort "
            f"(44 scans minus {len(EXCLUDED_RADIAN)} radian-scale: "
            f"{', '.join(EXCLUDED_RADIAN)})")
    P.log(f"subject {args.subject}  fold={folds[args.subject]}  "
          f"cohort={len(entries)} scans "
          f"(44 - {len(EXCLUDED_RADIAN)} radian-scale)")

    s = load_subject(entry)
    P.log(f"B0 {s['shape']}  brain {int(s['brain'].sum()):,} voxels")

    P.log("registration QA")
    qa = qa_registration(s, P)
    qa["fold"] = folds[args.subject]
    for k in ("pfc_voxels", "pfc_fraction_of_brain_pct", "pfc_centroid_mm",
              "pfc_extent_mm", "pfc_anterior_offset_mm", "pfc_same_grid"):
        if k in qa:
            P.log(f"    {k:28s} {qa[k]}")
    P.log(f"    verdict: {'OK' if qa['ok'] else 'FAILED'} — {qa['reason']}")

    fig_dir = out / "figures"
    if s["pfc"] is not None:
        triplane_png(fig_dir / f"registration_QA_{args.subject}.png",
                     s["b0"], [(s["pfc"], "white"), (s["brain"], "black")],
                     title=f"{args.subject}  B0 (Hz) with PFC (white) and brain "
                           f"(black) — {qa['reason']}")
        P.log(f"    QA figure -> {fig_dir / f'registration_QA_{args.subject}.png'}")

    write_json(out / f"s20_replication_{args.subject}_qa.json", qa)

    if not qa["ok"]:
        P.log("STOPPING: registration is not trustworthy; no numerical results "
              "will be produced from this ROI.")
        write_json(out / "input_hashes.json",
                   {"protected_before": hashes_before,
                    "protected_after": protected_hashes()})
        return 2

    P.log("loading coil maps (resampling to this grid)")
    maps, ch_names = coil_maps_for(s["affine"], s["shape"])
    P.log(f"    {maps.shape[-1]} channels -> {maps.shape}")

    basis = basis_with_a00(s["affine"], s["shape"])

    P.log("collinearity: independent energy of the array vs the harmonics")
    coll = collinearity_table(s, basis, maps, P)

    P.log("singular spectrum of the array over the PFC")
    spec = spectrum(s, maps)
    P.log(f"    normalised sv: {np.round(spec['singular_values_normalised'][:6], 3)}")
    P.log(f"    effective rank (participation) {spec['effective_rank_participation']}"
          f"  (entropy) {spec['effective_rank_entropy']}")

    P.log("shim configurations")
    configs = shim_configs(s, basis, maps, P)

    by = {r["config"]: r for r in configs}
    marginal = {
        "coil_over_SH_pfc_rmse": round(
            by["SH_only"]["pfc_rmse"] - by["SH+coil_regionweighted"]["pfc_rmse"], 2),
        "wholebrain_weighting_pfc_penalty_rmse": round(
            by["SH+coil_wholebrain"]["pfc_rmse"] - by["SH_only"]["pfc_rmse"], 2),
        "regionweighting_pfc_gain_rmse": round(
            by["SH+coil_wholebrain"]["pfc_rmse"]
            - by["SH+coil_regionweighted"]["pfc_rmse"], 2),
        "regionweighting_whole_cost_rmse": round(
            by["SH+coil_regionweighted"]["whole_rmse"]
            - by["SH+coil_wholebrain"]["whole_rmse"], 2),
    }
    P.log("marginal effects (Hz, RMSE):")
    for k, v in marginal.items():
        P.log(f"    {k:42s} {v:+.2f}")

    import csv
    with open(out / f"s20_replication_{args.subject}_collinearity.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(coll[0]))
        w.writeheader(); w.writerows(coll)
    flat = [{k: (v if not isinstance(v, dict) else "") for k, v in r.items()}
            for r in configs]
    keys = sorted({k for r in flat for k in r})
    with open(out / f"s20_replication_{args.subject}_configs.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(flat)

    hashes_after = protected_hashes()
    changed = [k for k in hashes_before
               if hashes_before[k] != hashes_after.get(k)]
    write_json(out / "input_hashes.json", {
        "protected_before": hashes_before, "protected_after": hashes_after,
        "protected_changed": changed,
        "subject_inputs": {
            "b0": str(s["hz_path"]), "brain": str(s["brain_path"]),
            "pfc": str(s["pfc_path"]), "coil_dir": str(_ROOT / "coils" / "pfc_12ch"),
        }})
    if changed:
        P.log(f"⚠ PROTECTED FILES CHANGED: {changed}")

    write_json(out / f"s20_replication_{args.subject}.json", {
        "subject": args.subject, "fold": folds[args.subject],
        "cohort_size": len(entries), "excluded_radian": list(EXCLUDED_RADIAN),
        "fold_counts": {f: sum(1 for v in folds.values() if v == f)
                        for f in ("train", "validation", "test")},
        "qa": qa,
        "collinearity": coll, "spectrum": spec,
        "configs": configs, "marginal": marginal,
        "conventions": {
            "sh8": SH8, "sh9_diagnostic": SH9, "sh5": SH5,
            "imax_a": IMAX, "lambda_rest": LAMBDA_REST,
            "sh_bounds": sh_bounds_production(SH8),
            "objective": "plain least squares, no pre-solve demeaning",
            "primary_metric": "ROI RMSE, no centre-frequency re-lock",
        },
        "channels": ch_names,
    })
    P.log(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
