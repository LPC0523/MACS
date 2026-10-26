"""Phase E-I -- carry the target modes back, test reachability, compare.

    D:\\python.exe tools/research/pfc_modes/phaseEI_reach_and_compare.py

E/F  each MNI target mode is warped into a validation scan's native space and
     fitted with the scan's **full 12 channels** of C_H, bounded at +-3 A. The
     PCA chooses *what to cancel*; it never restricts which currents may do it.

G    modes are ranked by three things at once -- cross-subject stability, PFC
     share, and bounded reachability -- because a mode that is stable and
     concentrated but unreachable is of no use, and so is a reachable one that
     no subject expresses.

H/I  three arms, all with f0 free (A00 is a deliverable now, not a nuisance):

       1  f0 + SH8
       2  f0 + SH8 + all 12 channels, region-weighted joint
       3  f0 + SH8 + currents fitted to the selected target modes

     Reported as the console would receive them, using C = S9 K + C_H:

       scanner_delta = theta - K_SH i
       f0_delta      = a00  - K_A00 i
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.optimize import minimize, LinearConstraint

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (            # noqa: E402
    SH8, SH9, IMAX, LAMBDA_REST, OUT_DIR, Progress, basis_with_a00, cohort,
    coil_maps_for, load_subject, sh_bounds_production, sh_matrix, write_json)
from core.hybrid_shim import GAMMA                        # noqa: E402
from tools.research.pfc_modes.phase1_slepian import TAPER_MM, taper_weight  # noqa: E402
from tools.research.pfc_modes.phaseBC_target_pca import project_sh9  # noqa: E402
from core.adaptive_joint import make_region_weights      # noqa: E402

TX_DIR = OUT_DIR / "transforms"
LAMBDA_LEAK = 1.0
MU_CURRENT = 0.0
N_TARGET_MODES = 6
SOLVER_OPTS = {"maxiter": 3000, "gtol": 1e-10, "xtol": 1e-12}


def bounded_fit(C_H, u, wP, wR, imax, lam_leak=LAMBDA_LEAK, mu=MU_CURRENT):
    """min ||C_H i - u||^2_wP + lam_leak ||C_H i||^2_wR + mu ||i||^2, |i| <= imax."""
    n = C_H.shape[1]
    H = C_H.T @ (wP[:, None] * C_H) + lam_leak * C_H.T @ (wR[:, None] * C_H)
    if mu:
        H = H + mu * np.eye(n)
    g = -(C_H.T @ (wP * u))
    c0 = 0.5 * float(np.sum(wP * u ** 2))
    H = H + H.T                              # d/di of 0.5 i'Hi with H symmetric
    H *= 0.5

    def fun(x):
        Hx = H @ x
        return c0 + float(g @ x) + 0.5 * float(x @ Hx), g + Hx

    res = minimize(fun, np.zeros(n), jac=True, hess=lambda _x: H,
                   method="trust-constr",
                   bounds=[(-imax, imax)] * n, options=SOLVER_OPTS)
    return res.x, res


def reach_fraction(C_H, u, w):
    """Unconstrained share of u that lies in C_H's span, in the w metric."""
    A = C_H * np.sqrt(w)[:, None]
    b = u * np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    fit = A @ coef
    den = float(b @ b)
    return (float(fit @ fit) / den) if den > 0 else float("nan")


def solve_arm(b, S9, C_eff, V, sh_bounds, weights, imax):
    """Joint solve over [a00, theta(8), coil coefficients]."""
    n_s, n_a = S9.shape[1], C_eff.shape[1]
    D = np.hstack([S9, C_eff]) if n_a else S9
    w = weights / np.sum(weights)
    H = D.T @ (D * w[:, None])
    g = D.T @ (w * b)
    c0 = 0.5 * float(np.sum(w * b ** 2))

    def fun(x):
        Hx = H @ x
        return c0 + float(g @ x) + 0.5 * float(x @ Hx), g + Hx

    # A00 is unbounded here: the console's f0 range is far wider than any shim
    # term and is not the binding constraint.
    lo = [-np.inf] + [bd[0] for bd in sh_bounds] + [-np.inf] * n_a
    hi = [np.inf] + [bd[1] for bd in sh_bounds] + [np.inf] * n_a
    cons = []
    if n_a:
        M = np.zeros((V.shape[0], n_s + n_a)); M[:, n_s:] = V
        cons.append(LinearConstraint(M, -imax, imax))
    res = minimize(fun, np.zeros(n_s + n_a), jac=True, hess=lambda _x: H,
                   method="trust-constr", bounds=list(zip(lo, hi)),
                   constraints=cons, options=SOLVER_OPTS)
    return res.x[:n_s], res.x[n_s:], res, float(fun(res.x)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="validation")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    P = Progress("Phase E-I — reachability of MNI target modes, and comparison")
    import ants
    from core import ants_io

    npz = np.load(OUT_DIR / "target_pca_modes.npz", allow_pickle=True)
    aff_m = npz["affine"]; shape_m = tuple(int(v) for v in npz["shape"])
    region = npz["region"]; wP_m = npz["wP"]
    # PCA modes come out of the SVD with unit norm, so on their own they carry
    # no physical amplitude. `bounded_reach` is not scale-invariant -- the +-3 A
    # limit is absolute -- so an unscaled mode is fitted with zero current and
    # would be reported as unreachable when it is only small. Each mode is
    # restored to the size it actually takes in the training cohort, the RMS of
    # its own coefficient across scans. The mean mode is already a field in Hz.
    coef = np.asarray(npz["coef_mean_removed"], dtype=float)
    scales = np.sqrt(np.mean(coef ** 2, axis=0))
    targets = [("mean", np.asarray(npz["mean_mode"], float))]
    for k in range(min(N_TARGET_MODES, npz["modes_mean_removed"].shape[0])):
        targets.append((f"pca{k}", npz["modes_mean_removed"][k] * scales[k]))
    P.log(f"{len(targets)} target modes from {int(region.sum()):,} MNI voxels")
    P.log("  physical amplitudes (weighted RMS over the MNI PFC, Hz): "
          + "  ".join(f"{t}={np.sqrt(np.mean(a ** 2)):.2f}" for t, a in targets))

    tmp = Path(tempfile.mkdtemp(prefix="pfcEI_"))
    entries, folds = cohort()
    sel = [e for e in entries if folds[e[0]] == args.fold]
    if args.limit:
        sel = sel[:args.limit]
    P.log(f"{len(sel)} scans in fold '{args.fold}'")

    # Write each target once as an MNI volume; warping is per subject.
    tgt_paths = []
    for tag, arr in targets:
        v = np.zeros(shape_m, np.float32); v[region] = arr
        p = tmp / f"target_{tag}.nii.gz"
        nib.save(nib.Nifti1Image(v, aff_m), str(p))
        tgt_paths.append((tag, p))

    reach_rows, cmp_rows = [], []
    for name, hz, brain_p, pfc_p in sel:
        tx = TX_DIR / name / "native_to_mni_0GenericAffine.mat"
        if not tx.is_file():
            P.log(f"  {name}: no transform, skipped"); continue
        s = load_subject((name, hz, brain_p, pfc_p))
        maps, _ = coil_maps_for(s["affine"], s["shape"])
        basis = basis_with_a00(s["affine"], s["shape"])
        brain, pfc = s["brain"], s["pfc"]
        b_H, C_H, K, S9m = project_sh9(s["b0"], maps, brain, basis)
        b = s["b0"][brain].astype(np.float64)

        wP_full = taper_weight(pfc, s["affine"], TAPER_MM)[brain]
        wR = (wP_full <= 0).astype(float)
        moving = ants_io.image_read(hz)

        P.log(f"  {name}  b_H {np.sqrt(np.mean(b_H ** 2)):.1f} Hz  "
              f"C_H {100 * np.sum(C_H ** 2) / np.sum(maps.reshape(-1, 12)[brain.ravel()] ** 2):.2f}%")

        u_native = {}
        for tag, p in tgt_paths:
            w = ants.apply_transforms(fixed=moving, moving=ants_io.image_read(p),
                                      transformlist=[str(tx)],
                                      whichtoinvert=[True], interpolator="linear")
            u = np.asarray(w.numpy(), np.float64)[brain]
            u_native[tag] = u
            if not np.any(wP_full > 0) or not np.any(np.abs(u) > 0):
                continue
            q = reach_fraction(C_H, u, wP_full)
            i_b, res = bounded_fit(C_H, u, wP_full, wR, IMAX)
            fit = C_H @ i_b
            num = float(np.sum(wP_full * (fit - u) ** 2))
            den = float(np.sum(wP_full * u ** 2))
            bounded_frac = 1.0 - num / den if den > 0 else float("nan")
            # How much of this scan's own residual the mode explains.
            expl = float(np.sum(wP_full * u * b_H)
                         / np.sqrt(max(np.sum(wP_full * u ** 2)
                                       * np.sum(wP_full * b_H ** 2), 1e-12)))
            reach_rows.append({
                "subject": name, "fold": folds[name], "mode": tag,
                "unconstrained_reach": round(q, 5),
                "bounded_reach": round(bounded_frac, 5),
                "pfc_share_of_mode": round(float(
                    np.sum(wP_full * u ** 2)
                    / max(np.sum(u ** 2), 1e-12)), 5),
                "align_with_own_bH": round(expl, 5),
                "current_rms": round(float(np.sqrt(np.mean(i_b ** 2))), 4),
                "current_peak": round(float(np.max(np.abs(i_b))), 4),
                "channels_at_limit": int(np.sum(np.abs(np.abs(i_b) - IMAX) < 1e-3)),
            })

        # ---- H/I: three arms -------------------------------------------
        S9 = sh_matrix(basis, SH9, brain)
        bounds8 = sh_bounds_production(SH8)
        weights = make_region_weights(pfc[brain], LAMBDA_REST, per_voxel=False)
        n_ch = maps.shape[-1]

        def report(tag, coefs, i_ch, obj):
            a00, theta = coefs[0], coefs[1:]
            corr = a00 * basis["A00"]
            for j, nm in enumerate(SH8):
                corr = corr + theta[j] * basis[nm]
            if i_ch is not None and np.size(i_ch):
                corr = corr + np.tensordot(maps, i_ch, axes=([3], [0]))
            resid = s["b0"] + corr
            cur = np.zeros(n_ch) if i_ch is None else np.asarray(i_ch, float)
            # High-order residual: what is left after every scanner harmonic,
            # i.e. the part only the array could ever have removed.
            rH = resid[brain] - S9 @ np.linalg.lstsq(S9, resid[brain],
                                                     rcond=None)[0]
            row = {"subject": name, "fold": folds[name], "arm": tag,
                   "objective": round(obj, 6)}
            rest = brain & ~pfc
            for t, m in (("pfc", pfc), ("rest", rest), ("whole", brain)):
                v = resid[m]
                row[f"{t}_rmse"] = round(float(np.sqrt(np.mean(v ** 2))), 3)
                row[f"{t}_std"] = round(float(np.std(v)), 3)
                row[f"{t}_mean"] = round(float(np.mean(v)), 3)
            row["pfc_highorder_rmse"] = round(float(np.sqrt(np.mean(
                rH[pfc[brain]] ** 2))), 3)
            row["current_rms"] = round(float(np.sqrt(np.mean(cur ** 2))), 4)
            row["current_peak"] = round(float(np.max(np.abs(cur))), 4)
            row["channels_at_limit"] = int(np.sum(np.abs(np.abs(cur) - IMAX) < 1e-3))
            # A00's basis column is the constant GAMMA, so its coefficient is
            # in uT like every other harmonic; the field it produces -- what
            # the console would be asked to shift f0 by -- is that times GAMMA.
            row["f0_delta_coef_ut"] = round(float(a00 - K[0] @ cur), 5)
            row["f0_delta_hz"] = round(float((a00 - K[0] @ cur) * GAMMA), 2)
            row["scanner_delta"] = {nm: round(float(theta[j] - K[1:][j] @ cur), 2)
                                    for j, nm in enumerate(SH8)}
            return row

        c1, _, r1, o1 = solve_arm(b, S9, np.zeros((len(b), 0)), None,
                                  bounds8, weights, IMAX)
        cmp_rows.append(report("1_f0_SH8", c1, None, o1))

        Cb = maps.reshape(-1, n_ch)[brain.ravel(), :].astype(np.float64)
        c2, i2, r2, o2 = solve_arm(b, S9, Cb, np.eye(n_ch), bounds8,
                                   weights, IMAX)
        cmp_rows.append(report("2_f0_SH8_full12", c2, i2, o2))

        # Arm 3: currents fitted to the selected target modes (full 12 channels
        # used for the fit -- the modes constrain the target, not the currents),
        # then the harmonics and f0 clean up whatever is left.
        sel_tags = [t for t, _ in tgt_paths]
        U = np.stack([u_native[t] for t in sel_tags], axis=1)
        coef, *_ = np.linalg.lstsq(U * np.sqrt(wP_full)[:, None],
                                   b_H * np.sqrt(wP_full), rcond=None)
        u_star = U @ coef
        i3, _ = bounded_fit(C_H, -u_star, wP_full, wR, IMAX)
        b3 = b + Cb @ i3
        c3, _, r3, o3 = solve_arm(b3, S9, np.zeros((len(b), 0)), None,
                                  bounds8, weights, IMAX)
        cmp_rows.append(report("3_f0_SH8_targetPCA", c3, i3, o3))

        for r in cmp_rows[-3:]:
            P.log(f"    {r['arm']:20s} PFC rmse {r['pfc_rmse']:7.2f} "
                  f"std {r['pfc_std']:7.2f}  hi-order {r['pfc_highorder_rmse']:6.2f}  "
                  f"whole {r['whole_rmse']:7.2f}  |I|rms {r['current_rms']:5.3f}  "
                  f"f0 {r['f0_delta_hz']:+8.1f}")

    P.log("")
    P.log("reachability by mode (mean over scans)")
    for tag in [t for t, _ in tgt_paths]:
        rs = [r for r in reach_rows if r["mode"] == tag]
        if not rs:
            continue
        P.log(f"  {tag:7s} unconstrained {np.mean([r['unconstrained_reach'] for r in rs]):.3f}  "
              f"bounded {np.mean([r['bounded_reach'] for r in rs]):+.3f}  "
              f"PFC share {np.mean([r['pfc_share_of_mode'] for r in rs]):.3f}  "
              f"|align b_H| {np.mean([abs(r['align_with_own_bH']) for r in rs]):.3f}  "
              f"|I|rms {np.mean([r['current_rms'] for r in rs]):.2f}")

    P.log("")
    P.log("arms (mean over scans, worst in brackets)")
    agg = []
    for arm in ("1_f0_SH8", "2_f0_SH8_full12", "3_f0_SH8_targetPCA"):
        rs = [r for r in cmp_rows if r["arm"] == arm]
        if not rs:
            continue
        a = {"arm": arm, "n": len(rs)}
        for k in ("pfc_rmse", "pfc_std", "pfc_highorder_rmse", "rest_rmse",
                  "whole_rmse", "current_rms", "current_peak"):
            v = np.array([r[k] for r in rs])
            a[k + "_mean"] = round(float(v.mean()), 3)
            a[k + "_worst"] = round(float(v.max()), 3)
        a["sat_mean"] = round(float(np.mean([r["channels_at_limit"] for r in rs])), 2)
        a["f0_absmean"] = round(float(np.mean([abs(r["f0_delta_hz"]) for r in rs])), 2)
        agg.append(a)
        P.log(f"  {arm:20s} PFC {a['pfc_rmse_mean']:7.2f} [{a['pfc_rmse_worst']:7.2f}]  "
              f"hi-order {a['pfc_highorder_rmse_mean']:6.2f}  "
              f"whole {a['whole_rmse_mean']:7.2f}  "
              f"|I|rms {a['current_rms_mean']:5.3f}  sat {a['sat_mean']:.1f}  "
              f"|f0| {a['f0_absmean']:.1f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fn, data in (("reachable_mode_scores.csv", reach_rows),
                     ("validation_arms_per_subject.csv",
                      [{k: (v if not isinstance(v, dict) else "")
                        for k, v in r.items()} for r in cmp_rows]),
                     ("validation_arms_metrics.csv", agg)):
        if not data:
            continue
        keys = sorted({k for r in data for k in r})
        with open(OUT_DIR / fn, "w", newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=keys)
            w_.writeheader(); w_.writerows(data)
    write_json(OUT_DIR / "validation_arms_detail.json",
               {"fold": args.fold, "n_scans": len(sel),
                "lambda_leak": LAMBDA_LEAK, "mu_current": MU_CURRENT,
                "taper_mm": TAPER_MM, "imax": IMAX,
                "aggregate": agg, "reach": reach_rows, "arms": cmp_rows})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
