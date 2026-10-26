"""Validation: does a reduced mode set buy anything over the full 12 channels?

    D:\\python.exe tools/research/pfc_modes/phase3_validation.py [--limit 3]

Methods, all on the same B0, masks, harmonics, weighting, bounds and solver:

    A  SH only
    B  SH + all 12 channels          (the shipping region-weighted joint)
    D  SH + K Slepian modes          K = 2, 3, 4, computed per scan

C (PCA modes) is not run: Phase 2 found the cohort has no usable common space
in the scanner frame -- the all-scan PFC intersection is empty -- so there is no
space in which a cross-subject residual PCA would mean anything.

The reduced set's feasible region is a subset of the full 12-channel one under
the identical objective, so D cannot beat B on the objective and is not claimed
to. What it might buy is smaller currents, fewer effective degrees of freedom,
less saturation, or a better worst case.

Bounds are applied to the real per-channel currents i = V a, never to the mode
coefficients a: a mode set with large-amplitude channel patterns could otherwise
report a "feasible" solution that the amplifiers would refuse.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, LinearConstraint

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (            # noqa: E402
    SH8, IMAX, LAMBDA_REST, OUT_DIR, Progress, basis_with_a00, cohort,
    coil_maps_for, load_subject, sh_bounds_production, sh_matrix, write_json)
from tools.research.pfc_modes.phase1_slepian import (    # noqa: E402
    TAPER_MM, taper_weight, whitened_gevp)
from core.adaptive_joint import make_region_weights      # noqa: E402

#: One tolerance for every method, so a difference in the table is a difference
#: in the problem and not in how hard the solver tried.
SOLVER_OPTS = {"maxiter": 3000, "gtol": 1e-10, "xtol": 1e-12}


def solve(b, A_sh, C_eff, V, sh_bounds, weights, imax):
    """min sum_i w_i (A_sh theta + C_eff a + b)_i^2, |V a| <= imax, theta in box.

    `V` maps mode coefficients to real channel currents; pass the identity for
    the full-channel case. Quadratic objective with an assembled Hessian and
    exact gradients, and a linear constraint on the currents.
    """
    n_sh, n_a = A_sh.shape[1], C_eff.shape[1]
    D = np.hstack([A_sh, C_eff]) if n_a else A_sh
    w = weights / np.sum(weights)
    H = D.T @ (D * w[:, None])
    g = D.T @ (w * b)
    c0 = 0.5 * float(np.sum(w * b ** 2))

    def fun(x):
        Hx = H @ x
        return c0 + float(g @ x) + 0.5 * float(x @ Hx), g + Hx

    # The objective is exactly quadratic, so hand trust-constr the true
    # Hessian. Left to its quasi-Newton fallback it warns "delta_grad == 0.0"
    # and can stop short of the optimum -- which would make the comparison
    # between a reduced mode set and the full 12 channels a comparison of how
    # far each solve happened to get.
    def hess(_x):
        return H

    lo = [bd[0] for bd in sh_bounds] + [-np.inf] * n_a
    hi = [bd[1] for bd in sh_bounds] + [np.inf] * n_a
    cons = []
    if n_a:
        M = np.zeros((V.shape[0], n_sh + n_a))
        M[:, n_sh:] = V
        cons.append(LinearConstraint(M, -imax, imax))

    x0 = np.zeros(n_sh + n_a)
    res = minimize(fun, x0, jac=True, hess=hess, method="trust-constr",
                   bounds=list(zip(lo, hi)), constraints=cons,
                   options=SOLVER_OPTS)
    theta, a = res.x[:n_sh], res.x[n_sh:]
    # The objective value itself, which is what the subset argument constrains
    # -- PFC RMSE alone is only one component of it and may legitimately move
    # the other way.
    obj = float(fun(res.x)[0])
    return theta, a, res, obj


def metrics(resid, brain, pfc):
    rest = brain & ~pfc
    out = {}
    for tag, m in (("pfc", pfc), ("rest", rest), ("whole", brain)):
        v = resid[m]
        out[f"{tag}_rmse"] = round(float(np.sqrt(np.mean(v ** 2))), 3)
        out[f"{tag}_std"] = round(float(np.std(v)), 3)
        out[f"{tag}_mean"] = round(float(np.mean(v)), 3)
    return out


def run_subject(s, maps, basis, P):
    brain, pfc = s["brain"], s["pfc"]
    b = s["b0"][brain].astype(np.float64)
    A_sh = sh_matrix(basis, SH8, brain)
    C = maps.reshape(-1, maps.shape[-1])[brain.ravel(), :].astype(np.float64)
    n_ch = C.shape[1]
    bounds = sh_bounds_production(SH8)

    in_region = pfc[brain]
    weights = make_region_weights(in_region, LAMBDA_REST, per_voxel=False)

    # Slepian modes for this scan, on the SH-independent part, exactly as
    # Phase 1 computes them.
    Q, _ = np.linalg.qr(A_sh)
    C_perp = C - Q @ (Q.T @ C)
    wP = taper_weight(pfc, s["affine"], TAPER_MM)[brain]
    wB = np.ones(C.shape[0])
    lam, Vmodes, rank = whitened_gevp(C_perp.T @ (wP[:, None] * C_perp),
                                      C_perp.T @ (wB[:, None] * C_perp))

    # Coil low-order equivalent, for the scanner delta reported to the console.
    low_coeff, *_ = np.linalg.lstsq(A_sh * np.sqrt(weights)[:, None],
                                    C * np.sqrt(weights)[:, None], rcond=None)

    def finish(tag, theta, i_ch, res, obj):
        corr = np.zeros_like(s["b0"])
        for j, n in enumerate(SH8):
            corr += theta[j] * basis[n]
        if i_ch is not None and i_ch.size:
            corr += np.tensordot(maps, i_ch, axes=([3], [0]))
        resid = s["b0"] + corr
        row = {"subject": s["name"], "method": tag}
        row.update(metrics(resid, brain, pfc))
        cur = np.zeros(n_ch) if i_ch is None else np.asarray(i_ch, float)
        row["current_rms"] = round(float(np.sqrt(np.mean(cur ** 2))), 4)
        row["current_peak"] = round(float(np.max(np.abs(cur))) if cur.size else 0.0, 4)
        row["current_abs_sum"] = round(float(np.sum(np.abs(cur))), 3)
        row["channels_at_limit"] = int(np.sum(np.abs(np.abs(cur) - IMAX) < 1e-3))
        row["bound_violation"] = round(
            float(max(0.0, np.max(np.abs(cur)) - IMAX)) if cur.size else 0.0, 6)
        row["n_free_params"] = len(SH8) + (0 if i_ch is None else
                                           (n_ch if tag == "B_full12" else
                                            int(tag.split("K")[-1])
                                            if "K" in tag else n_ch))
        sd = theta - (low_coeff @ cur if cur.size else 0.0)
        row["scanner_delta"] = {n: round(float(v), 2) for n, v in zip(SH8, sd)}
        row["solver_status"] = int(res.status) if res is not None else 0
        row["solver_ok"] = bool(res.success) if res is not None else True
        row["objective"] = round(obj, 8)
        return row

    rows = []
    theta, a, res, obj = solve(b, A_sh, np.zeros((len(b), 0)), None, bounds,
                               weights, IMAX)
    rows.append(finish("A_SH_only", theta, None, res, obj))

    theta, i_full, res, obj = solve(b, A_sh, C, np.eye(n_ch), bounds, weights, IMAX)
    rows.append(finish("B_full12", theta, i_full, res, obj))

    for K in (2, 3, 4):
        if K > Vmodes.shape[1]:
            continue
        Vk = Vmodes[:, :K]
        theta, a, res, obj = solve(b, A_sh, C_perp @ Vk, Vk, bounds, weights, IMAX)
        # The hardware emits the full field of these currents, not just the
        # SH-independent part the modes were built from.
        rows.append(finish(f"D_slepian_K{K}", theta, Vk @ a, res, obj))

    # The subset argument, checked rather than assumed: every reduced set is
    # feasible for the full problem, so none may reach a lower objective. A
    # violation means the full solve stopped short, and the whole comparison
    # would be measuring solver effort instead of the mode set.
    obj_b = next(r["objective"] for r in rows if r["method"] == "B_full12")
    for r in rows:
        if r["method"].startswith("D_"):
            r["objective_minus_full12"] = round(r["objective"] - obj_b, 10)
            r["subset_ok"] = bool(r["objective"] >= obj_b - 1e-9 * max(abs(obj_b), 1.0))
            if not r["subset_ok"]:
                P.log(f"    ⚠ {r['method']} beat the full 12ch objective by "
                      f"{obj_b - r['objective']:.3e} — the full solve did not "
                      f"reach its optimum")

    for r in rows:
        P.log(f"    {r['method']:16s} PFC rmse {r['pfc_rmse']:8.2f}  "
              f"whole {r['whole_rmse']:8.2f}  obj {r['objective']:10.2f}  "
              f"|I|rms {r['current_rms']:6.3f}  "
              f"peak {r['current_peak']:5.2f}  sat {r['channels_at_limit']:2d}  "
              f"{'ok' if r['solver_ok'] else 'FAIL'}")
    return rows, {"lambda": np.round(lam[:6], 4).tolist(), "rank": rank}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="validation")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()

    P = Progress(f"Phase 3 — validation ({args.fold}, limit={args.limit or 'all'})")
    entries, folds = cohort()
    sel = [e for e in entries if folds[e[0]] == args.fold]
    if args.limit:
        sel = sel[:args.limit]
    P.log(f"{len(sel)} scans: {[e[0] for e in sel]}")

    all_rows, extra = [], {}
    for e in sel:
        s = load_subject(e)
        P.log(f"  {s['name']}  brain {int(s['brain'].sum()):,}  "
              f"PFC {int(s['pfc'].sum()):,}")
        maps, _ = coil_maps_for(s["affine"], s["shape"])
        basis = basis_with_a00(s["affine"], s["shape"])
        rows, ex = run_subject(s, maps, basis, P)
        all_rows.extend(rows)
        extra[s["name"]] = ex

    methods = ["A_SH_only", "B_full12", "D_slepian_K2", "D_slepian_K3",
               "D_slepian_K4"]
    P.log("")
    P.log("aggregate over scans (mean, and worst single scan)")
    agg = []
    for m in methods:
        rs = [r for r in all_rows if r["method"] == m]
        if not rs:
            continue
        pfc = np.array([r["pfc_rmse"] for r in rs])
        whole = np.array([r["whole_rmse"] for r in rs])
        a = {"method": m, "n": len(rs),
             "pfc_rmse_mean": round(float(pfc.mean()), 3),
             "pfc_rmse_worst": round(float(pfc.max()), 3),
             "whole_rmse_mean": round(float(whole.mean()), 3),
             "whole_rmse_worst": round(float(whole.max()), 3),
             "current_rms_mean": round(float(np.mean([r["current_rms"] for r in rs])), 4),
             "current_peak_mean": round(float(np.mean([r["current_peak"] for r in rs])), 4),
             "sat_mean": round(float(np.mean([r["channels_at_limit"] for r in rs])), 2),
             "max_bound_violation": round(float(np.max([r["bound_violation"] for r in rs])), 8),
             "objective_mean": round(float(np.mean([r["objective"] for r in rs])), 4),
             "subset_ok_all": bool(all(r.get("subset_ok", True) for r in rs)),
             "all_solved": bool(all(r["solver_ok"] for r in rs))}
        agg.append(a)
        P.log(f"  {m:16s} PFC {a['pfc_rmse_mean']:8.2f} (worst {a['pfc_rmse_worst']:8.2f})  "
              f"whole {a['whole_rmse_mean']:8.2f}  |I|rms {a['current_rms_mean']:6.3f}  "
              f"sat {a['sat_mean']:4.1f}  viol {a['max_bound_violation']:.1e}")

    base = next((a for a in agg if a["method"] == "B_full12"), None)
    if base:
        P.log("")
        P.log("relative to the full 12-channel joint (positive PFC = worse)")
        for a in agg:
            if a["method"] == "B_full12":
                continue
            P.log(f"  {a['method']:16s} PFC {a['pfc_rmse_mean'] - base['pfc_rmse_mean']:+7.2f} Hz  "
                  f"|I|rms {a['current_rms_mean'] - base['current_rms_mean']:+7.3f} A  "
                  f"({100 * a['current_rms_mean'] / max(base['current_rms_mean'], 1e-9):5.1f}% of its current)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = [{k: (v if not isinstance(v, dict) else "") for k, v in r.items()}
            for r in all_rows]
    keys = sorted({k for r in flat for k in r})
    with open(OUT_DIR / "validation_per_subject.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(flat)
    with open(OUT_DIR / "validation_metrics.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(agg[0])); w.writeheader(); w.writerows(agg)
    write_json(OUT_DIR / "validation_detail.json",
               {"fold": args.fold, "n_scans": len(sel), "taper_mm": TAPER_MM,
                "lambda_rest": LAMBDA_REST, "imax": IMAX,
                "solver": {"method": "trust-constr", **SOLVER_OPTS},
                "aggregate": agg, "per_subject": all_rows, "modes": extra})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
