"""Which constraint on the rest of the brain actually binds?

Route 1 established a specific number: with the rest of the brain held at the
shipping solution's RMS, the PFC gains **2.47 Hz** (median, 40 scans). That is
a property of *one choice of constraint*, not of the hardware. It answers

    min PFC RMS   s.t.  rest MEAN SQUARE <= rest RMS of production

and nothing else. If the functional is wrong -- if what actually damages an EPI
image in the rest of the brain is not the field magnitude but its gradient, or
its tail rather than its mean -- then the 2.47 Hz is measuring the wrong wall
and there is more PFC available at no real cost.

This script changes the functional and re-measures. For every scan it solves

    pfc_free          min PFC RMS, box only (the dream bound; wrecks the rest)
    pfc_guarded_rms   min PFC RMS  s.t.  rest RMS      <= tau_rms   (route 1)
    pfc_guarded_grad  min PFC RMS  s.t.  rest |grad|   <= tau_grad
    pfc_guarded_cvar  min PFC RMS  s.t.  rest CVaR_0.9 <= tau_cvar

with every tau set from *this scan's own production solution*, plus a 1% slack
so the feasible set has an interior (the exact cap is on the boundary and the
conic solvers report `infeasible` rather than returning that point).

All four are solved in the **direct** parameterisation -- a plain box on the
scanner command `s` and on the currents `i` -- not in the projected form
`adaptive_joint` uses internally, because there the console envelope applies to
`theta - K i` while the solver bounds `theta`. See `controllability_audit` for
why; on this cohort that defect turns out not to bite (production's delivered
command clears the envelope by >=743 Hz on every scan), but the direct form is
what makes each of the four problems above unambiguous.

Conventions
-----------
* f0 is free: every field is measured after subtracting its in-brain mean, which
  is the centre frequency prescan would pick. `b0` and every design column are
  demeaned over the brain once, so `b + S s + C i` *is* the post-f0 residual and
  there is no separate f0 variable to drift.
* The rest gradient is the isotropic 3-vector gradient, in Hz per voxel, summed
  over the three grid axes and averaged over rest voxels. The true EPI axis is
  A->P; the isotropic version is used because it needs no orientation decision
  and still answers the question "is mean square the right functional?". A
  single-axis version will sharpen, not reverse, the comparison.
* Voxels whose +1 neighbour along any axis leaves the brain mask are excluded
  from the gradient average, so no difference ever straddles the skull.

Outputs `results/region_constraint_study/`:
    per_subject.csv     one row per (scan, config)
    summary.md          medians / IQR / P10 / P90, and the headline comparison
    <scan>.npz          the four solution vectors

    python tools/research/run_region_constraint_study.py --limit 3
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# tools/<group>/<script>.py -> tools/<group> -> tools -> project root
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import nibabel as nib  # noqa: E402

from core.adaptive_joint import solve_adaptive_joint  # noqa: E402
from core.coil_registry import get_coil  # noqa: E402
from core.controllability_audit import (  # noqa: E402
    DEFAULT_SCANNER_LIMIT, GUARD_SLACK, _factor, _rms, _solve_ls)
from core.hybrid_shim import SH12, build_sh_basis  # noqa: E402
from core.pipeline import _load_coil_maps  # noqa: E402
from core.subject_split import split_subjects  # noqa: E402
from common import find_subjects  # noqa: E402

OUT = ROOT / "results" / "region_constraint_study"

#: Mean of the worst 10% of |rest residual|, Rockafellar-Uryasev. Matches the
#: beta used by route 5, so the two studies' tail numbers are comparable.
CVAR_BETA = 0.90

CONFIGS = ("pfc_free", "pfc_guarded_rms", "pfc_guarded_grad", "pfc_guarded_cvar")

QUALIFIER = (
    "Every number here is a mean square (or a tail of one) over a mask. It says "
    "nothing about a measured EPI image, and a win on the gradient functional is "
    "not a win on the image until the gradient functional is itself validated "
    "against one -- which, on this data, it is not.")


def _demean_over_brain(mask_brain, *arrs):
    """Subtract each array's in-brain mean, so f0 is free by construction.

    Every input is a field on the grid (b0, or one SH / coil column), so each is
    flattened, demeaned over the brain and returned with its original shape.
    """
    mb = np.asarray(mask_brain, dtype=bool).ravel()
    out = []
    for a in arrs:
        a = np.asarray(a, dtype=np.float64)
        flat = a.ravel()
        out.append((flat - float(flat[mb].mean())).reshape(a.shape))
    return out


def _forward_diff(col, axis):
    """Forward difference on a 3-D array, same shape, zero at the last slice."""
    d = np.zeros_like(col)
    dst = [slice(None)] * 3
    src = [slice(None)] * 3
    dst[axis] = slice(0, -1)
    src[axis] = slice(1, None)
    d[tuple(dst)] = col[tuple(src)] - col[tuple(dst)]
    return d


def _grad_region(design_cols, b, valid, spacing):
    """Stacked design for the isotropic gradient, mean over `valid` voxels.

    Returns (A, b_lin) with `||A x + b_lin||^2 / A.shape[0] * 3` equal to
    mean over `valid` of |grad(b + sum_k cols_k x_k)|^2, in Hz^2 per voxel
    (with `spacing` folded in as Hz per mm).
    """
    blocks_a, blocks_b = [], []
    nb = valid.ravel()
    for axis in range(3):
        cols = [_forward_diff(c, axis).reshape(-1)[nb] / spacing[axis]
                for c in design_cols]
        blocks_a.append(np.column_stack(cols))
        blocks_b.append(_forward_diff(b, axis).reshape(-1)[nb] / spacing[axis])
    A = np.vstack(blocks_a) / np.sqrt(len(blocks_b[0]))
    b_lin = np.concatenate(blocks_b) / np.sqrt(len(blocks_b[0]))
    return A, b_lin


def _solve_cvar_pfc(A_p, b_p, A_r, b_r, beta, cap, s_lo, s_hi, imax, name):
    """min PFC RMS s.t. CVaR_beta(|rest residual|) <= cap.

    Rockafellar-Uryasev: CVaR = min_t t + 1/((1-beta)N) * sum(max(|r|-t, 0)).
    The pointwise term is an epigraph variable per rest voxel, so this is an LP
    in (s, i, t, u) -- solved as a conic program alongside the SOC objective.

    Both `r` and `t` are in Hz, so the constraint is already in Hz and needs no
    rescaling.
    """
    import cvxpy as cp

    n_sh, n_ch = A_p.shape[1] - A_r.shape[1], A_r.shape[1]
    n_sh, n_ch = len(SH12), A_p.shape[1] - len(SH12)
    n_r = A_r.shape[0]

    x = cp.Variable(A_p.shape[1])
    t = cp.Variable()
    u = cp.Variable(n_r, nonneg=True)
    cons = [u >= cp.abs(A_r @ x + b_r) - t,
            x >= np.r_[s_lo, np.full(n_ch, -imax)],
            x <= np.r_[s_hi, np.full(n_ch, imax)],
            t + cp.sum(u) / ((1.0 - beta) * n_r) <= cap]
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A_p @ x + b_p)), cons)
    status, used = None, None
    for solver in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=solver)
            status, used = prob.status, solver
        except Exception as e:  # noqa: BLE001
            status, used = "error: %s: %s" % (type(e).__name__, e), solver
            continue
        if x.value is not None and status in ("optimal", "optimal_inaccurate"):
            break
    if x.value is None:
        raise RuntimeError("%s: no solution (%s)" % (name, status))
    xv = np.asarray(x.value, dtype=float).ravel()
    return xv, {"status": status, "solver": used, "violation": 0.0}


def study_subject(b0, brain, pfc, maps, affine, imax, scanner_limit, log=None):
    log = log or (lambda *a: None)
    brain = np.asarray(brain, dtype=bool)
    pfc = np.asarray(pfc, dtype=bool) & brain
    rest = brain & ~pfc
    if not pfc.any():
        raise ValueError("PFC mask does not overlap the brain")

    n_sh, n_ch = len(SH12), maps.shape[-1]
    basis = build_sh_basis(affine, b0.shape[:3])

    # ── design on the whole grid, demeaned so f0 is free ────────────────
    cols3 = [np.asarray(basis[n], dtype=np.float64) for n in SH12] + \
            [np.asarray(maps[..., j], dtype=np.float64) for j in range(n_ch)]
    b0d, *cols3d = _demean_over_brain(brain, np.asarray(b0, np.float64), *cols3)

    mb = brain.ravel()
    ip = pfc.ravel()
    ir = rest.ravel()

    def design_at(flat_mask):
        return np.column_stack([c.reshape(-1)[flat_mask] for c in cols3d])

    A_p = design_at(ip)
    b_p = b0d.reshape(-1)[ip]
    A_r = design_at(ir)
    b_r = b0d.reshape(-1)[ir]

    # voxel spacing in mm, from the affine column norms
    spacing = np.linalg.norm(np.asarray(affine)[:3, :3], axis=0)

    # ── gradient rows: voxel and its +1 neighbour both inside the brain ──
    ok = brain.copy()
    for axis in range(3):
        shifted = np.zeros_like(brain)
        dst = [slice(None)] * 3
        src = [slice(None)] * 3
        dst[axis] = slice(1, None)
        src[axis] = slice(0, -1)
        shifted[tuple(dst)] = brain[tuple(src)]
        ok &= shifted
    grad_valid = ok & rest
    A_g, b_g = _grad_region(cols3d, b0d, grad_valid, spacing)

    # ── production reference ────────────────────────────────────────────
    s_lo, s_hi = np.full(n_sh, -scanner_limit), np.full(n_sh, scanner_limit)
    prod = solve_adaptive_joint(
        b0, brain, [basis[n] for n in SH12], maps, imax=imax,
        lambda_pfc_low=0.0,
        weight_kwargs={"base": 1.0, "alpha": 0.0, "top_fraction": 0.0},
        sh_bounds=list(zip(s_lo, s_hi)), log_cb=None)
    x_ref = np.r_[np.asarray(prod["scanner_sh_delta"], float),
                  np.asarray(prod["coil_currents"], float)]

    def metrics(x):
        s, i = x[:n_sh], x[n_sh:]
        f = b0d + sum(s[k] * cols3d[k] for k in range(n_sh))
        for j in range(n_ch):
            f = f + i[j] * cols3d[n_sh + j]
        fl = f.reshape(-1)
        pfc_rms = float(np.sqrt(np.mean(fl[ip] ** 2)))
        rest_rms = float(np.sqrt(np.mean(fl[ir] ** 2)))
        af = np.abs(fl[ir])
        kk = int(np.ceil((1.0 - CVAR_BETA) * af.size))
        rest_cvar = float(np.mean(np.sort(af)[af.size - kk:])) if kk else 0.0
        g2 = np.zeros(fl.size)
        gv = grad_valid.ravel()
        for axis in range(3):
            g2 += (_forward_diff(f, axis).reshape(-1) / spacing[axis]) ** 2
        rest_grad = float(np.sqrt(np.mean(g2[gv])))
        return dict(pfc_rms=pfc_rms, rest_rms=rest_rms,
                    rest_cvar=rest_cvar, rest_grad=rest_grad)

    ref = metrics(x_ref)
    # 1% slack on each cap: the exact cap sits on the boundary of the feasible
    # set and the conic solvers return `infeasible` rather than that point.
    tau_rms = ref["rest_rms"] * (1.0 + GUARD_SLACK)
    tau_grad = ref["rest_grad"] * (1.0 + GUARD_SLACK)
    tau_cvar = ref["rest_cvar"] * (1.0 + GUARD_SLACK)

    R_p, u_p, res_p = _factor(A_p, b_p)
    R_r, u_r, res_r = _factor(A_r, b_r)
    R_g, u_g, res_g = _factor(A_g, b_g)
    # `_factor`/`_rms` work in RMS of their own stacked vector; calibrate the
    # gradient and CVaR scales against the direct computation rather than
    # trusting an analytic factor.
    grad_scale = ref["rest_grad"] / max(_rms(R_g, u_g, res_g, x_ref), 1e-12)

    lo = np.r_[s_lo, np.full(n_ch, -imax)]
    hi = np.r_[s_hi, np.full(n_ch, imax)]
    rows = []

    def row(name, x, info, note):
        m = metrics(x)
        m.update({"config": name, "note": note,
                  "solver": info.get("solver", ""), "status": info.get("status", ""),
                  "current_rms": float(np.sqrt(np.mean(x[n_sh:] ** 2))),
                  "current_max": float(np.max(np.abs(x[n_sh:]))),
                  "at_limit": int(np.sum(np.abs(np.abs(x[n_sh:]) - imax) < 1e-3)),
                  "scanner_max_abs": float(np.max(np.abs(x[:n_sh]))),
                  "tau_rms": tau_rms, "tau_grad": tau_grad, "tau_cvar": tau_cvar,
                  "violation": float(info.get("violation", 0.0))})
        rows.append(m)

    # reference row: production itself, measured in all four functionals
    row("production", x_ref, {"status": "L-BFGS-B", "solver": "scipy"},
        "shipping whole-brain solve (projected form)")
    row("baseline", np.zeros(n_sh + n_ch), {"status": "baseline"}, "unshimmed")

    x, info = _solve_ls(R_p, u_p, lo, hi, name="pfc_free")
    row("pfc_free", x, info, "min PFC RMS, box only")

    x, info = _solve_ls(R_p, u_p, lo, hi, soc=[(R_r, u_r, res_r, tau_rms)],
                        name="pfc_guarded_rms")
    row("pfc_guarded_rms", x, info, "min PFC RMS s.t. rest RMS held (route 1)")

    x, info = _solve_ls(R_p, u_p, lo, hi,
                        soc=[(R_g, u_g, res_g, tau_grad / max(grad_scale, 1e-12))],
                        name="pfc_guarded_grad")
    row("pfc_guarded_grad", x, info, "min PFC RMS s.t. rest GRADIENT held")

    x, info = _solve_cvar_pfc(A_p, b_p, A_r, b_r, CVAR_BETA, tau_cvar,
                              s_lo, s_hi, imax, "pfc_guarded_cvar")
    row("pfc_guarded_cvar", x, info, "min PFC RMS s.t. rest CVaR held")

    return {"rows": rows, "ref": ref, "spacing": spacing.tolist(),
            "n_pfc": int(pfc.sum()), "n_rest": int(rest.sum()),
            "n_grad_voxels": int(grad_valid.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coil", default="PFC 12ch")
    ap.add_argument("--scanner-limit", type=float, default=DEFAULT_SCANNER_LIMIT)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    coil = get_coil(args.coil)
    found = find_subjects(args.limit, True)
    fold_of = split_subjects([n for n, *_ in found])

    all_rows, failures, per_scan = [], [], {}
    print("%d scans, coil %s (+-%.1f A)\n"
          % (len(found), coil.name, coil.current_limit_a))

    # Written row by row: a long cohort run that is interrupted still leaves
    # every scan it finished on disk.
    keys = ["subject", "fold", "config", "note", "pfc_rms", "rest_rms",
            "rest_grad", "rest_cvar", "current_rms", "current_max", "at_limit",
            "scanner_max_abs", "tau_rms", "tau_grad", "tau_cvar",
            "solver", "status", "violation"]
    csv_f = open(OUT / "per_subject.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()

    for k, (name, hz, bp, pp) in enumerate(found, 1):
        t0 = time.time()
        try:
            img = nib.load(str(hz))
            b0 = np.asanyarray(img.dataobj, dtype=np.float64)
            brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
            pfc = np.asanyarray(nib.load(str(pp)).dataobj) > 0.5
            maps, _ = _load_coil_maps(coil, img.affine, b0.shape[:3],
                                      lambda *a: None)
            out = study_subject(b0, brain, pfc, maps, img.affine,
                                coil.current_limit_a, args.scanner_limit)
        except Exception as e:  # noqa: BLE001
            failures.append((name, "%s: %s" % (type(e).__name__, e)))
            print("[%2d/%d] %-12s FAILED: %s" % (k, len(found), name, e))
            traceback.print_exc()
            continue

        fold = fold_of.get(name, "?")
        by = {r["config"]: r for r in out["rows"]}
        for r in out["rows"]:
            rec = dict(r, subject=name, fold=fold)
            all_rows.append(rec)
            writer.writerow(rec)
        csv_f.flush()
        per_scan[name] = out

        base, prod = by["baseline"], by["production"]
        gain = {c: prod["pfc_rms"] - by[c]["pfc_rms"] for c in CONFIGS}
        print("[%2d/%d] %-12s %-10s base %6.1f prod %6.1f | "
              "PFC gain Hz: free %+6.1f  rms %+5.2f  grad %+5.2f  cvar %+5.2f  "
              "(%.0fs)"
              % (k, len(found), name, fold, base["pfc_rms"], prod["pfc_rms"],
                 gain["pfc_free"], gain["pfc_guarded_rms"],
                 gain["pfc_guarded_grad"], gain["pfc_guarded_cvar"],
                 time.time() - t0))

        np.savez_compressed(
            OUT / ("%s.npz" % name),
            configs=np.array([r["config"] for r in out["rows"]]),
            pfc_rms=np.array([r["pfc_rms"] for r in out["rows"]]),
            rest_rms=np.array([r["rest_rms"] for r in out["rows"]]),
            rest_grad=np.array([r["rest_grad"] for r in out["rows"]]),
            rest_cvar=np.array([r["rest_cvar"] for r in out["rows"]]),
            spacing=np.asarray(out["spacing"]),
            n_pfc=out["n_pfc"], n_rest=out["n_rest"],
            n_grad_voxels=out["n_grad_voxels"])

    csv_f.close()
    if not all_rows:
        print("nothing solved"); return

    # ── summary ─────────────────────────────────────────────────────────
    def gains(cfg):
        cur = {r["subject"]: r for r in all_rows if r["config"] == cfg}
        pro = {r["subject"]: r for r in all_rows if r["config"] == "production"}
        return [pro[k]["pfc_rms"] - cur[k]["pfc_rms"]
                for k in sorted(cur) if k in pro]

    def fmt(v):
        if not v:
            return "n/a"
        v = sorted(v)
        q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
        return ("median %+6.2f  IQR %+6.2f..%+6.2f  P10 %+6.2f  P90 %+6.2f  "
                "worst %+6.2f" % (st.median(v), q(0.25), q(0.75), q(0.10),
                                  q(0.90), v[0]))

    lines = ["# Region-constraint study — which functional on the rest actually binds?", "",
             "> " + QUALIFIER, "",
             "%d scans solved, %d failed. Coil `%s`, +-%.1f A. "
             "Every cap is this scan's own production value +%d%%."
             % (len(per_scan), len(failures), coil.name, coil.current_limit_a,
                round(100 * GUARD_SLACK)), "",
             "## PFC gained over production (Hz), by which functional is held", "",
             "| constraint on the rest | PFC gain |", "|---|---|"]
    for cfg, lbl in (("pfc_free", "none (dream bound)"),
                     ("pfc_guarded_rms", "rest RMS = production (route 1)"),
                     ("pfc_guarded_grad", "rest gradient = production"),
                     ("pfc_guarded_cvar", "rest CVaR(0.9) = production")):
        lines.append("| %s | %s |" % (lbl, fmt(gains(cfg))))

    lines += ["", "## What each choice costs in the functionals it did NOT hold", "",
              "These are the numbers that decide whether a gain above is free or "
              "merely moved to a quantity the constraint ignored.", ""]
    for hold in ("pfc_free", "pfc_guarded_rms", "pfc_guarded_grad",
                 "pfc_guarded_cvar"):
        cur = {r["subject"]: r for r in all_rows if r["config"] == hold}
        pro = {r["subject"]: r for r in all_rows if r["config"] == "production"}
        keys = [k for k in sorted(cur) if k in pro]
        lines.append("**%s** — held: %s" % (
            hold, {"pfc_free": "nothing", "pfc_guarded_rms": "rest RMS",
                   "pfc_guarded_grad": "rest gradient",
                   "pfc_guarded_cvar": "rest CVaR"}[hold]))
        for o, lbl in (("pfc_rms", "PFC RMS"), ("rest_rms", "rest RMS"),
                       ("rest_grad", "rest gradient"), ("rest_cvar", "rest CVaR")):
            dv = [cur[k][o] - pro[k][o] for k in keys]
            lines.append("  * %-14s change vs production: %s" % (lbl, fmt(dv)))
        lines.append("")
    if failures:
        lines += ["## Failures", ""] + ["  * %s — %s" % f for f in failures]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\nwrote %s" % (OUT / "summary.md"))


if __name__ == "__main__":
    main()
