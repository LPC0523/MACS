"""Route 5: optimise the PFC's worst decile instead of its mean square.

Routes 1 and 2 established that the shipping solution is Pareto-optimal in
(PFC RMS, rest RMS): at a 0% rest budget nothing improves the PFC without
raising rest RMS. That claim is confined to regional MSE/RMS under the
identical feasible set, and it says nothing about the tail. A mean square will
trade a large error over few voxels for a small error over many, so two
solutions with the same RMS can have quite different worst deciles. Whether
that difference matters downstream is not established here and is not assumed:
this module measures a property of the field, and no EPI image is involved.

So this module asks the same constrained question with a different objective:

    minimise   CVaR_beta( |residual| over the PFC )
    subject to rest RMS <= (1 + budget) * rest_reference
               scanner delta inside the deployment envelope
               |i| <= imax

with `beta = 0.9`, i.e. the mean of the worst 10% of |residual| -- the quantity
already reported as `pfc_worst10`, so the objective and the metric are again
the same number.

If the RMS solution were also tail-optimal this would return it. Any difference
is a direct measurement of how much the RMS front misses, at the same price in
rest homogeneity.

Formulation
-----------
Rockafellar-Uryasev: CVaR_beta(Z) = min_t { t + E[(Z - t)+] / (1 - beta) }.
With Z = |A x + b| this is a linear program in (x, t, u), convex and exact --
no smoothing, no surrogate:

    minimise  t + sum(u) / ((1 - beta) n)
    s.t.      u >= 0,  u >= +(A x + b) - t,  u >= -(A x + b) - t

The PFC has ~17k voxels, so `u` has ~17k entries. That is the one place in this
project where the full design matrix does reach the solver: unlike a mean
square, a tail measure cannot be reduced to a 20x20 normal form, because it
depends on which voxels are extreme and not only on second moments. The rest
constraint stays factored.

Scope
-----
Pareto optimality applies only to regional MSE/RMS under the identical feasible
set; it does not imply Pareto optimality for CVaR, EPI displacement, signal
dropout, or measured EPI image quality. This module addresses the first of
those omissions and none of the others: CVaR of the field is still not a
measurement of an EPI image.
"""

from __future__ import annotations

import numpy as np

from core.controllability_audit import _factor, _rms
from core.pfc_guarded_joint import GUARD_TOL_HZ, GuardViolation, _box, _build, _residual
from core.shim_metrics import metric_row

__all__ = ["DEFAULT_BETA", "cvar_of", "solve_cvar_guarded"]

#: Tail fraction. 0.9 means the mean of the worst 10%, matching `pfc_worst10`.
DEFAULT_BETA = 0.9


def cvar_of(values, beta: float = DEFAULT_BETA) -> float:
    """Mean of the worst `1 - beta` fraction of |values|. The reported metric.

    Computed by sorting rather than from the optimiser's `t`, so the number in
    the report never depends on the solver having converged.
    """
    v = np.sort(np.abs(np.asarray(values, dtype=float)))
    if v.size == 0:
        return float("nan")
    k = int(np.floor(beta * v.size))
    k = min(max(k, 0), v.size - 1)
    return float(np.mean(v[k:]))


def solve_cvar_guarded(b0, brain_mask, pfc_mask, sh_maps, coil_maps,
                       scanner_bounds, imax: float = 3.0,
                       rest_reference: float = None,
                       reference_solution=None,
                       budget: float = 0.0,
                       beta: float = DEFAULT_BETA,
                       max_voxels: int = 0,
                       log_cb=None) -> dict:
    """Minimise the PFC tail with the rest of the brain held at a level.

    `max_voxels` > 0 subsamples the PFC before building the LP. Left at 0 the
    full mask is used; if it is set, the value actually used is recorded on the
    returned row so a subsampled run can never be mistaken for a full one.
    """
    import cvxpy as cp

    log = log_cb or (lambda *_: None)
    d = _build(b0, brain_mask, pfc_mask, sh_maps, coil_maps)
    n_sh, n_ch = d["n_sh"], d["n_ch"]
    R_r, u_r, res_r = d["f_rest"]
    lo, hi = _box(scanner_bounds, n_sh, n_ch, imax)

    if rest_reference is None:
        if reference_solution is None:
            raise ValueError("pass rest_reference or reference_solution")
        rest_reference = _rms(R_r, u_r, res_r,
                              np.asarray(reference_solution, dtype=float))
    rest_reference = float(rest_reference)
    cap = rest_reference * (1.0 + float(budget))

    # Rebuild the PFC design in full: the tail needs the voxels themselves.
    pfc, brain = d["pfc"], d["brain"]
    flat = coil_maps.reshape(-1, n_ch) if n_ch else np.zeros((b0.size, 0))
    mb_b = float(b0[brain].mean())
    mb_S = np.array([float(m[brain].mean()) for m in d["sh_maps"]])
    mb_C = flat[brain.ravel()].mean(axis=0) if n_ch else np.zeros(0)
    A = np.hstack([np.stack([m[pfc] for m in d["sh_maps"]], axis=-1) - mb_S,
                   (flat[pfc.ravel()] - mb_C) if n_ch
                   else np.zeros((int(pfc.sum()), 0))]).astype(np.float64)
    y = (b0[pfc] - mb_b).astype(np.float64)

    n_full = A.shape[0]
    used = n_full
    if max_voxels and n_full > max_voxels:
        # Deterministic stride, not a random draw: the same subject must give
        # the same answer on every run.
        step = int(np.ceil(n_full / max_voxels))
        A, y = A[::step], y[::step]
        used = A.shape[0]

    x = cp.Variable(A.shape[1])
    t = cp.Variable()
    u = cp.Variable(A.shape[0], nonneg=True)
    r = A @ x + y
    cons = [u >= r - t, u >= -r - t, x >= lo, x <= hi,
            cp.norm(R_r @ x + u_r) <= np.sqrt(
                max(cap ** 2 - res_r, 0.0))]
    if cap ** 2 < res_r:
        raise GuardViolation(
            "rest cap %.4f Hz is below the unremovable floor %.4f Hz"
            % (cap, np.sqrt(res_r)))
    obj = cp.Minimize(t + cp.sum(u) / ((1.0 - beta) * A.shape[0]))

    prob = cp.Problem(obj, cons)
    status, solver = None, None
    for s in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=s)
            status, solver = prob.status, s
        except Exception as e:
            status, solver = "error: %s: %s" % (type(e).__name__, e), s
            continue
        if x.value is not None and status in ("optimal", "optimal_inaccurate"):
            break
    if x.value is None:
        raise RuntimeError("cvar_guarded: no solution (%s)" % status)

    xv = np.asarray(x.value, dtype=float).ravel()
    rest_hz = _rms(R_r, u_r, res_r, xv)
    over_guard = rest_hz - cap
    over_box = max(float(np.max(lo - xv, initial=0.0)),
                   float(np.max(xv - hi, initial=0.0)))
    if over_guard > GUARD_TOL_HZ or over_box > 1e-6:
        raise GuardViolation(
            "solution violates its constraints and will not be clipped: "
            "rest %.4f over cap %.4f by %.3e, box by %.3e"
            % (rest_hz, cap, over_guard, over_box))

    resid = _residual(b0, d, xv)
    row = metric_row("cvar(beta=%.2f,budget=%.4g)" % (beta, budget),
                     resid, d["pfc"], d["rest"], d["brain"])
    f0 = row["f0_hz"]
    row.update({
        "budget": float(budget), "beta": float(beta),
        "objective": "PFC CVaR_%.2f" % beta,
        # Recomputed over the *full* mask from the residual, so a subsampled
        # solve is still scored on every voxel it will actually be used on.
        "pfc_cvar": cvar_of(resid[d["pfc"]] + f0, beta),
        "rest_reference_hz": rest_reference, "rest_cap_hz": float(cap),
        "pfc_voxels_total": int(n_full), "pfc_voxels_used": int(used),
        "scanner_sh_delta": xv[:n_sh].copy(), "coil_currents": xv[n_sh:].copy(),
        "current_max": float(np.max(np.abs(xv[n_sh:]))) if n_ch else 0.0,
        "solver": solver, "status": status,
        "violation": float(max(over_guard, over_box, 0.0)),
    })
    log("  CVaR budget %.4g -> tail %.2f Hz, RMS %.2f Hz, rest %.2f / %.2f"
        % (budget, row["pfc_cvar"], row["pfc_rms"], row["rest_rms"], cap))
    return row
