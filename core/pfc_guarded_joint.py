"""Route 2: the best PFC the hardware can reach without giving the rest away.

Route 1 measured where the loss is, and it is not the hardware. Aiming the
scanner alone at the PFC reaches 59.5 Hz at the median while the shipping
whole-brain solution leaves the PFC at 84.0 Hz; on six of forty subjects
production leaves the PFC *worse* than no shim at all. The 24 Hz sits in the
choice of objective, not in the coils. What route 1 could not say is how much
of it is collectable, because a whole-brain objective is not merely a bad
choice -- it is protecting the rest of the head.

So the question this module answers is the constrained one:

    minimise   PFC RMS
    subject to rest RMS <= (1 + budget) * rest_reference
               scanner delta inside the deployment envelope
               |i| <= imax

`budget = 0` is the lexicographic case and the headline result: improve the PFC
while the rest of the brain is *no worse than the shipping solution leaves it*.
Larger budgets trace a Pareto front, so the exchange rate is visible rather
than assumed. Route 1 reported this bound with a 1% budget for a solver reason
that no longer applies; the measured room between `rest_reference` and the
lowest reachable rest RMS is 0.25-0.53 Hz on this cohort, so budget 0 is
strictly feasible with an interior and needs no slack.

Metric and objective are the same quantity
------------------------------------------
Every design matrix is centred on its in-brain mean, which builds the shared
centre frequency into the design: `b_c + S_c s + C_c i` is exactly the residual
after the single f0 that `shim_metrics` reports, for every region at once. The
optimiser therefore minimises the number that gets published, with no free f0
to drift between them. Demeaning PFC and rest separately would hide a genuine
frequency offset between the two regions; prescan sets one f0 for the head.

What "Pareto-optimal" does and does not cover
--------------------------------------------
Pareto optimality applies only to regional MSE/RMS under the identical
feasible set; it does not imply Pareto optimality for CVaR, EPI displacement,
signal dropout, or measured EPI image quality. Production being on this front
means only that no feasible point improves PFC RMS without raising rest RMS.
A solution off the front may still be better on a tail measure or in an actual
EPI image, and nothing here measures those.

For the same reason a 1% rest-RMS budget is a bound on one scalar summary of
one region, not a statement that the rest of the brain is unaffected in EPI.

`rest` is `brain & ~pfc`, always. Air is not in it.

Nothing is ever clipped after the solve. A returned vector that violates the
guard or the envelope raises; a clipped vector is not the solution to anything.
"""

from __future__ import annotations

import numpy as np

from core.controllability_audit import _factor, _rms, _solve_ls
from core.shim_metrics import metric_row

__all__ = ["GuardViolation", "solve_pfc_guarded", "pareto_front",
           "DEFAULT_BUDGETS"]

#: Rest-degradation budgets, as a fraction of the reference rest RMS. The first
#: is the result; the rest are Pareto points showing what buying more PFC costs.
DEFAULT_BUDGETS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05)

#: How far past the cap a returned solution may sit before it is rejected, in
#: Hz. Solver-level slop, not a budget: at 1e-3 Hz nothing physical is hidden.
GUARD_TOL_HZ = 1e-3


class GuardViolation(RuntimeError):
    """The returned solution does not respect the guard or the envelope."""


def _build(b0, brain, pfc, sh_maps, coil_maps):
    """Brain-centred designs for PFC, rest and brain. Returns a dict."""
    brain = np.asarray(brain, dtype=bool)
    pfc = np.asarray(pfc, dtype=bool) & brain
    rest = brain & ~pfc
    if not pfc.any():
        raise ValueError("PFC mask does not overlap the brain")
    if not rest.any():
        raise ValueError("rest region is empty: the PFC mask covers the brain")

    n_ch = 0 if coil_maps is None else coil_maps.shape[-1]
    flat = (np.zeros((b0.size, 0)) if not n_ch
            else coil_maps.reshape(-1, n_ch))
    S_full = [np.asarray(m, dtype=np.float64) for m in sh_maps]

    mb_b = float(b0[brain].mean())
    mb_S = np.array([float(m[brain].mean()) for m in S_full])
    mb_C = (flat[brain.ravel()].mean(axis=0) if n_ch
            else np.zeros(0))

    def region(m):
        S = np.stack([f[m] for f in S_full], axis=-1) - mb_S
        C = (flat[m.ravel()] - mb_C) if n_ch else np.zeros((int(m.sum()), 0))
        return np.hstack([S, C]).astype(np.float64), (b0[m] - mb_b).astype(np.float64)

    A_p, y_p = region(pfc)
    A_r, y_r = region(rest)
    A_b, y_b = region(brain)
    return {
        "pfc": pfc, "rest": rest, "brain": brain,
        "n_sh": len(S_full), "n_ch": n_ch,
        "f_pfc": _factor(A_p, y_p),
        "f_rest": _factor(A_r, y_r),
        "f_brain": _factor(A_b, y_b),
        "sh_maps": S_full, "coil_maps": coil_maps,
    }


def _residual(b0, d, x):
    """The field after applying `x`, before the shared f0."""
    corr = np.zeros_like(b0, dtype=np.float64)
    for k, m in enumerate(d["sh_maps"]):
        corr += x[k] * m
    if d["n_ch"]:
        corr += np.tensordot(d["coil_maps"], x[d["n_sh"]:], axes=([3], [0]))
    return b0 + corr


def _box(scanner_bounds, n_sh, n_ch, imax):
    sb = np.asarray(scanner_bounds, dtype=float).reshape(n_sh, 2)
    lo = np.r_[sb[:, 0], np.full(n_ch, -abs(imax))]
    hi = np.r_[sb[:, 1], np.full(n_ch, abs(imax))]
    if np.any(lo > hi):
        raise ValueError("scanner_bounds has lo > hi; the envelope is empty")
    return lo, hi


def solve_pfc_guarded(b0, brain_mask, pfc_mask, sh_maps, coil_maps,
                      scanner_bounds, imax: float = 3.0,
                      rest_reference: float = None,
                      budget: float = 0.0,
                      reference_solution=None,
                      log_cb=None) -> dict:
    """Minimise PFC RMS with the rest of the brain held at a reference level.

    `scanner_bounds` is the per-term (lo, hi) window on the scanner *delta* --
    the intersection of the delta and absolute envelopes, asymmetric whenever
    the baseline is non-zero. Take it from `SystemShimState.delta_bounds()`;
    it is not recomputed here, so there is one place the policy lives.

    `rest_reference` is the rest RMS the guard is written against, in Hz. Pass
    `reference_solution` instead to have it measured from a solution vector
    (the shipping whole-brain solve, normally). One of the two is required:
    guessing it from the data would silently change what "no worse" means.
    """
    log = log_cb or (lambda *_: None)
    d = _build(b0, brain_mask, pfc_mask, sh_maps, coil_maps)
    n_sh, n_ch = d["n_sh"], d["n_ch"]
    R_p, u_p, res_p = d["f_pfc"]
    R_r, u_r, res_r = d["f_rest"]
    lo, hi = _box(scanner_bounds, n_sh, n_ch, imax)

    if rest_reference is None:
        if reference_solution is None:
            raise ValueError(
                "pass rest_reference or reference_solution: the guard has to "
                "be written against a stated level, not inferred")
        rest_reference = _rms(R_r, u_r, res_r,
                              np.asarray(reference_solution, dtype=float))
    rest_reference = float(rest_reference)
    cap = rest_reference * (1.0 + float(budget))

    # The floor: no choice of x gets the rest below this. If the cap is under
    # it the request is impossible, and saying so beats returning the closest
    # thing and calling it guarded.
    x_floor, _ = _solve_ls(R_r, u_r, lo, hi, name="rest_floor")
    rest_floor = _rms(R_r, u_r, res_r, x_floor)
    if cap < rest_floor - GUARD_TOL_HZ:
        raise GuardViolation(
            "rest cap %.4f Hz is below the lowest reachable rest RMS %.4f Hz; "
            "no solution can satisfy it" % (cap, rest_floor))

    x, info = _solve_ls(R_p, u_p, lo, hi,
                        soc=[(R_r, u_r, res_r, cap)],
                        name="pfc_guarded(budget=%.4g)" % budget)

    # Measured, not trusted. `_solve_ls` already reports the violation it can
    # see; this recomputes it from the returned vector against the same cap.
    rest_hz = _rms(R_r, u_r, res_r, x)
    over_box = max(float(np.max(lo - x, initial=0.0)),
                   float(np.max(x - hi, initial=0.0)))
    over_guard = rest_hz - cap
    if over_guard > GUARD_TOL_HZ or over_box > 1e-6:
        raise GuardViolation(
            "solution violates its constraints and will not be clipped: "
            "rest %.4f Hz over cap %.4f Hz by %.3e, box by %.3e"
            % (rest_hz, cap, over_guard, over_box))

    resid = _residual(b0, d, x)
    row = metric_row("guarded(budget=%.4g)" % budget, resid,
                     d["pfc"], d["rest"], d["brain"])
    row.update({
        "budget": float(budget),
        "rest_reference_hz": rest_reference,
        "rest_cap_hz": float(cap),
        "rest_floor_hz": float(rest_floor),
        "rest_floor_margin_hz": float(rest_reference - rest_floor),
        "guard_slack_used_hz": float(cap - rest_hz),
        "scanner_sh_delta": x[:n_sh].copy(),
        "coil_currents": x[n_sh:].copy(),
        "current_max": float(np.max(np.abs(x[n_sh:]))) if n_ch else 0.0,
        "at_limit": int(np.sum(np.abs(np.abs(x[n_sh:]) - abs(imax)) < 1e-3))
                    if n_ch else 0,
        "sh_at_limit": int(np.sum((x[:n_sh] - lo[:n_sh] < 1e-6)
                                  | (hi[:n_sh] - x[:n_sh] < 1e-6))),
        "solver": info["solver"], "status": info["status"],
        "violation": float(max(over_guard, over_box, 0.0)),
    })
    log("  budget %.4g -> PFC %.2f Hz, rest %.2f / cap %.2f Hz"
        % (budget, row["pfc_rms"], row["rest_rms"], cap))
    return row


def pareto_front(b0, brain_mask, pfc_mask, sh_maps, coil_maps,
                 scanner_bounds, imax: float = 3.0,
                 rest_reference: float = None, reference_solution=None,
                 budgets=DEFAULT_BUDGETS, log_cb=None) -> list:
    """`solve_pfc_guarded` at each budget. Budget 0 first, as the result.

    A budget whose solve fails is recorded with its error rather than dropped:
    a missing point on a Pareto front reads as "not worth plotting" when it
    actually means "could not be computed".
    """
    out = []
    for beta in budgets:
        try:
            out.append(solve_pfc_guarded(
                b0, brain_mask, pfc_mask, sh_maps, coil_maps, scanner_bounds,
                imax=imax, rest_reference=rest_reference,
                reference_solution=reference_solution, budget=beta,
                log_cb=log_cb))
        except Exception as e:
            out.append({"config": "guarded(budget=%.4g)" % beta,
                        "budget": float(beta),
                        "error": "%s: %s" % (type(e).__name__, e)})
    return out
