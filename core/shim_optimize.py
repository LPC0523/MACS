"""Shim current optimisation, extracted from the Qt worker as a pure function.

    loss(i) =        mean((A_roi  i + b_roi )^2)                    data
              + l_p  mean((A_rest i + b_rest)^2)                    protect rest
              + l_l  mean((A_rest i          )^2)                   coil leakage
              + l_r  mean((i / current_limit )^2)                   current ridge

    i0 = ones(n)/n,  bounds = +/- current_limit,  L-BFGS-B

The middle two terms answer different questions and are deliberately separate.
`lambda_penalty` protects the *final field* outside the ROI -- it cares that the
rest of the brain ends up homogeneous, and so it must include `b_rest`.
`lambda_leakage` only asks the coils not to radiate outside the ROI, regardless
of what the field there was to begin with. Folding them into one term (which is
what a single `mean((A_rest i)^2)` labelled "非ROI约束" was doing) means the
number cannot be interpreted: reducing it might mean the rest of the brain got
better, or merely that the coils went quiet.

`outside_mask` defines "rest". It must be the brain: every voxel outside the ROI
would put almost the whole penalty on air, where no shimming requirement exists,
and would make the solution depend on the field of view -- change the crop and
the currents change though neither head nor coils did. Whole-brain shimming
leaves `rest` empty, and both rest terms are then correctly skipped.

`lambda_ridge` is normalised twice over: by `current_limit`, so it means the
same for a 3 A array as for a 1.5 A one, and by the ROI's pre-shim mean square,
so it means the same on a 50 Hz field as on a 150 Hz one. lambda_ridge = 0.01
therefore reads as "at full-scale current the ridge is worth 1% of the field we
started with", which is comparable across subjects. Without the second
normalisation the term is a fixed number against a data term that varies by
orders of magnitude, and the same lambda does nothing on one subject and
dominates on the next. See `sweep_ridge` for choosing it.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["optimize_currents", "evaluate_masks", "sweep_ridge",
           "DEFAULT_RIDGE", "RIDGE_LADDER", "RIDGE_TOL_HZ"]

#: Frozen from `tests/test_ridge_sweep.py` across the batch -- see that script
#: for the evidence. Not tuned on a single subject.
DEFAULT_RIDGE = 0.0

#: Geometric ladder for `sweep_ridge`. Wide enough to reach "currents almost
#: zero" at the top, so the knee is always bracketed.
RIDGE_LADDER = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0)

#: How much ROI RMSE may degrade, in Hz, in exchange for smaller currents.
RIDGE_TOL_HZ = 1.0


def optimize_currents(b0: np.ndarray, mask: np.ndarray, coil_maps: np.ndarray,
                      lambda_penalty: float = 0.5,
                      current_limit: float = 3.0,
                      log_cb=None,
                      bounds=None, initial=None, unit: str = "A",
                      outside_mask=None,
                      lambda_leakage: float = 0.0,
                      lambda_ridge: float = None) -> dict:
    """Least-squares shim currents.

    b0            : (nx, ny, nz) target field in Hz
    mask          : (nx, ny, nz) ROI, >0.5 is inside
    coil_maps     : (nx, ny, nz, n_channels) Hz per amp
    outside_mask  : where "rest" lives -- pass the brain mask. `rest` is then
                    `brain & ~roi`, and is empty for whole-brain shimming.
    lambda_penalty: weight on the rest's *final* field (protect the rest)
    lambda_leakage: weight on the coils' field in the rest alone (no b_rest)
    lambda_ridge  : normalised current ridge; None uses `DEFAULT_RIDGE`

    `bounds` and `initial` exist so the same objective can solve for spherical
    harmonic deltas, whose limits are per-term and asymmetric rather than one
    symmetric current ceiling.
    """
    log = log_cb or (lambda *_: None)
    n_ch = coil_maps.shape[-1]
    lambda_ridge = DEFAULT_RIDGE if lambda_ridge is None else float(lambda_ridge)

    b_flat = b0.flatten()
    roi_idx = mask.flatten() > 0.5

    if outside_mask is None:
        # Legacy behaviour, kept only for callers that have no brain mask to
        # give. Everything in the production path passes one.
        rest_idx = ~roi_idx
        rest_label = "含脑外"
    else:
        rest_idx = (np.asarray(outside_mask).flatten() > 0.5) & ~roi_idx
        rest_label = "brain & ~roi"

    b = b_flat[roi_idx].astype(np.float64)
    flat_maps = coil_maps.reshape(-1, n_ch)
    A = flat_maps[roi_idx].astype(np.float64)
    A_rest = flat_maps[rest_idx].astype(np.float64)
    b_rest = b_flat[rest_idx].astype(np.float64)
    n_rest = A_rest.shape[0]

    lam_p = float(lambda_penalty)
    lam_l = float(lambda_leakage)
    if n_rest == 0:
        # Whole-brain shimming: there is no "other brain region" to protect or
        # to keep the coils out of. The mean of an empty set is NaN, which would
        # poison the objective and drive the solver to near-zero current, so
        # both rest terms are dropped rather than evaluated.
        if lam_p or lam_l:
            log("note: rest 区域为空（ROI 即全脑），保护项与泄漏项均不参与")
        lam_p = lam_l = 0.0

    log(f"ROI: {len(b):,} voxels | rest: {n_rest:,} ({rest_label}) | "
        f"{n_ch} channels")

    lim2 = max(float(current_limit) ** 2, 1e-12)
    # Scale the ridge to the problem: at full-scale current it is worth
    # `lambda_ridge` times the energy the ROI started with.
    ridge_scale = float(np.mean(b ** 2)) if b.size else 0.0
    ridge_k = lambda_ridge * ridge_scale / lim2

    def loss_mse(w):
        total = np.mean((A @ w + b) ** 2)
        if lam_p:
            total += lam_p * np.mean((A_rest @ w + b_rest) ** 2)
        if lam_l:
            total += lam_l * np.mean((A_rest @ w) ** 2)
        if ridge_k:
            total += ridge_k * np.mean(w ** 2)
        return total

    if bounds is None:
        bounds = [(-current_limit, current_limit)] * n_ch
        log(f"Optimising (L-BFGS-B, λ_protect={lam_p}, λ_leak={lam_l}, "
            f"λ_ridge={lambda_ridge}, |I| <= {current_limit} {unit})...")
    else:
        bounds = [tuple(map(float, x)) for x in bounds]
        if len(bounds) != n_ch:
            raise ValueError(f"bounds has {len(bounds)} entries for {n_ch} variables")
        log(f"Optimising (L-BFGS-B, λ_protect={lam_p}, per-term bounds)...")

    initial_weights = (np.ones(n_ch) / n_ch if initial is None
                       else np.asarray(initial, dtype=float))
    # A supplied start can sit outside a narrowed window; L-BFGS-B would then
    # begin on the boundary and can fail to leave it.
    initial_weights = np.array([min(max(v, lo), hi)
                                for v, (lo, hi) in zip(initial_weights, bounds)])

    # Explicit tolerances. On scipy's defaults this stopped after 25 iterations
    # on at least one subject, at a loss 4% above the optimum -- and nothing in
    # the output said so, because a converged-looking answer and an abandoned
    # one are the same shape. The objective is cheap here, so there is no reason
    # to leave the stopping rule to chance.
    res = minimize(loss_mse, initial_weights, bounds=bounds, method="L-BFGS-B",
                   options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12,
                            "maxls": 60})
    currents = res.x
    if not res.success:
        log(f"note: solver reported '{res.message}' after {res.nit} iterations")

    residual = b0 + np.tensordot(coil_maps, currents, axes=([3], [0]))
    stats = evaluate_masks(b0, residual, mask)

    at_bound = int(sum(
        min(abs(c - lo), abs(c - hi)) < 1e-3 for c, (lo, hi) in zip(currents, bounds)))
    if at_bound:
        log(f"note: {at_bound}/{n_ch} variable(s) sit at their limit "
            "-- the solution is constrained, not converged freely")

    log(f"Done. ROI RMSE {stats['roi_rmse_before']:.1f} -> "
        f"{stats['roi_rmse_after']:.1f} Hz")

    return {
        "currents": currents,
        "residual": residual,
        "stats": stats,
        "loss": float(loss_mse(currents)),
        "n_iter": int(res.nit),
        "success": bool(res.success),
        "channels_at_limit": at_bound,
        "rest_voxels": int(n_rest),
        "lambda_ridge": float(lambda_ridge),
    }


def sweep_ridge(b0, mask, coil_maps, *, ladder=None, tol_hz: float = None,
                log_cb=None, **kwargs) -> dict:
    """Pick the ridge that costs the least current for no real loss of homogeneity.

    Without a ridge the solve is free to use large, partly cancelling currents
    whenever two channels overlap: the objective cannot tell that apart from a
    small solution, but the amplifier and the heat can. A ridge that is guessed
    instead of measured is no better -- how much current buys the last Hz is a
    property of this subject and this array.

    So: solve along a ladder of ridge weights, then take the *largest* ridge
    (smallest currents) whose ROI RMSE is still within `tol_hz` of the best any
    ridge achieved. The reported numbers make the trade auditable rather than
    implicit.
    """
    log = log_cb or (lambda *_: None)
    ladder = tuple(RIDGE_LADDER if ladder is None else ladder)
    tol = RIDGE_TOL_HZ if tol_hz is None else float(tol_hz)
    limit = float(kwargs.get("current_limit", 3.0))

    trials = []
    for lam in ladder:
        r = optimize_currents(b0, mask, coil_maps, lambda_ridge=lam,
                              log_cb=None, **kwargs)
        cur = r["currents"]
        trials.append({
            "lambda_ridge": float(lam),
            "roi_rmse": float(r["stats"]["roi_rmse_after"]),
            "roi_std": float(r["stats"]["roi_std_after"]),
            "current_rms": float(np.sqrt(np.mean(cur ** 2))),
            "current_max": float(np.max(np.abs(cur))),
            "at_limit": int(r["channels_at_limit"]),
            "result": r,
        })

    best = min(t["roi_rmse"] for t in trials)
    ok = [t for t in trials if t["roi_rmse"] <= best + tol]
    chosen = max(ok, key=lambda t: t["lambda_ridge"])
    no_ridge = next(t for t in trials if t["lambda_ridge"] == 0.0)

    log(f"  ridge 扫参：无 ridge RMSE {no_ridge['roi_rmse']:.2f} Hz，"
        f"最优 {best:.2f} Hz，容差 {tol:g} Hz")
    log(f"  选定 λ_ridge={chosen['lambda_ridge']:g}："
        f"RMSE {chosen['roi_rmse']:.2f} Hz（退化 "
        f"{chosen['roi_rmse'] - best:+.2f}），电流 RMS "
        f"{no_ridge['current_rms']:.2f} → {chosen['current_rms']:.2f} A，"
        f"最大 {no_ridge['current_max']:.2f} → {chosen['current_max']:.2f} A，"
        f"饱和 {no_ridge['at_limit']} → {chosen['at_limit']}")

    return {
        "chosen": chosen,
        "no_ridge": no_ridge,
        "best_rmse": float(best),
        "degradation_hz": float(chosen["roi_rmse"] - best),
        "tol_hz": tol,
        "trials": [{k: v for k, v in t.items() if k != "result"} for t in trials],
        "result": chosen["result"],
    }


def evaluate_masks(b0: np.ndarray, residual: np.ndarray,
                   roi_mask: np.ndarray) -> dict:
    """Before/after RMSE, std and peak-to-peak in the ROI and over the brain."""
    roi = roi_mask > 0.5
    brain = b0 != 0

    def rmse(v, m):
        return float(np.sqrt(np.mean(v[m] ** 2))) if m.any() else 0.0

    def std(v, m):
        return float(np.std(v[m])) if m.any() else 0.0

    def ptp(v, m):
        return float(np.ptp(v[m])) if m.any() else 0.0

    return {
        "roi_rmse_before": rmse(b0, roi),      "roi_rmse_after": rmse(residual, roi),
        "roi_std_before":  std(b0, roi),       "roi_std_after":  std(residual, roi),
        "roi_ptp_before":  ptp(b0, roi),       "roi_ptp_after":  ptp(residual, roi),
        "brain_rmse_before": rmse(b0, brain),  "brain_rmse_after": rmse(residual, brain),
        "brain_std_before":  std(b0, brain),   "brain_std_after":  std(residual, brain),
    }
