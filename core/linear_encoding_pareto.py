"""How much gradient can the array actually deliver? An SOCP.

IDEALIZED OFFLINE SIMULATION.

The program
-----------
    maximise    beta_actual = d^T G i

    subject to  d^T G i >= 0
                ||(I - d d^T) G i||      <= tan(theta_max) * d^T G i
                sqrt(mean_W[(R i)^2])    <= epsilon_nonlinear * (d^T G i)
                                            * sigma_position
                lower_j <= i_j <= upper_j

There is no independent requested or nominal beta anywhere in this module, and
none may be reintroduced. An earlier version maximised a free amplitude against
a relative fit error; that objective rewarded plain under-delivery -- a field
of 1200 Hz/m sits within 10% of a 1333 Hz/m target -- and could prefer a
current pattern delivering 900 at gain 0.7 over one delivering 1000 at gain
1.0. The objective is the gradient the coils produce.

`epsilon_nonlinear` bounds the nonlinear spatial component alone, relative to
the linear field the delivered gradient makes over this region.
`theta_max_deg` bounds the direction error separately. They are different
failures with different fixes, and an overall NRMSE cannot tell them apart:
with the cone dropped, the optimum on the 12ch PFC ROI tilted 74 degrees with
an orthogonal leakage 3.5x the parallel component while every scalar fit
metric still looked reasonable.

`beta_ray_limited` is kept as a labelled baseline: it scales the minimum-norm
unit-gradient pattern along one fixed ray in current space until a channel
reaches its bound. It is a witness, not a ceiling -- the SOCP is free to pick a
different current direction and usually does better.

Size
----
The design has up to a million rows. The nonlinear residual reaches the solver
only through its J x J weighted Gram factor, so CVXPY sees a cone of size J
rather than 216,000.

No post-solve clipping. Bounds are constraints; a returned vector that violates
any of them is reported as a numerical failure, never trimmed into compliance.
"""

from __future__ import annotations

import numpy as np

from core.linear_encoding_metrics import encoding_metrics, to_mt_per_m
from core.spatial_grid import direction_field, unit_vector

__all__ = ["EPSILON_LADDER", "THETA_LADDER", "dynamic_bounds",
           "beta_ray_limited", "max_gradient_socp", "position_sigma",
           "pareto_frontier", "SOLVER_TOL", "MAX_CONE_DEG"]

#: What a returned vector must satisfy before it may be called feasible. A
#: solver status of `optimal_inaccurate` is not evidence; these are measured on
#: the vector itself.
#: Widest cone accepted. Beyond this the constraint is vacuous and tan()
#: starts to blow up; "no cone" is theta_max_deg=None.
MAX_CONE_DEG = 45.0

SOLVER_TOL = {
    "box_violation_a": 1e-6,
    "nonlinear_constraint_relative_violation": 1e-3,
    "direction_cone_relative_violation": 1e-3,
}

#: Relative shape errors to sweep. Reported as infeasible where they are.
EPSILON_LADDER = (0.20, 0.30, 0.40, 0.50)

#: Optional direction-cone half angles, in degrees.
THETA_LADDER = (2.0, 5.0, 10.0)


def dynamic_bounds(n_channels, limit_a=3.0, i_static=None, symmetric=True):
    """(lower, upper) per-channel bounds on the *dynamic* current.

    With `symmetric=True` the encoding waveform is assumed bipolar and centred:
    the amplifier must survive `i_static + i` and `i_static - i` alike, so the
    usable magnitude is `Imax - |i_static|` on both sides. This is the case for
    a sinusoid and it is the default.

    With `symmetric=False` the window is the raw asymmetric one,
    `[-Imax - i_static, Imax - i_static]`, which suits a single-sign blip. That
    window is wider on one side and a caller using it must check the actual
    waveform sample by sample -- it does not by itself guarantee that a
    two-sided excursion stays legal.
    """
    J = int(n_channels)
    lim = (np.full(J, float(limit_a)) if not np.ndim(limit_a)
           else np.abs(np.asarray(limit_a, dtype=np.float64).ravel()))
    base = (np.zeros(J) if i_static is None
            else np.asarray(i_static, dtype=np.float64).ravel())
    if base.shape[0] != J:
        raise ValueError("static currents have %d entries for %d channels"
                         % (base.shape[0], J))
    if np.any(np.abs(base) > lim + 1e-12):
        raise ValueError("static baseline already exceeds the current limit")
    if symmetric:
        room = lim - np.abs(base)
        return -room, room
    return -lim - base, lim - base


def beta_ray_limited(model, direction, limit_a=3.0, i_static=None,
                     symmetric=True, ridge=0.0) -> dict:
    """The baseline: scale the minimum-norm unit pattern until a channel stops.

    Named for what it is. This is the largest amplitude *along one fixed ray in
    current space*, and calling it the array's maximum would overstate it --
    `max_gradient_socp` is free to choose a different current vector and
    generally finds more.
    """
    a, info = model.unit_gradient_currents(direction, ridge=ridge)
    lo, hi = dynamic_bounds(model.n_channels, limit_a, i_static, symmetric)
    s = np.inf
    for j in range(a.size):
        if a[j] > 0:
            s = min(s, hi[j] / a[j])
        elif a[j] < 0:
            s = min(s, lo[j] / a[j])
    beta = float(max(0.0, s if np.isfinite(s) else 0.0))
    i_pk = a * beta
    m = encoding_metrics(model, i_pk, direction=direction,
                         requested_gradient_hz_per_m=beta)
    tight = int(np.argmax(np.abs(a))) if a.size else -1
    return {
        "method": "beta_min_norm_ray_limit",
        "description": ("Maximum amplitude obtained by scaling the "
                        "minimum-norm unit-gradient current pattern along a "
                        "fixed current-space ray."),
        "beta_ray_limited_hz_per_m": beta,
        "beta_ray_limited_mt_per_m": float(to_mt_per_m(beta)),
        "achieved_gradient_hz_per_m": m["gradient_along_target_hz_per_m"],
        "achieved_gradient_mt_per_m": float(to_mt_per_m(
            m["gradient_along_target_hz_per_m"])),
        "gradient_gain": m.get("gradient_gain", float("nan")),
        "unit_currents": a.tolist(),
        "peak_currents": i_pk.tolist(),
        "binding_channel": (model.channel_names[tight] if tight >= 0 else ""),
        "metrics": m,
    }


def _residual_factor(model):
    """(L, info) with ||L i|| == sqrt(mean_W[(R i)^2]) up to a stated clip.

    The weighted Gram of the nonlinear residual is symmetric positive
    semidefinite in exact arithmetic; in floating point its smallest
    eigenvalues come out slightly negative. An earlier version added a jitter
    to force a Cholesky through, which works but makes the factor a
    regularised approximation while the docstring called it exact.

    Here the spectrum is taken directly and only numerically-negative
    eigenvalues are clipped to zero, which is the smallest change that makes
    the matrix a valid Gram. What was clipped is reported rather than
    swallowed: a genuinely negative eigenvalue would mean the residual
    decomposition itself is wrong, and that should be visible.
    """
    w = model.weights
    Rw = model.residual * np.sqrt(w)[:, None]
    Q = (Rw.T @ Rw) / model.w_sum
    Q = 0.5 * (Q + Q.T)
    lam, V = np.linalg.eigh(Q)
    scale = max(float(np.abs(lam).max()), 1e-300)
    tol = 1e-12 * scale
    clipped = int(np.sum(lam < 0))
    lam_c = np.clip(lam, 0.0, None)
    L = (np.sqrt(lam_c)[:, None]) * V.T
    info = {
        "min_eigenvalue": float(lam.min()),
        "max_eigenvalue": float(lam.max()),
        "clipped_eigenvalue_count": clipped,
        "clipping_tolerance": float(tol),
        "regularisation": "eigenvalue clipping at zero (no jitter added)",
        "suspicious": bool(lam.min() < -tol),
    }
    return L, info


def position_sigma(model, direction):
    """sqrt(mean_W[(d.(r-r0))^2]) -- the ROI's spread along `direction`, metres.

    This is what turns a relative nonlinearity budget into a field in Hz: a
    gradient of `beta` over a region of spread `sigma` produces a linear field
    of RMS `beta * sigma`, and the residual is measured against that.
    """
    d = unit_vector(direction)
    t = direction_field(model.points, d, centre=model.centre)
    w = model.weights
    t = t - float(w @ t) / model.w_sum
    return float(np.sqrt(float(w @ (t ** 2)) / model.w_sum))


def max_gradient_socp(model, direction, epsilon, limit_a=3.0, i_static=None,
                      symmetric=True, theta_max_deg=None,
                      solvers=("CLARABEL", "SCS")) -> dict:
    """Largest *delivered* gradient along `direction`, at a stated nonlinearity.

        maximise    d'G i
        subject to  d'G i >= 0
                    ||(I - dd')G i||        <= tan(theta_max) * d'G i
                    sqrt(mean_W[(R i)^2])   <= epsilon * (d'G i) * sigma
                    lower <= i <= upper

    There is no free nominal amplitude. An earlier version maximised a separate
    `beta` against a relative fit error, and that objective could be inflated by
    plain under-delivery: a field of 1200 Hz/m sits within 10% of a 1333 Hz/m
    target, so the solver was rewarded for claiming 1333. Worse than a cosmetic
    naming problem, it changed the answer -- given a choice between delivering
    1000 at unit gain and 900 at gain 0.7, the old objective preferred the 900.
    The optimum here is the gradient the coils actually produce.

    `epsilon` now bounds the nonlinear residual *only*, relative to the linear
    field the delivered gradient makes over this region. Direction error is
    bounded separately by `theta_max_deg`, because the two are different
    failures with different fixes and an overall NRMSE cannot tell them apart.
    """
    import cvxpy as cp

    d = unit_vector(direction)
    G = model.G                                   # (3, J) Hz/m/A
    L, resid_factor_info = _residual_factor(model)   # (J, J)
    sigma = position_sigma(model, d)              # metres
    lo, hi = dynamic_bounds(model.n_channels, limit_a, i_static, symmetric)

    i = cp.Variable(model.n_channels)
    g_par = (d @ G) @ i
    P = np.eye(3) - np.outer(d, d)

    cons = [i >= lo, i <= hi, g_par >= 0,
            cp.norm(L @ i) <= float(epsilon) * sigma * g_par]
    if theta_max_deg is not None:
        th = float(theta_max_deg)
        # tan(90 deg) is not a number, and a cone approaching it constrains
        # nothing. "No cone at all" is a separate, meaningful choice and is
        # spelled theta_max_deg=None rather than theta_max_deg=90.
        if not (0.0 < th < 90.0):
            raise ValueError(
                "theta_max_deg must lie strictly between 0 and 90; pass None "
                "for no direction cone (got %r)" % (theta_max_deg,))
        if th > MAX_CONE_DEG:
            raise ValueError(
                "theta_max_deg %.3g exceeds %.0f deg, past which the cone "
                "stops constraining anything useful; pass None instead"
                % (th, MAX_CONE_DEG))
        cons.append(cp.norm((P @ G) @ i)
                    <= float(np.tan(np.deg2rad(th))) * g_par)

    prob = cp.Problem(cp.Maximize(g_par), cons)

    def _verify(vec):
        """Measured on the returned vector. A solver status is not evidence."""
        gv = G @ vec
        pa = float(gv @ d)
        orth_ = float(np.linalg.norm(gv - pa * d))
        box = max(float(np.max(lo - vec, initial=0.0)),
                  float(np.max(vec - hi, initial=0.0)))
        rr = float(np.linalg.norm(L @ vec))
        nl_rhs = float(epsilon) * sigma * pa
        nl = max(0.0, (rr - nl_rhs) / max(nl_rhs, 1e-30))
        th_v = 0.0
        if theta_max_deg is not None:
            rhs = float(np.tan(np.deg2rad(float(theta_max_deg)))) * pa
            th_v = max(0.0, (orth_ - rhs) / max(rhs, 1e-30))
        floor = 1e-9 * max(float(np.abs(G).max()) * float(np.abs(hi).max()), 1.0)
        finite = bool(np.all(np.isfinite(vec)))
        above = bool(pa > floor)
        return {
            "par": pa, "orth": orth_, "resid_rms": rr,
            "box_violation_a": box,
            "nonlinear_constraint_relative_violation": nl,
            "direction_cone_relative_violation": th_v,
            "finite": finite, "above_floor": above,
            "ok": bool(finite and above
                       and box <= SOLVER_TOL["box_violation_a"]
                       and nl <= SOLVER_TOL[
                           "nonlinear_constraint_relative_violation"]
                       and th_v <= SOLVER_TOL[
                           "direction_cone_relative_violation"]),
        }

    status, used, chk, best = None, None, None, None
    attempts = []
    for sname in solvers:
        try:
            prob.solve(solver=sname)
            st = prob.status
        except Exception as e:
            attempts.append({"solver": sname,
                             "status": "error: %s" % type(e).__name__})
            continue
        if i.value is None:
            attempts.append({"solver": sname, "status": st})
            continue
        v = np.asarray(i.value, dtype=np.float64).ravel()
        c = _verify(v)
        attempts.append({"solver": sname, "status": st,
                         "box": c["box_violation_a"],
                         "nonlinear": c["nonlinear_constraint_relative_violation"],
                         "cone": c["direction_cone_relative_violation"],
                         "accepted": c["ok"]})
        # `optimal_inaccurate` is neither a reason to accept nor to reject.
        # What decides is whether the vector satisfies its own constraints, so
        # an inaccurate-but-valid answer is kept and an optimal-but-violating
        # one moves on to the next solver.
        if c["ok"]:
            status, used, chk, best = st, sname, c, v
            break
        if best is None:
            status, used, chk, best = st, sname, c, v

    out = {"method": "max_delivered_gradient_socp",
           "epsilon_nonlinear": float(epsilon),
           "theta_max_deg": theta_max_deg, "solver": used, "status": status,
           "symmetric_bipolar_bounds": bool(symmetric),
           "position_sigma_m": sigma,
           "feasible": False, "gradient_parallel_actual_hz_per_m": 0.0}
    out["solver_attempts"] = attempts
    out["residual_factor"] = resid_factor_info

    if best is None:
        out["status"] = "numerical_failure"
        out["note"] = ("no solver returned a vector for epsilon=%.2f%s"
                       % (epsilon, "" if theta_max_deg is None
                          else " and theta<=%.0f deg" % theta_max_deg))
        return out

    if not chk["ok"]:
        if (chk["finite"] and not chk["above_floor"]
                and chk["box_violation_a"] <= SOLVER_TOL["box_violation_a"]):
            # A real "only the zero current is feasible", not a failure.
            out["shape_reachable"] = False
            out["status"] = status
            out["solver"] = used
            out["note"] = ("only the zero current satisfies epsilon=%.2f on "
                           "this region: the array cannot make a field this "
                           "close to linear along this direction" % epsilon)
            return out
        out["status"] = "numerical_failure"
        out["solver"] = used
        out["note"] = ("every solver returned a vector violating its own "
                       "constraints (box %.2e A, nonlinear %.2e, cone %.2e). "
                       "Reporting failure rather than a solution that is not "
                       "one." % (chk["box_violation_a"],
                                 chk["nonlinear_constraint_relative_violation"],
                                 chk["direction_cone_relative_violation"]))
        out["box_violation_a"] = chk["box_violation_a"]
        out["nonlinear_constraint_relative_violation"] = chk[
            "nonlinear_constraint_relative_violation"]
        out["direction_cone_relative_violation"] = chk[
            "direction_cone_relative_violation"]
        return out

    iv = best
    g_vec = G @ iv
    par, orth = chk["par"], chk["orth"]
    resid_rms = chk["resid_rms"]
    box_viol = chk["box_violation_a"]
    nl_viol = chk["nonlinear_constraint_relative_violation"]
    th_viol = chk["direction_cone_relative_violation"]
    out["shape_reachable"] = True
    out["status"] = status
    out["solver"] = used
    m = encoding_metrics(model, iv, direction=d,
                         requested_gradient_hz_per_m=par)

    out.update({
        "feasible": True,
        # ── primary metrics ──────────────────────────────────────────
        "gradient_parallel_actual_hz_per_m": par,
        "gradient_parallel_actual_mt_per_m": float(to_mt_per_m(par)),
        "gradient_magnitude_actual_hz_per_m": float(np.linalg.norm(g_vec)),
        "gradient_magnitude_actual_mt_per_m": float(to_mt_per_m(
            np.linalg.norm(g_vec))),
        "direction_error_deg": float(np.degrees(np.arctan2(orth, par))),
        "orthogonal_leakage_ratio": orth / par,
        "orthogonal_leakage_hz_per_m": orth,
        "nonlinear_residual_ratio": resid_rms / max(par * sigma, 1e-30),
        "nonlinear_residual_rms_hz": resid_rms,
        # ── secondary ────────────────────────────────────────────────
        "target_fit_nrmse": m["nrmse"],
        "_target_fit_nrmse_note": ("includes orthogonal linear leakage and "
                                   "nonlinear residual together"),
        "r_squared": m["r_squared"], "correlation": m["correlation"],
        "nonlinear_p95_hz": m["nonlinear_p95_hz"],
        "nonlinear_max_hz": m["nonlinear_max_hz"],
        "constant_offset_hz": m["constant_offset_hz"],
        "linear_fraction": m["linear_fraction"],
        "currents_a": iv.tolist(),
        "current_peak_a": float(np.abs(iv).max()),
        "current_rms_a": float(np.sqrt(np.mean(iv ** 2))),
        "saturated_channels": [model.channel_names[j] for j in range(iv.size)
                               if abs(iv[j] - hi[j]) < 1e-6
                               or abs(iv[j] - lo[j]) < 1e-6],
        "bounds_lower_a": lo.tolist(), "bounds_upper_a": hi.tolist(),
        "box_violation_a": box_viol,
        "nonlinear_constraint_relative_violation": nl_viol,
        "direction_cone_relative_violation": th_viol,
        "clipping_applied": False,
        # ── legacy / debug only, never a headline ────────────────────
        "_legacy_requested_beta_hz_per_m": par,
    })
    return out


def pareto_frontier(model, direction, epsilons=EPSILON_LADDER, limit_a=3.0,
                    i_static=None, symmetric=True, theta_max_deg=None) -> list:
    """The ray baseline plus `max_gradient_socp` across an epsilon ladder.

    The ray point is what a fixed minimum-norm pattern delivers; every SOCP
    point is free to choose a different current direction and should never
    deliver less once its nonlinearity budget covers the ray's own residual.
    """
    ray = beta_ray_limited(model, direction, limit_a, i_static, symmetric)
    rm = ray["metrics"]
    sigma = position_sigma(model, direction)
    par = rm["gradient_along_target_hz_per_m"]
    rows = [{
        "method": "beta_min_norm_ray_limit", "epsilon_nonlinear": None,
        "theta_max_deg": None, "feasible": True, "shape_reachable": True,
        "gradient_parallel_actual_hz_per_m": par,
        "gradient_parallel_actual_mt_per_m": float(to_mt_per_m(par)),
        "gradient_magnitude_actual_hz_per_m": rm["achieved_gradient_magnitude_hz_per_m"],
        "gradient_magnitude_actual_mt_per_m": rm["achieved_gradient_magnitude_mt_per_m"],
        "direction_error_deg": rm["direction_error_deg"],
        "orthogonal_leakage_ratio": (rm["orthogonal_leakage_hz_per_m"]
                                     / max(par, 1e-30)),
        "orthogonal_leakage_hz_per_m": rm["orthogonal_leakage_hz_per_m"],
        "nonlinear_residual_rms_hz": rm["nonlinear_rms_hz"],
        "nonlinear_residual_ratio": (rm["nonlinear_rms_hz"]
                                     / max(par * sigma, 1e-30)),
        "target_fit_nrmse": rm["nrmse"], "r_squared": rm["r_squared"],
        "correlation": rm["correlation"],
        "nonlinear_p95_hz": rm["nonlinear_p95_hz"],
        "nonlinear_max_hz": rm["nonlinear_max_hz"],
        "constant_offset_hz": rm["constant_offset_hz"],
        "linear_fraction": rm["linear_fraction"],
        "current_peak_a": rm["current_peak_a"],
        "current_rms_a": rm["current_rms_a"],
        "currents_a": ray["peak_currents"],
        "saturated_channels": [ray["binding_channel"]],
        "position_sigma_m": sigma,
        "status": "closed-form", "solver": "lstsq",
        "box_violation_a": 0.0,
        "nonlinear_constraint_relative_violation": 0.0,
        "direction_cone_relative_violation": 0.0,
        "clipping_applied": False,
    }]
    for eps in epsilons:
        rows.append(max_gradient_socp(model, direction, eps, limit_a, i_static,
                                      symmetric, theta_max_deg))
    return rows
