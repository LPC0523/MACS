"""How much of the PFC field can this hardware actually reach? A parameter-free audit.

Before asking which optimiser is best it is worth knowing how much room any
optimiser has. This measures that: the best each subset of the hardware can do
under its real limits, and how far the shipping solution sits from the best any
solution could.

Parameterisation
----------------
Everything here uses the *direct* form

    r = b + S s + C i + f

where `s` is the scanner command itself, `i` the coil currents and `f` a single
free centre frequency. The projected form used inside `adaptive_joint` --
`b + S theta + C_perp i` with `scanner = theta - K i` -- describes the same
physics, since `S theta + (C - S K) i == S(theta - K i) + C i`. It is a
reparameterisation that helps a box-constrained solver separate the variables,
and nothing more.

The direct form is used here deliberately. In the projected form the scanner's
limit is a constraint on `theta - K i`, not on `theta`, and applying it to
`theta` -- which is what the production solver does -- can understate the
command by up to 2.6x on this array. Auditing in the same parameterisation
would inherit that error and overstate what the hardware can reach. In the
direct form the scanner limit is a plain box on `s`, and cannot be got wrong.

High-order controllability
--------------------------
`P` projects onto the span of the SH basis *restricted to the PFC*. Because
`S s` lies entirely in that span, the high-order part of the residual

    (I - P)(b + S s + C i) = h + C_perp i,    h = (I-P)b,  C_perp = (I-P)C

does not depend on `s` at all. So "how much high-order residual can the coils
remove" is a question about `i` alone -- but only if the scanner is free to
absorb the low-order field the coils emit alongside it. Realising `C_perp i`
means the scanner must cancel `P C i`, and that command has to be inside the
console's limits. Bounding only `|i| <= 3 A` and ignoring that would report a
controllability the machine cannot deliver, which is why `high_full` exists
next to `high_bounded`.

Scope of the optimality claims
------------------------------
Pareto optimality applies only to regional MSE/RMS under the identical
feasible set; it does not imply Pareto optimality for CVaR, EPI displacement,
signal dropout, or measured EPI image quality. Every bound below is a bound on
a mean square over a mask, and on nothing else.

Solvers
-------
The design matrices have a million rows and about twenty columns, so every
problem is reduced to its normal equations first -- `H = A'A/n`, `g = A'b/n`,
`c = b'b/n` -- and only the small quadratic form is handed to CVXPY. Feasibility
and the primal constraint violation are checked on every solve; a solution that
merely reports `optimal` is not taken at its word.
"""

from __future__ import annotations

import numpy as np

from core.hybrid_shim import SH12, build_sh_basis
from core.shim_metrics import regional_metrics

__all__ = ["audit_subject", "CONFIGS", "DEFAULT_SCANNER_LIMIT"]

#: The deployment envelope per SH term, in the app's units, applied to both
#: the delta and the absolute setting. A deliberately conservative limit inside
#: what the scanner accepts -- see `core/shim_state`. It is expected to bind:
#: an SH-only fit to the PFC alone asks for B21 of 3000-4400 on this cohort, so
#: `high_full` differing from `high_bounded` is the envelope being enforced,
#: not a configuration problem.
DEFAULT_SCANNER_LIMIT = 2000.0

#: Feasibility tolerance. Reported, never silently absorbed.
VIOL_TOL = 1e-6

#: How much the guarded bound may let the rest of the brain degrade, as a
#: fraction of the production solution's rest RMS. Needed for a non-empty
#: interior; small enough to be physically meaningless.
GUARD_SLACK = 0.01

CONFIGS = (
    "none",              # unshimmed, for scale
    "sh_only",           # scanner alone
    "coil_only",         # coils alone, +-imax
    "high_unconstrained",   # coils vs high-order residual, no limits
    "high_bounded",         # + |i| <= imax
    "high_full",            # + the scanner command that realising it requires
    "production",           # what the app ships today
    "bound_pfc",            # best possible PFC, all constraints
    "bound_whole",          # best possible whole brain, all constraints
    "bound_pfc_guarded",    # best possible PFC with the rest held at production
)


def _normal_form(A, b):
    """(H, g, c) with mean((A x + b)^2) == x'Hx + 2 g'x + c."""
    n = max(A.shape[0], 1)
    return (A.T @ A) / n, (A.T @ b) / n, float(b @ b) / n


def _factor(A, b):
    """(R, u, resid) with mean((A x + b)^2) == ||R x + u||^2 + resid.

    A million-row design reduced to a 20x20 triangular factor. Handing the
    solver `sum_squares(R x + u)` rather than `quad_form(x, H)` matters here:
    the rest-of-brain Gram matrix has a condition number around 1e8, and in
    quadratic form the conic solvers return `optimal_inaccurate` with the guard
    violated by several Hz -- enough for a "bound" to come out worse than the
    solution it is supposed to bound. The factored form halves the exponent on
    the conditioning and the same problems then solve to `optimal`.

    `resid` is the part of the mean-square that no choice of x can remove.
    """
    H, g, c = _normal_form(A, b)
    n = H.shape[0]
    jitter = 1e-14 * max(float(np.trace(H)) / max(n, 1), 1.0)
    for _ in range(10):
        try:
            R = np.linalg.cholesky(H + jitter * np.eye(n)).T
            break
        except np.linalg.LinAlgError:
            jitter *= 100.0
    else:
        raise RuntimeError("design matrix is not positive semidefinite")
    u = np.linalg.solve(R.T, g)
    return R, u, max(c - float(u @ u), 0.0)


def _rms(R, u, resid, x):
    """What a factored form evaluates to at `x`, back in Hz."""
    v = R @ x + u
    return float(np.sqrt(max(float(v @ v) + resid, 0.0)))


def _solve_ls(R, u, lo, hi, extra=None, soc=None, name="ls"):
    """min ||R x + u||^2 over a box, plus optional linear and norm constraints.

    `extra`: list of (M, lo_m, hi_m) meaning lo_m <= M x <= hi_m.
    `soc`:   list of (R_c, u_c, resid_c, cap) meaning the region with that
             factor must stay at or below `cap` Hz RMS.
    Returns (x, info). `info["violation"]` is in Hz and is measured on the
    returned vector, never inferred from the solver's status.
    """
    import cvxpy as cp

    x = cp.Variable(R.shape[1])
    cons = []
    if lo is not None:
        cons.append(x >= np.asarray(lo, dtype=float))
    if hi is not None:
        cons.append(x <= np.asarray(hi, dtype=float))
    for M, ml, mh in (extra or []):
        if ml is not None:
            cons.append(M @ x >= np.asarray(ml, dtype=float))
        if mh is not None:
            cons.append(M @ x <= np.asarray(mh, dtype=float))
    for R_c, u_c, resid_c, cap in (soc or []):
        # ||R_c x + u_c||^2 + resid_c <= cap^2, as a second-order cone.
        rhs = float(cap) ** 2 - float(resid_c)
        if rhs < 0:
            raise RuntimeError(
                "%s: cap %.3f Hz is below the unremovable floor %.3f Hz -- "
                "infeasible by construction" % (name, cap, np.sqrt(resid_c)))
        cons.append(cp.norm(R_c @ x + u_c) <= np.sqrt(rhs))

    prob = cp.Problem(cp.Minimize(cp.sum_squares(R @ x + u)), cons)
    status, used = None, None
    for solver in ("CLARABEL", "SCS"):
        try:
            prob.solve(solver=solver)
            status, used = prob.status, solver
        except Exception as e:
            status, used = "error: %s: %s" % (type(e).__name__, e), solver
            continue
        if x.value is not None and status in ("optimal", "optimal_inaccurate"):
            break

    if x.value is None:
        raise RuntimeError("%s: no solution (%s)" % (name, status))
    xv = np.asarray(x.value, dtype=float).ravel()

    viol = 0.0
    if lo is not None:
        viol = max(viol, float(np.max(np.asarray(lo) - xv, initial=0.0)))
    if hi is not None:
        viol = max(viol, float(np.max(xv - np.asarray(hi), initial=0.0)))
    for M, ml, mh in (extra or []):
        Mx = M @ xv
        if ml is not None:
            viol = max(viol, float(np.max(np.asarray(ml) - Mx, initial=0.0)))
        if mh is not None:
            viol = max(viol, float(np.max(Mx - np.asarray(mh), initial=0.0)))
    for R_c, u_c, resid_c, cap in (soc or []):
        # In Hz. A mean-square left squared would make the tolerance scale with
        # the square of the field and read as a far tighter check than it is.
        viol = max(viol, _rms(R_c, u_c, resid_c, xv) - float(cap))

    return xv, {"status": status, "solver": used, "violation": float(viol)}


def audit_subject(b0, brain_mask, pfc_mask, coil_maps, affine,
                  imax: float = 3.0,
                  scanner_baseline=None, scanner_limits=None,
                  scanner_delta_limits=None, scanner_bounds=None,
                  log_cb=None) -> dict:
    """Every configuration in `CONFIGS` for one subject.

    The scanner box is the intersection of the delta envelope
    (`scanner_delta_limits`, on the change) and the absolute envelope
    (`scanner_limits`, on `baseline + change`) -- the same constraint
    `SystemShimState.delta_bounds` produces, per term and asymmetric whenever
    the baseline is non-zero. Pass `scanner_bounds` as a list of (lo, hi) to
    supply that window directly, e.g. straight from a loaded shim state.

    Returns {"rows": [...metrics per config...], "spectrum": {...}}.
    """
    log = log_cb or (lambda *_: None)

    brain = np.asarray(brain_mask, dtype=bool)
    pfc = np.asarray(pfc_mask, dtype=bool) & brain
    rest = brain & ~pfc
    if not pfc.any():
        raise ValueError("PFC mask does not overlap the brain")

    n_sh = len(SH12)
    n_ch = coil_maps.shape[-1]
    basis = build_sh_basis(affine, b0.shape[:3])

    base = (np.zeros(n_sh) if scanner_baseline is None
            else np.asarray(scanner_baseline, dtype=float))
    lim = (np.full(n_sh, DEFAULT_SCANNER_LIMIT) if scanner_limits is None
           else np.abs(np.asarray(scanner_limits, dtype=float)))
    dlim = (np.full(n_sh, DEFAULT_SCANNER_LIMIT) if scanner_delta_limits is None
            else np.abs(np.asarray(scanner_delta_limits, dtype=float)))

    if scanner_bounds is not None:
        sb = np.asarray(scanner_bounds, dtype=float).reshape(n_sh, 2)
        s_lo, s_hi = sb[:, 0].copy(), sb[:, 1].copy()
    else:
        # The intersection of the two envelopes, exactly as
        # `SystemShimState.delta_bounds` builds it. Asymmetric whenever the
        # baseline is non-zero, which is the normal case on a real console;
        # with a zero baseline it reduces to the symmetric +-limit.
        s_lo = np.maximum(-dlim, -lim - base)
        s_hi = np.minimum(+dlim, +lim - base)
    frozen = s_lo > s_hi
    if frozen.any():
        # Baseline already outside the envelope. Hold the term rather than
        # emit a delta that could not be sent.
        s_lo[frozen] = 0.0
        s_hi[frozen] = 0.0
        log("  %d 项基线已在包络外，已冻结" % int(frozen.sum()))

    flat_coil = coil_maps.reshape(-1, n_ch)

    # f0 is not a free variable here. The reported metric fixes it at the value
    # prescan would pick -- minus the whole-brain mean -- so the objective must
    # use the same one or the "bound" can come out worse than the solution it is
    # supposed to bound. Subtracting each column's in-brain mean builds that f0
    # into the design: `b_c + S_c s + C_c i` IS the residual after the shared f0,
    # for every region, with no extra variable and no way for the two to drift.
    mb_b = float(b0[brain].mean())
    mb_S = np.array([float(basis[n][brain].mean()) for n in SH12])
    mb_C = flat_coil[brain.ravel()].mean(axis=0).astype(np.float64)

    def region(m):
        S = np.stack([basis[n][m] for n in SH12], axis=-1).astype(np.float64) - mb_S
        C = flat_coil[m.ravel()].astype(np.float64) - mb_C
        return S, C, b0[m].astype(np.float64) - mb_b

    S_p, C_p, b_p = region(pfc)
    S_b, C_b, b_b = region(brain)
    S_r, C_r, b_r = region(rest)

    # ── high-order decomposition inside the PFC ──────────────────────
    # The low-order subspace is span{1, SH12} restricted to the PFC. The
    # constant belongs in it: a uniform offset is what the centre frequency is,
    # and leaving it out would count an f0 error as high-order residual the
    # coils get credit for removing.
    ones_p = np.ones((b_p.size, 1))
    ones_b = np.ones((b_b.size, 1))
    L_p = np.hstack([ones_p, S_p])
    Q, _ = np.linalg.qr(L_p)
    h = b_p - Q @ (Q.T @ b_p)
    C_perp = C_p - Q @ (Q.T @ C_p)

    sv = np.linalg.svd(C_perp, compute_uv=False)
    tol_rank = sv.max() * max(C_perp.shape) * np.finfo(float).eps if sv.size else 0.0
    eff_rank = int(np.sum(sv > max(tol_rank, 1e-9 * sv.max()))) if sv.size else 0
    # How much of the uncontrollable part actually lies in the coils' reach.
    Qc, _ = np.linalg.qr(C_perp) if C_perp.size else (np.zeros((h.size, 0)), None)
    h_in = float(np.linalg.norm(Qc.T @ h) / max(np.linalg.norm(h), 1e-12)) if Qc.size else 0.0
    ang = np.degrees(np.arccos(np.clip(
        np.linalg.svd(Q.T @ (np.linalg.qr(C_p)[0]), compute_uv=False), -1, 1)))
    low_share, high_share = [], []
    for j in range(n_ch):
        tot = np.linalg.norm(C_p[:, j])
        hi_ = np.linalg.norm(C_perp[:, j])
        low_share.append(float(np.sqrt(max(tot ** 2 - hi_ ** 2, 0.0)) / max(tot, 1e-12)))
        high_share.append(float(hi_ / max(tot, 1e-12)))

    spectrum = {
        "singular_values": sv.tolist(),
        "effective_rank": eff_rank,
        "condition_number": float(sv.max() / sv.min()) if sv.size and sv.min() > 0 else float("inf"),
        "h_in_coil_span": h_in,
        "principal_angles_deg": ang.tolist(),
        "coil_low_share": low_share,
        "coil_high_share": high_share,
        "pfc_voxels": int(pfc.sum()),
        "rest_voxels": int(rest.sum()),
    }

    rows, by_config = [], {}

    def record(name, s_vec, i_vec, info=None, objective=""):
        corr = np.zeros_like(b0, dtype=np.float64)
        for k, n in enumerate(SH12):
            corr += s_vec[k] * basis[n]
        if n_ch:
            corr += np.tensordot(coil_maps, i_vec, axes=([3], [0]))
        r = b0 + corr
        m = regional_metrics(r, pfc, rest, brain)
        cmd = base + s_vec
        m.update({
            "config": name,
            "objective": objective,
            # Kept on the row so the caller can archive the actual solution,
            # not just what it scored.
            "scanner_sh_delta": np.asarray(s_vec, dtype=float).copy(),
            "coil_currents": np.asarray(i_vec, dtype=float).copy(),
            "current_rms": float(np.sqrt(np.mean(i_vec ** 2))) if n_ch else 0.0,
            "current_max": float(np.max(np.abs(i_vec))) if n_ch else 0.0,
            "at_limit": int(np.sum(np.abs(np.abs(i_vec) - imax) < 1e-3)) if n_ch else 0,
            "scanner_margin": float(np.min(lim - np.abs(cmd))),
            "scanner_max_abs": float(np.max(np.abs(cmd))),
            "solver": (info or {}).get("solver", ""),
            "status": (info or {}).get("status", "closed-form"),
            "violation": float((info or {}).get("violation", 0.0)),
        })
        rows.append(m)
        by_config[name] = m
        return m

    zero_i = np.zeros(n_ch)

    # 0 — baseline, so every improvement below has something to be read against
    record("none", np.zeros(n_sh), zero_i, {"status": "baseline"}, "—")

    # 1 — scanner alone, PFC objective
    R, u, _ = _factor(S_p, b_p)
    x, info = _solve_ls(R, u, s_lo, s_hi, name="sh_only")
    record("sh_only", x, zero_i, info, "PFC RMS")
    log("  sh_only 完成")

    # 2 — coils alone, PFC objective
    R, u, _ = _factor(C_p, b_p)
    x, info = _solve_ls(R, u, np.full(n_ch, -imax), np.full(n_ch, imax),
                        name="coil_only")
    record("coil_only", np.zeros(n_sh), x, info, "PFC RMS")

    # 3a/b/c — the high-order question. `s` does not appear: S s lies entirely
    # inside the projected-out subspace, so it cannot touch h + C_perp i.
    R_h, u_h, _ = _factor(C_perp, h)
    sig_h = float(np.std(h))

    def high_row(name, xi, info, objective):
        r_high = h + C_perp @ xi
        # The command needed to leave exactly the high-order part behind: the
        # scanner has to cancel the coils' low-order field and the field's own
        # low-order part.
        s_need = (np.linalg.lstsq(L_p, -(b_p + C_p @ xi), rcond=None)[0])[1:]
        m = record(name, s_need, xi, info, objective)
        m.update({
            "high_sigma_before": sig_h,
            "high_sigma_after": float(np.std(r_high)),
            "high_improve_pct": float(100.0 * (1 - np.std(r_high) / max(sig_h, 1e-12))),
            "high_improve_hz": float(sig_h - np.std(r_high)),
        })
        return m

    xi, *_ = np.linalg.lstsq(C_perp, -h, rcond=None)
    high_row("high_unconstrained", xi, {"status": "lstsq"}, "PFC high-order")

    xi, info = _solve_ls(R_h, u_h, np.full(n_ch, -imax), np.full(n_ch, imax),
                         name="high_bounded")
    high_row("high_bounded", xi, info, "PFC high-order")

    # The scanner command implied by a given i, as a linear map:
    #   s(i) = -pinv(S_p) (b_p + C_p i) = s0 - K i
    pinvL = np.linalg.pinv(L_p)
    s0 = -(pinvL @ b_p)[1:]
    Kmap = (pinvL @ C_p)[1:]
    xi, info = _solve_ls(
        R_h, u_h, np.full(n_ch, -imax), np.full(n_ch, imax),
        extra=[(-Kmap, s_lo - s0, s_hi - s0)], name="high_full")
    high_row("high_full", xi, info, "PFC high-order + scanner feasible")
    log("  high-order 三档完成")

    # 4 — what the app ships
    from core.adaptive_joint import solve_adaptive_joint

    prod = solve_adaptive_joint(
        b0, brain, [basis[n] for n in SH12], coil_maps, imax=imax,
        lambda_pfc_low=0.0,
        weight_kwargs={"base": 1.0, "alpha": 0.0, "top_fraction": 0.0},
        sh_bounds=list(zip(s_lo, s_hi)), log_cb=None)
    record("production", prod["scanner_sh_delta"], prod["coil_currents"],
           {"status": "L-BFGS-B", "solver": "scipy"}, "whole-brain RMS")

    # 5 — the best any solution could do, under every real constraint
    for name, (S_q, C_q, b_q, obj) in {
        "bound_pfc": (S_p, C_p, b_p, "PFC RMS"),
        "bound_whole": (S_b, C_b, b_b, "whole-brain RMS"),
    }.items():
        R, u, _ = _factor(np.hstack([S_q, C_q]), b_q)
        x, info = _solve_ls(R, u, np.r_[s_lo, np.full(n_ch, -imax)],
                            np.r_[s_hi, np.full(n_ch, imax)], name=name)
        record(name, x[:n_sh], x[n_sh:], info, obj)
    # 5b — the bound that actually answers the question. `bound_pfc` above is
    # free to wreck the rest of the brain, and on every subject it does; a gap
    # measured against it fires for everyone and separates nobody. This one asks
    # what is reachable in the PFC *while the rest stays at least as good as the
    # shipping solution* -- the only kind of improvement that could be adopted.
    R_p, u_p, _ = _factor(np.hstack([S_p, C_p]), b_p)
    R_g, u_g, res_g = _factor(np.hstack([S_r, C_r]), b_r)
    # The guard sits exactly on the production solution, which is therefore
    # feasible only on the boundary -- the feasible set has empty interior and
    # the conic solvers report `infeasible` rather than returning that point.
    # GUARD_SLACK opens an interior. 1% of the rest RMS is far below anything
    # measurable and is stated in the config's objective string so the number
    # is never mistaken for an exact no-degradation guarantee.
    tau = float(by_config["production"]["rest_rms"]) * (1.0 + GUARD_SLACK)
    x, info = _solve_ls(
        R_p, u_p, np.r_[s_lo, np.full(n_ch, -imax)],
        np.r_[s_hi, np.full(n_ch, imax)],
        soc=[(R_g, u_g, res_g, tau)], name="bound_pfc_guarded")
    m = record("bound_pfc_guarded", x[:n_sh], x[n_sh:], info,
               f"PFC RMS s.t. rest <= {tau:.1f} Hz (+{GUARD_SLACK:.0%})")
    m["guard_rest_hz"] = tau
    log("  理论上限完成")

    by = by_config
    gap = {
        # The headline number: what a PFC-aware method could win without giving
        # anything back elsewhere.
        "gap_guarded_hz": by["production"]["pfc_rms"] - by["bound_pfc_guarded"]["pfc_rms"],
        # The same against the unguarded bound. Always larger, never adoptable
        # as-is; kept to show how much of that gap the rest of the brain pays.
        "gap_pfc_hz": by["production"]["pfc_rms"] - by["bound_pfc"]["pfc_rms"],
        "gap_whole_hz": by["production"]["brain_rms"] - by["bound_whole"]["brain_rms"],
        "rest_cost_unguarded_hz": (by["bound_pfc"]["rest_rms"]
                                   - by["production"]["rest_rms"]),
    }
    p_pfc = max(by["production"]["pfc_rms"], 1e-12)
    gap["gap_guarded_pct"] = 100.0 * gap["gap_guarded_hz"] / p_pfc
    gap["gap_pfc_pct"] = 100.0 * gap["gap_pfc_hz"] / p_pfc
    gap["gap_whole_pct"] = 100.0 * gap["gap_whole_hz"] / max(by["production"]["brain_rms"], 1e-12)

    return {"rows": rows, "spectrum": spectrum, "gap": gap}
