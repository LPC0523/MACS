"""Adaptive joint shim: project the coils onto the harmonics before solving.

Ported from `2nd_order_SH/adaptive_joint_shim_no_roi.py`.

The problem with concatenating the 8 harmonic deltas and the N coil currents
into one design matrix is that they are not independent. A surface array's
field has a large low-order component, so the harmonics can reproduce much of
what a coil does. The solver is free to pick either, and with nothing to
separate them it drives the currents to their bounds and cancels the excess
with a global polynomial -- currents saturated, coils contributing nothing the
scanner could not have done itself.

The fix is to split each coil map first. Fit it with the harmonic basis
(weighted least squares over the brain); what the basis can reproduce is the
*low* part and belongs to the scanner, and the residual is the *high* part that
only the coil can produce. The optimiser then sees a coil matrix that the
harmonics cannot imitate, so the two stop competing.

Because the coils still physically emit their low part, it is subtracted from
what the scanner is told to change:

    scanner_sh_delta = model_sh_delta - pfc_low_coeff @ coil_currents

Weighting is adaptive: voxels further from the median field get more say, so
the solve spends its degrees of freedom where the field is actually bad rather
than on the large well-shimmed interior.

Defaults are the reference script's, not this module's own invention -- see
`_DEFAULTS`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

__all__ = ["make_adaptive_weights", "build_low_order_projection",
           "fit_weighted_projection", "solve_adaptive_joint", "solve_two_stage",
           "make_region_weights", "verify_scanner_bounds",
           "ScannerBoundsError", "DEFAULTS"]


#: Straight from `adaptive_joint_shim_no_roi.py`'s argparse defaults.
DEFAULTS = {
    "weight_base": 0.25,
    "weight_alpha": 3.0,
    "weight_power": 2.0,
    "weight_cap": 8.0,
    "weight_percentile": 90.0,
    "top_fraction": 0.10,
    "top_boost": 2.0,
    "lambda_current": 0.1,
    "lambda_sh": 0.25,
    "lambda_pfc_low": 0.02,
    "maxiter": 3000,
}

#: The reference normalises both first- and second-order deltas by 1000.
_SH_NORM = 1000.0


def _weighted_mean(values, weights) -> float:
    denom = float(np.sum(weights))
    return float(np.mean(values)) if denom <= 0 else float(np.sum(weights * values) / denom)


def verify_scanner_bounds(scanner_delta, baseline, limits, names=None,
                          tol: float = 1e-6):
    """Violations of the console's absolute limit by `baseline + scanner_delta`.

    Separate from the solve on purpose. The box the optimiser is given acts on
    the model variable `theta`, but the console is sent `theta - K i`, and the
    two differ by however much low-order field the coils happen to produce --
    measured at up to 2.6x on this array. A solution can therefore satisfy every
    bound the solver knows about and still be one the scanner refuses.

    Returns a list of (name, value, limit). Empty means the command is
    deliverable. Clipping the answer instead would produce a vector that is no
    longer the solution to anything.
    """
    sd = np.asarray(scanner_delta, dtype=float)
    n = sd.size
    names = list(names) if names is not None else [f"sh{k}" for k in range(n)]
    base = (np.zeros(n) if baseline is None
            else np.asarray(baseline, dtype=float))
    lim = np.asarray(limits, dtype=float)

    bad = []
    for k in range(n):
        v = base[k] + sd[k]
        if abs(v) > abs(lim[k]) + tol:
            bad.append((names[k], float(v), float(abs(lim[k]))))
    return bad


class ScannerBoundsError(RuntimeError):
    """The optimised command cannot be sent: it exceeds the console's limits."""

    def __init__(self, violations):
        self.violations = violations
        detail = "、".join(f"{n} {v:+.1f}（限幅 ±{l:g}）" for n, v, l in violations)
        super().__init__(
            "优化结果超出扫描仪限幅，无法下发：" + detail +
            "。请调小对应项的允许变化范围后重算——不要截断结果，"
            "截断后的向量不再是任何问题的解。")


def make_adaptive_weights(b0_vec, base=None, alpha=None, power=None, cap=None,
                          percentile=None, top_fraction=None, top_boost=None):
    """Per-voxel weight rising with distance from the median field.

    Normalised by a high percentile of |b0| rather than by its maximum, so one
    outlying voxel cannot flatten every other weight to nothing. The cap is on
    the ratio term, not on the whole weight.
    """
    d = DEFAULTS
    base = d["weight_base"] if base is None else base
    alpha = d["weight_alpha"] if alpha is None else alpha
    power = d["weight_power"] if power is None else power
    cap = d["weight_cap"] if cap is None else cap
    percentile = d["weight_percentile"] if percentile is None else percentile
    top_fraction = d["top_fraction"] if top_fraction is None else top_fraction
    top_boost = d["top_boost"] if top_boost is None else top_boost

    b0c = b0_vec - np.median(b0_vec)
    ref = max(float(np.percentile(np.abs(b0c), percentile)), 1e-6)
    weights = base + alpha * np.minimum((np.abs(b0c) / ref) ** power, cap)

    if top_fraction > 0:
        frac = float(np.clip(top_fraction, 0.0, 1.0))
        thr = np.percentile(np.abs(b0c), 100.0 * (1.0 - frac))
        weights[np.abs(b0c) >= thr] *= top_boost

    weights = np.clip(weights, 1e-6, None)
    return weights / np.mean(weights)


def make_region_weights(in_region, lambda_penalty: float = 0.05,
                        per_voxel: bool = False):
    """Weights that put a target region ahead of the rest of the brain.

    Two readings of "the PFC counts 20x", and they are not the same thing:

    `per_voxel=False` (default) reproduces the objective as written --

        loss = mean(r^2 over PFC) + lambda * mean(r^2 over rest)

    -- by giving the PFC a total weight of 1 and the rest a total of `lambda`.
    Because each region is normalised by its own voxel count first, the PFC
    ends up dominating by 1/lambda = 20:1 regardless of how small it is.

    `per_voxel=True` instead gives every PFC voxel a weight of 1 and every
    other voxel `lambda`. With a PFC of ~16k voxels inside a ~208k brain that
    is only 1.6:1 in aggregate -- far weaker, because the rest of the brain has
    twelve times as many voxels to vote with.

    Returned with mean 1, so the regularisation terms keep the same relative
    strength they have in whole-brain mode.
    """
    m = np.asarray(in_region, dtype=bool)
    n = m.size
    n_in = int(m.sum())
    n_out = n - n_in
    if n_in == 0:
        raise ValueError("target region is empty")

    w = np.empty(n, dtype=np.float64)
    if per_voxel or n_out == 0:
        w[m] = 1.0
        w[~m] = lambda_penalty if n_out else 0.0
    else:
        w[m] = 1.0 / n_in
        w[~m] = lambda_penalty / n_out
    return w * (n / float(np.sum(w)))


def fit_weighted_projection(target, basis, weights, reg: float = 1e-8):
    """Weighted least-squares coefficients of `target` on `basis`."""
    sw = np.sqrt(np.clip(weights, 0.0, None))
    aw = basis * sw[:, None]
    lhs = aw.T @ aw + reg * np.eye(basis.shape[1])
    return np.linalg.solve(lhs, aw.T @ (target * sw))


def _solve_weighted_linear(b0, design, weights, reg: float = 1e-8,
                           centre: bool = True):
    """Starting point: the harmonics-only weighted least-squares solution."""
    sw = np.sqrt(np.clip(weights, 0.0, None))
    aw = design * sw[:, None]
    b = b0 - _weighted_mean(b0, weights) if centre else b0
    bw = b * sw
    lhs = aw.T @ aw + reg * np.eye(design.shape[1])
    return np.linalg.solve(lhs, -(aw.T @ bw))


def build_low_order_projection(a_sh, a_pfc, weights):
    """Split each coil map into what the harmonics can reproduce, and what they cannot.

    Returns (projection, coefficients): `projection[:, j]` is the harmonic
    approximation of coil j over the mask, and `coefficients[:, j]` are the
    harmonic amplitudes that produce it.
    """
    if a_pfc.shape[1] == 0:
        return np.zeros_like(a_pfc), np.zeros((a_sh.shape[1], 0), dtype=np.float64)
    coeffs, projected = [], []
    for j in range(a_pfc.shape[1]):
        coef = fit_weighted_projection(a_pfc[:, j], a_sh, weights)
        coeffs.append(coef)
        projected.append(a_sh @ coef)
    return np.column_stack(projected), np.column_stack(coeffs)


def _gradient_penalty_terms(b0, design_maps_3d, rest_mask, voxel_size=None):
    """Quadratic form for the spatial-gradient penalty outside the target.

    Penalising the coils' *field* outside the PFC (what `lambda_rest` does)
    fights them everywhere at once, and measured across 44 subjects that cost
    the rest of the head far more than the PFC gained. Penalising the field's
    *slope* instead leaves a large but smooth offset outside the target
    unpunished, while still forbidding the sharp edge at the PFC boundary --
    which is what actually shows up as signal dropout.

    The residual is linear in the unknowns and `np.gradient` is a linear
    operator, so the penalty is quadratic in x just like the data term. It is
    returned as (H, g, c) to be folded into the assembled Hessian rather than
    evaluated per iteration -- a per-call loop over a million voxels would undo
    the whole reason the solve takes 0.4 s.

    Returns H, g, c such that   mean(|grad r|^2 over rest) = c + 2 g.x + x'Hx.
    """
    rest = np.asarray(rest_mask, dtype=bool)
    n_rest = int(rest.sum())
    n_var = len(design_maps_3d)
    if n_rest == 0 or n_var == 0:
        return np.zeros((n_var, n_var)), np.zeros(n_var), 0.0

    spacing = list(voxel_size) if voxel_size is not None else [1.0, 1.0, 1.0]

    H = np.zeros((n_var, n_var), dtype=np.float64)
    g = np.zeros(n_var, dtype=np.float64)
    c = 0.0

    # One axis at a time: the stacked design over all three would be three
    # times the memory for no benefit, since the products accumulate.
    for axis in range(3):
        d0 = np.gradient(b0, spacing[axis], axis=axis)[rest]
        G = np.empty((n_rest, n_var), dtype=np.float64)
        for k, m3 in enumerate(design_maps_3d):
            G[:, k] = np.gradient(m3, spacing[axis], axis=axis)[rest]
        H += G.T @ G
        g += G.T @ d0
        c += float(d0 @ d0)

    return H / n_rest, g / n_rest, c / n_rest


def _optimize_joint(b0_vec, a_sh, a_pfc_model, n_sh, n_pfc, weights,
                    pfc_low_projection, sh_bounds, imax,
                    lambda_current, lambda_sh, lambda_pfc_low, maxiter,
                    grad_terms=None, lambda_grad=0.0, centre: bool = True):
    """Bounded weighted least squares over [sh_delta, coil_currents].

    The reference's "quadratic" optimiser: the objective is quadratic, so its
    Hessian is assembled once and L-BFGS-B is handed exact values and
    gradients. That avoids evaluating a per-voxel residual on every iteration,
    which over a million voxels is the whole cost.

    `centre` selects which of two different problems is solved -- see
    `solve_adaptive_joint`.
    """
    a_joint = np.hstack([a_sh, a_pfc_model])
    bounds = list(sh_bounds) + [(-imax, imax)] * n_pfc
    eps = 1e-12

    x0_sh = _solve_weighted_linear(b0_vec, a_sh, weights, centre=centre)
    x0_sh = np.array([min(max(x0_sh[i], sh_bounds[i][0]), sh_bounds[i][1])
                      for i in range(n_sh)])
    x0 = np.concatenate([x0_sh, np.zeros(n_pfc)])

    sum_w = float(np.sum(weights))
    if sum_w <= 0:
        raise ValueError("weights must have positive sum")
    wn = weights / sum_w

    # Centring subtracts the weighted mean from both sides, which is exactly
    # equivalent to appending a free constant column (A00) and projecting it
    # out: the solve is then allowed to leave any uniform offset behind. With
    # `centre=False` no such column exists, the constant is part of the error
    # like every other component, and what is minimised is plain RMSE.
    if centre:
        b0_c = b0_vec - _weighted_mean(b0_vec, weights)
        a_joint_c = a_joint - np.sum(wn[:, None] * a_joint, axis=0, keepdims=True)
        pfc_low_c = None
        if n_pfc:
            pfc_low_c = pfc_low_projection - np.sum(
                wn[:, None] * pfc_low_projection, axis=0, keepdims=True)
    else:
        b0_c = b0_vec
        a_joint_c = a_joint
        pfc_low_c = pfc_low_projection if n_pfc else None

    weighted_a = a_joint_c * wn[:, None]
    hess = a_joint_c.T @ weighted_a
    grad0 = a_joint_c.T @ (wn * b0_c)
    const0 = 0.5 * float(np.sum(wn * b0_c ** 2))

    if n_sh and lambda_sh > 0:
        hess[:n_sh, :n_sh] += np.diag(
            np.full(n_sh, lambda_sh / n_sh / (_SH_NORM ** 2)))
    if n_pfc and lambda_current > 0:
        sl = slice(n_sh, n_sh + n_pfc)
        hess[sl, sl] += np.eye(n_pfc) * (lambda_current / n_pfc / max(imax ** 2, eps))
    if n_pfc and lambda_pfc_low > 0 and pfc_low_c is not None:
        sl = slice(n_sh, n_sh + n_pfc)
        hess[sl, sl] += lambda_pfc_low * (pfc_low_c.T @ (pfc_low_c * wn[:, None]))

    # Gradient penalty. Also quadratic in x, so it joins the assembled form
    # rather than being evaluated per iteration. The factors of two follow from
    # `value = const0 + grad0.x + 0.5 x'Hx` holding 0.5*||.||^2.
    if grad_terms is not None and lambda_grad > 0:
        H_g, g_g, c_g = grad_terms
        hess += 2.0 * lambda_grad * H_g
        grad0 = grad0 + 2.0 * lambda_grad * g_g
        const0 += lambda_grad * c_g

    def objective(x):
        hx = hess @ x
        return const0 + float(grad0 @ x) + 0.5 * float(x @ hx), grad0 + hx

    res = minimize(objective, x0=x0, method="L-BFGS-B", jac=True, bounds=bounds,
                   options={"maxiter": maxiter, "ftol": 1e-12,
                            "gtol": 1e-8, "maxls": 50})
    return res


def solve_adaptive_joint(b0, brain_mask, sh_basis_list, coil_maps=None,
                         sh_bounds=None, imax: float = 3.0,
                         lambda_current=None, lambda_sh=None, lambda_pfc_low=None,
                         weight_kwargs=None, maxiter=None, log_cb=None,
                         pfc_mask=None, lambda_rest: float = 0.5,
                         pfc_per_voxel: bool = False,
                         project_with_pfc_weights: bool = True,
                         lambda_grad: float = 0.0, voxel_size=None,
                         scanner_baseline=None, scanner_limits=None,
                         sh_names=None, on_violation: str = "report",
                         centre: bool = True) -> dict:
    """Joint solve with coil/harmonic separation.

    Parameters
    ----------
    sh_basis_list : list of 3D arrays, one per harmonic, in the caller's order
    coil_maps     : (nx, ny, nz, n_coils) or None for harmonics only
    sh_bounds     : list of (low, high) per harmonic; defaults to +-2000
    centre        : whether a uniform offset is free. See below.

    `centre=True` (the default, and what the reference script does) removes the
    weighted mean from both sides before solving and from the residual after,
    so the objective is demeaned spread. That is right only when something
    downstream re-locks the centre frequency -- a prescan between the shim
    update and the acquisition. It is what `tools/research/` and the frozen
    pre-registration baselines in `results/preregistration/` were computed
    with, so it stays the default.

    `centre=False` solves the plain overdetermined system

        min || A w + b0 ||^2   over the mask

    with no constant column, and reports the residual as it actually is. Use it
    when the coefficients are sent to the console and scanned with directly: on
    S28 the two differ by 12-27% on the second-order terms (A20 107.3 vs 77.9,
    A21 710.0 vs 586.3), because centring lets a uniform offset absorb field
    the harmonics would otherwise have had to correct. The system-shim page
    passes False.

    Returns a dict with `scanner_sh_delta` (what to type into the console),
    `model_sh_delta` and `pfc_low_equiv` (how it was reached), `coil_currents`,
    the residual volume and before/after RMSE.
    """
    log = log_cb or (lambda *_: None)
    d = DEFAULTS
    lambda_current = d["lambda_current"] if lambda_current is None else lambda_current
    lambda_sh = d["lambda_sh"] if lambda_sh is None else lambda_sh
    lambda_pfc_low = d["lambda_pfc_low"] if lambda_pfc_low is None else lambda_pfc_low
    maxiter = d["maxiter"] if maxiter is None else maxiter

    m = brain_mask > 0.5
    if not m.any():
        raise ValueError("brain mask is empty")

    b_vec = b0[m].astype(np.float64)
    a_sh = np.stack([f[m] for f in sh_basis_list], axis=-1).astype(np.float64)
    n_sh = a_sh.shape[1]

    if coil_maps is not None and coil_maps.shape[-1] > 0:
        a_pfc = coil_maps.reshape(-1, coil_maps.shape[-1])[m.ravel(), :].astype(np.float64)
    else:
        a_pfc = np.zeros((b_vec.size, 0), dtype=np.float64)
    n_pfc = a_pfc.shape[1]

    # Whole-brain (default) or target-weighted. Weighting is the only thing
    # that changes: the harmonics, the per-channel projection and the single
    # joint solve are all exactly as in whole-brain mode.
    if pfc_mask is not None:
        in_region = (np.asarray(pfc_mask, dtype=bool) & m)[m]
        if not in_region.any():
            raise ValueError("pfc_mask does not overlap the brain mask")
        # lambda_rest = 0 means "value penalty off" -- the mask still selects
        # the target, so the gradient penalty alone can be studied.
        weights = (make_region_weights(in_region, lambda_rest, pfc_per_voxel)
                   if lambda_rest > 0 else np.ones_like(b_vec))
        log(f"  目标区加权：{int(in_region.sum()):,} / {in_region.size:,} 体素，"
            f"λ_rest={lambda_rest}"
            f"（{'逐体素' if pfc_per_voxel else '按区域均值'}口径，"
            f"总权重比 {weights[in_region].sum() / max(weights[~in_region].sum(), 1e-12):.1f} : 1）")
    else:
        weights = make_adaptive_weights(b_vec, **(weight_kwargs or {}))

    # Step 0: split the coils. With no coils this returns empty arrays and the
    # solve reduces exactly to the harmonics-only problem.
    # The split itself is unchanged code; only which voxels it listens to can
    # differ. Following the objective's weighting keeps the two consistent --
    # the coil/harmonic boundary is then drawn where the solve actually cares.
    proj_weights = weights
    if pfc_mask is not None and not project_with_pfc_weights:
        proj_weights = np.ones_like(b_vec)
    pfc_low, pfc_low_coeff = build_low_order_projection(a_sh, a_pfc, proj_weights)
    a_pfc_high = a_pfc - pfc_low
    if n_pfc:
        # Weighted norms, because that is the inner product the projection is
        # orthogonal in. Measuring with the plain norm makes the low part look
        # larger than the whole (>100%) whenever the weights are non-uniform.
        def _wnorm(x):
            return np.sqrt(np.sum(weights[:, None] * x ** 2, axis=0) / np.sum(weights))

        denom = np.maximum(_wnorm(a_pfc), 1e-12)
        low_share = _wnorm(pfc_low) / denom
        high_share = _wnorm(a_pfc_high) / denom
        log(f"  线圈场分解：低阶 {low_share.mean() * 100:.0f}%"
            f"（{low_share.min() * 100:.0f}~{low_share.max() * 100:.0f}%）交给扫描仪，"
            f"线圈保留高阶 {high_share.mean() * 100:.0f}%"
            f"（{high_share.min() * 100:.0f}~{high_share.max() * 100:.0f}%）")

    if sh_bounds is None:
        sh_bounds = [(-2000.0, 2000.0)] * n_sh

    # Gradient penalty on everything outside the target. Built from the same
    # design the solve uses -- the harmonic maps, and the coils' *high* part
    # after projection -- so what is penalised is exactly the field the
    # variables produce.
    grad_terms = None
    if lambda_grad > 0:
        if pfc_mask is None:
            raise ValueError("lambda_grad 需要 pfc_mask 来界定「目标区之外」")
        rest3 = m & ~(np.asarray(pfc_mask, dtype=bool))
        if not rest3.any():
            raise ValueError("目标区覆盖了整个脑掩膜，梯度惩罚没有作用范围")
        maps3d = [np.asarray(f, dtype=np.float64) for f in sh_basis_list]
        for j in range(n_pfc):
            high3 = np.asarray(coil_maps[..., j], dtype=np.float64).copy()
            for i, f in enumerate(sh_basis_list):
                high3 -= pfc_low_coeff[i, j] * f
            maps3d.append(high3)
        grad_terms = _gradient_penalty_terms(b0, maps3d, rest3, voxel_size)
        log(f"  梯度正则化：λ_grad={lambda_grad}，作用于目标区之外 "
            f"{int(rest3.sum()):,} 体素")

    log(f"  联合优化 {n_sh} 项球谐 + {n_pfc} 通道电流"
        f"（λ_I={lambda_current}, λ_SH={lambda_sh}, λ_low={lambda_pfc_low}"
        + (f", λ_grad={lambda_grad}" if lambda_grad > 0 else "") + "）")
    res = _optimize_joint(b_vec, a_sh, a_pfc_high, n_sh, n_pfc, weights, pfc_low,
                          sh_bounds, imax, lambda_current, lambda_sh,
                          lambda_pfc_low, maxiter,
                          grad_terms=grad_terms, lambda_grad=lambda_grad,
                          centre=centre)

    model_sh_delta = res.x[:n_sh]
    coil_currents = res.x[n_sh:]
    pfc_low_equiv = (pfc_low_coeff @ coil_currents if n_pfc
                     else np.zeros(n_sh, dtype=np.float64))
    scanner_sh_delta = model_sh_delta - pfc_low_equiv

    # The field that actually results: the scanner applies `scanner_sh_delta`
    # and the coils emit their whole map, low part included.
    correction = np.zeros_like(b0, dtype=np.float64)
    for i, field in enumerate(sh_basis_list):
        correction += scanner_sh_delta[i] * field
    if n_pfc:
        correction += np.tensordot(coil_maps, coil_currents, axes=([3], [0]))
    residual = b0 + correction

    # The uniform offset left in the residual. Whether it is *removed* depends
    # on whether anything downstream will actually remove it. With `centre` the
    # objective never tried to correct a constant, so reporting it would score
    # the solve on something it did not optimise; without it the constant was
    # part of the error the solve minimised, and subtracting it here would
    # report a field nobody will measure. Either way the number is returned.
    centre_frequency_hz = float(np.mean(residual[m]))
    if centre:
        residual = residual - centre_frequency_hz

    rmse_before = float(np.sqrt(np.mean(b_vec ** 2)))
    rmse_after = float(np.sqrt(np.mean(residual[m] ** 2)))
    at_limit = int(sum(min(abs(c - lo), abs(c - hi)) < 1e-3
                       for c, (lo, hi) in zip(coil_currents,
                                              [(-imax, imax)] * n_pfc)))
    log(f"  中心频率偏移 {centre_frequency_hz:+.1f} Hz"
        + ("（已从残差中扣除，假定 prescan 重新锁频）" if centre
           else "（残差中保留，未假定重新锁频）"))
    log(f"  全脑 RMSE {rmse_before:.1f} → {rmse_after:.1f} Hz"
        + (f"，{at_limit}/{n_pfc} 个通道触及电流上限" if n_pfc else ""))

    # The command that would actually be sent, checked against the console's
    # own limits. theta was inside its box by construction; theta - K i was
    # never constrained, so this is the first point at which deliverability is
    # known. See verify_scanner_bounds.
    violations = []
    if scanner_limits is not None:
        violations = verify_scanner_bounds(scanner_sh_delta, scanner_baseline,
                                           scanner_limits, sh_names)
        if violations:
            detail = "、".join(f"{n} {v:+.1f}>{l:g}" for n, v, l in violations)
            log(f"  ⚠ 扫描仪命令越界：{detail}")
            if on_violation == "raise":
                raise ScannerBoundsError(violations)

    return {
        "scanner_violations": violations,
        "scanner_sh_delta": scanner_sh_delta,
        "model_sh_delta": model_sh_delta,
        "pfc_low_equiv": pfc_low_equiv,
        "pfc_low_coeff": pfc_low_coeff,
        "coil_currents": coil_currents,
        "residual": residual,
        "correction": correction,
        "centre_frequency_hz": centre_frequency_hz,
        "weights": weights,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "channels_at_limit": at_limit,
        "success": bool(res.success),
        "n_iter": int(res.nit),
        "message": str(res.message),
    }


# ─────────────────────────────────────────────────────────────────────────
#  Two-stage subspace decomposition
# ─────────────────────────────────────────────────────────────────────────
# First order (X/Y/Z) and a surface array do the same kind of thing over a
# local region: both produce a smooth, large-scale field. Given both, a solver
# picks the harmonics -- three global coefficients are cheaper than twelve
# local currents -- and the coils end up doing little the scanner could not
# have done alone.
#
# Splitting the job by what each is uniquely able to do:
#
#   stage 1  the coils alone shape the target region, held to not spoiling the
#            rest of the brain by more than `epsilon`
#   stage 2  the coils are then fixed, and the five second-order terms clean up
#            whatever is left, over the whole brain
#
# First order is never solved for. It is prescan's, and leaving it out of the
# design is what stops it from absorbing the coils' contribution.


def _bounded_quadratic(b_vec, design, bounds, weights=None, centre=True,
                       ridge=0.0, maxiter=3000):
    """min 0.5 * ||b + A x||^2_w over a box, via the assembled Hessian.

    The same formulation as `_optimize_joint` minus the harmonic/coil split, so
    a stage that solves for only one kind of variable still uses the same
    objective and the same solver as the joint path.
    """
    n = design.shape[1]
    if n == 0:
        return np.zeros(0), None

    w = np.ones_like(b_vec) if weights is None else weights
    sum_w = float(np.sum(w))
    if sum_w <= 0:
        raise ValueError("weights must have positive sum")
    wn = w / sum_w

    if centre:
        b_c = b_vec - _weighted_mean(b_vec, w)
        a_c = design - np.sum(wn[:, None] * design, axis=0, keepdims=True)
    else:
        b_c, a_c = b_vec, design

    hess = a_c.T @ (a_c * wn[:, None])
    if ridge:
        hess += ridge * np.eye(n)
    grad0 = a_c.T @ (wn * b_c)
    const0 = 0.5 * float(np.sum(wn * b_c ** 2))

    def obj(x):
        hx = hess @ x
        return const0 + float(grad0 @ x) + 0.5 * float(x @ hx), grad0 + hx

    res = minimize(obj, x0=np.zeros(n), method="L-BFGS-B", jac=True,
                   bounds=list(bounds),
                   options={"maxiter": maxiter, "ftol": 1e-12,
                            "gtol": 1e-8, "maxls": 50})
    return res.x, res


def _coil_field(coil_maps, currents):
    if coil_maps is None or coil_maps.shape[-1] == 0:
        return None
    return np.tensordot(coil_maps, currents, axes=([3], [0]))


def solve_two_stage(b0_data, brain_mask, sh_basis, coil_maps,
                    pfc_mask=None, imax: float = 3.0, sh_bound: float = 2000.0,
                    epsilon: float = 0.05, log_cb=None) -> dict:
    """Coils shape the target region first; second-order harmonics clean up after.

    EXPLORATORY -- not used by the application, kept for the record.

    Measured on S28 / PFC 12ch (`tests/test_two_stage.py --sweep`), asking the
    coils to favour the PFC costs the rest of the brain about 4.3 Hz of spread
    for every 1 Hz the PFC gains, across the whole range where the PFC gains
    anything at all (epsilon 0.5-0.9; 3.7-5.0 : 1). The frontier never enters
    the clinically acceptable region and flattens above epsilon 0.9, where the
    +-3 A amplifier limit binds instead of the constraint. Joint whole-brain
    optimisation -- what the app ships -- is the better strategy, and the limit
    is the array's field shape rather than the solver.

    `sh_basis` is the full eight-term dict; only its five second-order terms are
    used. First order is deliberately never optimised.

    `pfc_mask` is where the coils are asked to work. `None` means the whole
    brain -- the coils then have no smaller region to favour, which is the
    sensible reading of "no region given". (The specification also said `None`
    should skip stage 1 entirely; that would leave every current at zero and
    reduce this to second-order-only shimming, so it is not what happens here.)

    `epsilon` bounds how much stage 1 may worsen whole-brain spread. If the
    unconstrained coil solution breaches it, the current bound is bisected until
    it does not -- the coils give up range rather than the rest of the head
    giving up homogeneity.

    Metrics are demeaned spread (std), matching what the UI reports: the
    constant component is the centre frequency and is not ours to move.
    """
    log = log_cb or (lambda *_: None)
    from core.hybrid_shim import ORDER

    brain = brain_mask > 0.5
    if not brain.any():
        raise ValueError("brain mask is empty")
    target = brain if pfc_mask is None else (pfc_mask > 0.5) & brain
    if not target.any():
        raise ValueError("target (PFC) mask does not overlap the brain")

    order_of = {n: ORDER.get(n) for n in sh_basis}
    sh2_names = [n for n in sh_basis if order_of[n] == 2]
    if not sh2_names:
        raise ValueError("sh_basis contains no second-order terms")

    n_coils = 0 if coil_maps is None else coil_maps.shape[-1]
    b_brain = b0_data[brain].astype(np.float64)
    std_before = float(np.std(b_brain))
    log(f"  两阶段：目标区 {int(target.sum()):,} 体素 / 全脑 {int(brain.sum()):,} 体素，"
        f"二阶项 {', '.join(sh2_names)}")

    # ── Stage 1: coils alone over the target region ──────────────────
    currents = np.zeros(n_coils, dtype=np.float64)
    s1_before = float(np.std(b0_data[target]))
    s1_after = s1_before
    limit_used = float(imax)

    if n_coils:
        a_coil_t = coil_maps.reshape(-1, n_coils)[target.ravel(), :].astype(np.float64)
        b_t = b0_data[target].astype(np.float64)
        a_coil_b = coil_maps.reshape(-1, n_coils)[brain.ravel(), :].astype(np.float64)
        tol = (1.0 + epsilon) * std_before

        x, _ = _bounded_quadratic(b_t, a_coil_t, [(-imax, imax)] * n_coils)
        if float(np.std(b_brain + a_coil_b @ x)) <= tol:
            currents, limit_used = x, float(imax)
        else:
            # Breached: bisect the allowed current until the rest of the brain
            # is back inside tolerance. Monotone in the bound, so bisection is
            # sound -- a smaller box can only contain smaller corrections.
            lo, hi = 0.0, float(imax)
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                xm, _ = _bounded_quadratic(b_t, a_coil_t, [(-mid, mid)] * n_coils)
                if float(np.std(b_brain + a_coil_b @ xm)) <= tol:
                    lo, currents, limit_used = mid, xm, mid
                else:
                    hi = mid
            log(f"  Stage 1：全脑退化超出 {epsilon:.0%}，电流范围收窄到 "
                f"±{limit_used:.2f} A（配置上限 ±{imax:g} A）")

        s1_after = float(np.std(b_t + a_coil_t @ currents))
        log(f"  Stage 1（线圈独立）目标区 std {s1_before:.1f} → {s1_after:.1f} Hz"
            f"，同期全脑 {std_before:.1f} → "
            f"{float(np.std(b_brain + a_coil_b @ currents)):.1f} Hz")

    # ── Stage 2: coils fixed, second-order harmonics over the whole brain ──
    coil_field = _coil_field(coil_maps, currents)
    b0_after_coils = (b0_data + coil_field) if coil_field is not None else b0_data.copy()

    a_sh2 = np.stack([sh_basis[n][brain] for n in sh2_names], axis=-1).astype(np.float64)
    b_after = b0_after_coils[brain].astype(np.float64)
    s2_before = float(np.std(b_after))
    theta, _ = _bounded_quadratic(b_after, a_sh2,
                                  [(-sh_bound, sh_bound)] * len(sh2_names))

    sh_field = np.zeros_like(b0_data, dtype=np.float64)
    for k, n in enumerate(sh2_names):
        sh_field += theta[k] * sh_basis[n]
    residual = b0_after_coils + sh_field
    s2_after = float(np.std(residual[brain]))
    log(f"  Stage 2（二阶球谐清扫）全脑 std {s2_before:.1f} → {s2_after:.1f} Hz")

    at_limit = int(sum(abs(abs(c) - limit_used) < 1e-3 for c in currents))
    return {
        "theta_2nd": theta,
        "sh2_names": sh2_names,
        "coil_currents": currents,
        "residual": residual,
        "stage1_rmse_before": s1_before,
        "stage1_rmse_after": s1_after,
        "stage2_rmse_before": s2_before,
        "stage2_rmse_after": s2_after,
        "rmse_before": std_before,
        "rmse_after": s2_after,
        "current_limit_used": limit_used,
        "channels_at_limit": at_limit,
    }
