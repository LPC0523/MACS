"""Safety bounds, encoding quality, and an electrical model that admits it has
no data.

IDEALIZED OFFLINE ENCODING SIMULATION.

Clipping is not available here, at all
--------------------------------------
When a current history exceeds +-3 A there are exactly two answers: refuse, or
scale the whole dynamic waveform by one factor. Clipping sample by sample would
keep the numbers inside the bound while quietly changing the waveform into a
different one -- and since the k-space trajectory is the integral of that
waveform, a clipped run reports a trajectory the currents no longer produce.
The failure is invisible in every plot. `reject` is the default because a
refusal is a fact and a scaled result is an answer to a different question.

Uniform scaling keeps the relative amplitude and phase of every channel and the
shape of every waveform: one scalar multiplies the dynamic part, the static
shim baseline is untouched, and the achieved gradient and trajectory are
recomputed and reported rather than assumed to scale.

The electrical model
--------------------
No resistance, inductance, mutual inductance, voltage limit or amplifier
transfer function is recorded anywhere in this project. Voltage and power are
therefore reported as `unknown`, never as zero: a zero would read as "no
voltage needed", which is the opposite of "we have no idea". If the caller
supplies R and L, the estimate is computed and labelled as depending on them.
"""

from __future__ import annotations

import numpy as np

from core.linear_encoding_waveform import check_time_axis, derivative

__all__ = ["ConstraintViolation", "check_current_bounds", "apply_strategy",
           "encoding_metrics", "electrical_estimate", "gram_matrix",
           "sinusoid_phase_terms", "to_mt_per_m", "STRATEGIES",
           "COUPLING_NOT_CHARACTERISED", "GAMMA_HZ_PER_T", "GAMMA_HZ_PER_MT"]

#: Proton gyromagnetic ratio over 2*pi.
#:     42.577e6 Hz/T  ==  42577 Hz/mT
#: so  G[mT/m] = G[Hz/m] / 42577. Used only to restate a gradient in familiar
#: units; nothing here depends on the field strength.
GAMMA_HZ_PER_T = 42_577_000.0
GAMMA_HZ_PER_MT = GAMMA_HZ_PER_T / 1e3          # 42577.0


def to_mt_per_m(hz_per_m):
    """Hz/m -> mT/m. The only place the factor appears.

    Always convert the quantity you are about to print, and print its name.
    A table once carried a requested beta in its Hz/m column and the *achieved*
    gradient magnitude in its mT/m column: both numbers were right, the pair
    was not, and the discrepancy read as a broken gamma when it was really a
    gain of 0.48 hiding in plain sight.
    """
    import numpy as _np
    return _np.asarray(hz_per_m, dtype=float) / GAMMA_HZ_PER_MT

STRATEGIES = ("reject", "uniform_scale")

#: The scanner's own gradients induce currents in a local coil array. Measuring
#: that needs a transfer function nobody has taken here, so the status is
#: stated rather than modelled. Filling in zero would assert there is no
#: coupling, which is a much stronger claim than having no data.
COUPLING_NOT_CHARACTERISED = "not_characterized"


class ConstraintViolation(RuntimeError):
    """A current history that cannot be run under the chosen strategy."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {}


def check_current_bounds(t, i_total, limit_a=3.0, i_static=None,
                         channel_names=None) -> dict:
    """Per-channel, per-sample bound check. Reports; never modifies.

    `i_total` already includes the static shim baseline, because the amplifier
    sees one current per channel and does not care which part of it is
    "encoding". The headroom left for the dynamic part is therefore asymmetric
    whenever the baseline is not zero, and that asymmetry is reported per
    channel rather than assumed away.
    """
    t = check_time_axis(t)
    I = np.asarray(i_total, dtype=np.float64)
    if I.ndim != 2 or I.shape[0] != t.shape[0]:
        raise ValueError("i_total must be (n_time, n_channels) matching t")
    if not np.all(np.isfinite(I)):
        raise ValueError("i_total contains non-finite values")
    J = I.shape[1]
    lim = np.abs(np.broadcast_to(np.asarray(limit_a, dtype=np.float64).ravel()
                                 if np.ndim(limit_a) else
                                 np.full(J, float(limit_a)), (J,)))
    base = (np.zeros(J) if i_static is None
            else np.asarray(i_static, dtype=np.float64).ravel())
    if base.shape[0] != J:
        raise ValueError("static currents have %d entries for %d channels"
                         % (base.shape[0], J))
    names = list(channel_names) if channel_names else \
        ["ch%d" % (k + 1) for k in range(J)]

    dyn = I - base
    di = derivative(t, I)
    over = np.abs(I) > lim[None, :] + 1e-12
    n_viol = int(over.sum())
    first_idx = int(np.argmax(over.any(axis=1))) if over.any() else -1

    per = []
    for j in range(J):
        col = I[:, j]
        per.append({
            "channel": names[j],
            "limit_a": float(lim[j]),
            "static_a": float(base[j]),
            "max_a": float(col.max()), "min_a": float(col.min()),
            "peak_abs_a": float(np.abs(col).max()),
            "rms_a": float(np.sqrt(np.mean(col ** 2))),
            "headroom_up_a": float(lim[j] - col.max()),
            "headroom_down_a": float(col.min() + lim[j]),
            "dynamic_window_lower_a": float(-lim[j] - base[j]),
            "dynamic_window_upper_a": float(lim[j] - base[j]),
            "dynamic_peak_abs_a": float(np.abs(dyn[:, j]).max()),
            "max_abs_didt_a_per_s": float(np.abs(di[:, j]).max()),
            "violations": int(over[:, j].sum()),
            "saturated": bool(over[:, j].any()),
        })

    return {
        "ok": n_viol == 0,
        "limit_a": lim.tolist(),
        "violation_count": n_viol,
        "violating_channels": [names[j] for j in range(J) if over[:, j].any()],
        "first_violation_index": first_idx,
        "first_violation_time_s": (float(t[first_idx]) if first_idx >= 0
                                   else None),
        "global_peak_abs_a": float(np.abs(I).max()),
        "global_max_abs_didt_a_per_s": float(np.abs(di).max()),
        "static_within_bounds": bool(np.all(np.abs(base) <= lim + 1e-12)),
        "per_channel": per,
        "clipping_applied": False,
    }


def _max_uniform_scale(dyn, base, lim) -> float:
    """Largest s with  |base + s*dyn(t)| <= lim  for every channel and sample."""
    s = np.inf
    J = dyn.shape[1]
    for j in range(J):
        hi = lim[j] - base[j]           # room upward from the baseline
        lo = -lim[j] - base[j]          # room downward (negative)
        if hi < 0 or lo > 0:
            return 0.0                  # baseline already outside the bound
        dmax = float(dyn[:, j].max())
        dmin = float(dyn[:, j].min())
        if dmax > 0:
            s = min(s, hi / dmax)
        if dmin < 0:
            s = min(s, lo / dmin)       # both negative -> positive ratio
    return float(max(0.0, s if np.isfinite(s) else 1.0))


def apply_strategy(t, dynamic_currents, i_static=None, limit_a=3.0,
                   strategy="reject", channel_names=None) -> dict:
    """Enforce the bound by refusing or by one uniform scale factor.

    Returns a dict with `scale`, the accepted `currents` (total, including the
    baseline) and the bound report. Raises `ConstraintViolation` under
    `reject`, and also whenever the *static* baseline alone is out of bounds --
    no scaling of the dynamic part can rescue that.
    """
    if strategy not in STRATEGIES:
        raise ValueError("strategy must be one of %s" % (STRATEGIES,))
    t = check_time_axis(t)
    dyn = np.asarray(dynamic_currents, dtype=np.float64)
    if dyn.ndim != 2 or dyn.shape[0] != t.shape[0]:
        raise ValueError("dynamic currents must be (n_time, n_channels)")
    J = dyn.shape[1]
    base = (np.zeros(J) if i_static is None
            else np.asarray(i_static, dtype=np.float64).ravel())
    lim = (np.full(J, float(limit_a)) if not np.ndim(limit_a)
           else np.abs(np.asarray(limit_a, dtype=np.float64).ravel()))
    names = list(channel_names) if channel_names else \
        ["ch%d" % (k + 1) for k in range(J)]

    if np.any(np.abs(base) > lim + 1e-12):
        bad = [names[j] for j in range(J) if abs(base[j]) > lim[j] + 1e-12]
        raise ConstraintViolation(
            "静态匀场基线本身已超出 ±%g A：%s。缩放动态部分无法修复，"
            "请先修正 baseline。" % (float(lim.max()), "、".join(bad)))

    report = check_current_bounds(t, dyn + base, lim, base, names)
    requested_peak = float(np.abs(dyn).max())

    if report["ok"]:
        return {"strategy": strategy, "scale": 1.0, "scaled": False,
                "currents": dyn + base, "dynamic_currents": dyn,
                "requested_peak_dynamic_a": requested_peak,
                "feasible_peak_dynamic_a": requested_peak,
                "bounds": report, "clipping_applied": False}

    if strategy == "reject":
        raise ConstraintViolation(
            "电流超出 ±%g A：%d 个采样点越界，通道 %s，首次在 t=%.6g s。"
            "默认策略为 reject，未产生任何输出。改用 uniform_scale 可整体缩放。"
            % (float(lim.max()), report["violation_count"],
               "、".join(report["violating_channels"]),
               report["first_violation_time_s"] or 0.0),
            report)

    s = _max_uniform_scale(dyn, base, lim)
    if s <= 0:
        raise ConstraintViolation(
            "在当前 baseline 下没有任何正的缩放因子能满足边界。", report)
    scaled = dyn * s
    after = check_current_bounds(t, scaled + base, lim, base, names)
    if not after["ok"]:
        # Only floating-point slack should ever get here; if it does the scale
        # is wrong and saying so beats shipping an out-of-bounds waveform.
        raise ConstraintViolation(
            "缩放后仍有 %d 个采样点越界，缩放因子计算有误。"
            % after["violation_count"], after)
    return {"strategy": strategy, "scale": float(s), "scaled": True,
            "currents": scaled + base, "dynamic_currents": scaled,
            "requested_peak_dynamic_a": requested_peak,
            "feasible_peak_dynamic_a": float(np.abs(scaled).max()),
            "bounds": after, "clipping_applied": False}


def gram_matrix(model) -> dict:
    """Weighted Gram of the ROI-demeaned coil maps, plus its spectrum.

    Says how many genuinely different spatial patterns the array can make. A
    12-channel array whose maps resemble one another has a spectrum that falls
    off a cliff, and reading "12 channels" as "12 encoding dimensions" would be
    wrong by whatever that cliff hides.
    """
    s = model.singular_values()
    Cw = model.C_demeaned * np.sqrt(model.weights)[:, None]
    G = Cw.T @ Cw / model.w_sum
    return {
        "gram": G.tolist(),
        "singular_values": s.tolist(),
        "singular_values_normalised": (s / s[0]).tolist() if s.size and s[0] > 0
                                      else [],
        "effective_rank_1e-3": model.effective_rank(1e-3),
        "effective_rank_1e-2": model.effective_rank(1e-2),
        "n_channels": int(model.n_channels),
        "condition_number": (float(s[0] / s[-1]) if s.size and s[-1] > 0
                             else float("inf")),
    }


def encoding_metrics(model, currents, direction=None,
                     requested_gradient_hz_per_m=None) -> dict:
    """How linear is the field this current vector makes, inside the ROI?

    Everything is computed for one instant, on the ROI, about the ROI centre.
    The constant term is reported separately and excluded from every quality
    figure: a uniform offset changes no spin's position, only its phase.
    """
    i = np.asarray(currents, dtype=np.float64).ravel()
    w = model.weights
    wsum = model.w_sum

    total = model.field_at(i)                       # Hz
    const = float(model.f0 @ i)                     # Hz
    g_vec = model.G @ i                             # Hz/m
    resid = model.residual_at(i)                    # Hz
    linear_only = total - const - resid             # the pure gradient part

    def wrms(v):
        return float(np.sqrt(np.sum(w * v ** 2) / wsum))

    total_dm = total - float(w @ total) / wsum
    g_norm = float(np.linalg.norm(g_vec))
    out = {
        "constant_offset_hz": const,
        "gradient_vector_hz_per_m": g_vec.tolist(),
        # "achieved" in the name because it is not the requested amplitude and
        # confusing the two is the mistake this naming exists to prevent.
        "achieved_gradient_magnitude_hz_per_m": g_norm,
        "achieved_gradient_magnitude_mt_per_m": float(to_mt_per_m(g_norm)),
        "field_rms_hz": wrms(total_dm),
        "linear_rms_hz": wrms(linear_only),
        "nonlinear_rms_hz": wrms(resid),
        "nonlinear_p95_hz": float(np.percentile(np.abs(resid), 95)),
        "nonlinear_max_hz": float(np.abs(resid).max()),
        "nonlinear_mean_hz": float(np.sum(w * resid) / wsum),
        "nonlinear_std_hz": float(np.sqrt(
            np.sum(w * (resid - np.sum(w * resid) / wsum) ** 2) / wsum)),
        "roi_field_range_hz": [float(total.min()), float(total.max())],
        "current_peak_a": float(np.abs(i).max()),
        "current_rms_a": float(np.sqrt(np.mean(i ** 2))),
    }
    den = float(np.sum(w * total_dm ** 2))
    out["linear_fraction"] = (float(max(0.0, 1.0 - np.sum(w * resid ** 2) / den))
                              if den > 0 else float("nan"))

    if direction is not None:
        from core.spatial_grid import direction_field, unit_vector

        d = unit_vector(direction)
        along = float(g_vec @ d)
        leak = g_vec - along * d
        out["target_direction"] = d.tolist()
        # The physically separate pieces, kept apart. NRMSE charges for a
        # tilted-but-linear field and for a bumpy on-axis one identically, and
        # the two need different remedies.
        out["gradient_along_target_hz_per_m"] = along
        out["gradient_along_target_mt_per_m"] = float(to_mt_per_m(along))
        out["orthogonal_leakage_hz_per_m"] = float(np.linalg.norm(leak))
        out["orthogonal_leakage_mt_per_m"] = float(to_mt_per_m(
            np.linalg.norm(leak)))
        out["direction_error_deg"] = (
            float(np.degrees(np.arccos(np.clip(along / g_norm, -1.0, 1.0))))
            if g_norm > 0 else float("nan"))

        tgt = direction_field(model.points, d, centre=model.centre) * (
            requested_gradient_hz_per_m if requested_gradient_hz_per_m
            else along)
        tgt = tgt - float(w @ tgt) / wsum
        err = total_dm - tgt
        out["target_rms_hz"] = wrms(tgt)
        out["nrmse"] = (wrms(err) / wrms(tgt)) if wrms(tgt) > 0 else float("nan")
        ss_tot = float(np.sum(w * (tgt - float(w @ tgt) / wsum) ** 2))
        out["r_squared"] = (float(1.0 - np.sum(w * err ** 2) / ss_tot)
                            if ss_tot > 0 else float("nan"))
        sa, sb = np.sqrt(np.sum(w * total_dm ** 2)), np.sqrt(np.sum(w * tgt ** 2))
        out["correlation"] = (float(np.sum(w * total_dm * tgt) / (sa * sb))
                              if sa > 0 and sb > 0 else float("nan"))
        if requested_gradient_hz_per_m:
            req = float(requested_gradient_hz_per_m)
            out["requested_gradient_hz_per_m"] = req
            out["requested_gradient_mt_per_m"] = float(to_mt_per_m(req))
            out["gradient_gain"] = float(along / req)
    return out


def sinusoid_phase_terms(gradient_peak_hz_per_m, frequency_hz, position_m):
    """Three different phase numbers for a sinusoidal gradient, named apart.

    For g(t) = beta sin(wt) along d, the field at offset u = d.(r-r0) is
    B = beta u sin(wt) and the accumulated phase is

        phi(u, t) = 2 pi beta u (1 - cos wt) / w                    [rad]

    which is a non-negative bump, not a symmetric oscillation. Three quantities
    get called "phase swing" and they differ by factors of two:

      peak_from_t0        max_t phi - phi(0)  =  4 pi beta u / w
                          the largest phase a spin reaches, measured from the
                          start of the waveform. This is what dephasing across
                          a voxel cares about.
      ac_amplitude        half the peak-to-peak of the oscillating part
                          =  2 pi beta u / w
                          the steady-state modulation depth once the constant
                          (1) term is dropped.
      peak_to_peak        max_t phi - min_t phi  =  4 pi beta u / w
                          identical to peak_from_t0 here because phi starts at
                          its minimum; they part company for any waveform that
                          does not.

    `position_m` is the signed offset along the direction, so the spatial
    spread across an ROI is obtained by passing its extremes.
    """
    b = float(gradient_peak_hz_per_m)
    f = float(frequency_hz)
    if f <= 0:
        raise ValueError("frequency must be positive")
    w = 2 * np.pi * f
    u = np.asarray(position_m, dtype=np.float64)
    signed = 2 * np.pi * b * u / w          # keeps the spatial sign
    ac = np.abs(signed)                     # amplitudes are magnitudes
    def _v(x):
        return x.tolist() if getattr(x, "ndim", 0) else float(x)
    return {
        "frequency_hz": f,
        "gradient_peak_hz_per_m": b,
        # Amplitudes and peak-to-peak are magnitudes and are never negative;
        # a negative peak-to-peak is not a quantity. The spatial sign lives in
        # `signed_phase_extremum_rad`, which is where a caller should look to
        # know which side of the origin a voxel is on.
        "phase_peak_from_t0_magnitude_rad": _v(2.0 * ac),
        "phase_ac_amplitude_rad": _v(ac),
        "phase_peak_to_peak_rad": _v(2.0 * ac),
        "signed_phase_extremum_rad": _v(2.0 * signed),
        "k_half_cycle_cycles_per_m": b / (np.pi * f),
        "k_full_cycle_cycles_per_m": 0.0,
        "note": ("phi = 2 pi beta u (1-cos wt)/w. A full cycle returns k to "
                 "zero exactly; the half-cycle value is the excursion, not a "
                 "conventional imaging gradient area."),
    }


def electrical_estimate(t, currents, resistance_ohm=None, inductance_h=None,
                        mutual_inductance_h=None, voltage_limit_v=None,
                        channel_names=None) -> dict:
    """Voltage and power, or an honest `unknown`.

    With no R and no L this returns `available: False` and leaves every number
    as None. Substituting zero would read as "no voltage required", which is a
    claim; "unknown" is the fact.

    Given `mutual_inductance_h` the full matrix is used -- v = R i + L di/dt
    with L the whole matrix, not just its diagonal. Channels in a dense array
    are coupled, and using only self-inductance understates the voltage a
    channel needs whenever its neighbours are moving.
    """
    t = check_time_axis(t)
    I = np.asarray(currents, dtype=np.float64)
    J = I.shape[1]
    names = list(channel_names) if channel_names else \
        ["ch%d" % (k + 1) for k in range(J)]

    out = {
        "available": False,
        "resistance_provided": resistance_ohm is not None,
        "inductance_provided": inductance_h is not None,
        "mutual_inductance_provided": mutual_inductance_h is not None,
        "voltage_limit_provided": voltage_limit_v is not None,
        "peak_voltage_v": None, "rms_voltage_v": None,
        "resistive_power_w": None, "per_channel": None,
        "voltage_violations": None,
        "amplifier_bandwidth_status": "unknown",
        "transfer_function_status": "unknown",
        "gradient_coupling_model_status": COUPLING_NOT_CHARACTERISED,
        "note": ("R、L、互感、电压上限和放大器传递函数在本项目中均无记录。"
                 "电压与功率显示为 unknown，而不是 0——0 会被读成"
                 "「不需要电压」，那是一个断言，不是事实。"),
    }
    if resistance_ohm is None and inductance_h is None:
        return out

    R = (np.zeros(J) if resistance_ohm is None
         else np.broadcast_to(np.asarray(resistance_ohm, dtype=np.float64).ravel(),
                              (J,)).copy())
    didt = derivative(t, I)
    if mutual_inductance_h is not None:
        L = np.asarray(mutual_inductance_h, dtype=np.float64)
        if L.shape != (J, J):
            raise ValueError("mutual_inductance_h must be (%d, %d), got %s"
                             % (J, J, L.shape))
        v_ind = didt @ L.T
    elif inductance_h is not None:
        Ld = np.broadcast_to(np.asarray(inductance_h, dtype=np.float64).ravel(),
                             (J,))
        v_ind = didt * Ld[None, :]
    else:
        v_ind = np.zeros_like(didt)
    v = I * R[None, :] + v_ind

    per = []
    for j in range(J):
        per.append({
            "channel": names[j],
            "peak_voltage_v": float(np.abs(v[:, j]).max()),
            "rms_voltage_v": float(np.sqrt(np.mean(v[:, j] ** 2))),
            "resistive_power_w": (float(R[j] * np.mean(I[:, j] ** 2))
                                  if resistance_ohm is not None else None),
        })
    out.update({
        "available": True,
        "peak_voltage_v": float(np.abs(v).max()),
        "rms_voltage_v": float(np.sqrt(np.mean(v ** 2))),
        "resistive_power_w": (float(np.sum(R * np.mean(I ** 2, axis=0)))
                              if resistance_ohm is not None else None),
        "per_channel": per,
        "note": ("电压来自用户提供的 R/L，%s互感矩阵。放大器带宽与传递函数"
                 "仍为 unknown，因此这只是电路估算，不代表功放能跟上。"
                 % ("使用了完整 " if mutual_inductance_h is not None else "未提供")),
    })
    if voltage_limit_v is not None:
        VL = np.broadcast_to(np.asarray(voltage_limit_v, dtype=np.float64).ravel(),
                             (J,))
        viol = np.abs(v) > VL[None, :] + 1e-12
        out["voltage_violations"] = {
            "count": int(viol.sum()),
            "channels": [names[j] for j in range(J) if viol[:, j].any()],
        }
    return out
