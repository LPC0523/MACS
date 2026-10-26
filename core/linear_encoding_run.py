"""The whole offline simulation as plain functions. No Qt, no hardware.

IDEALIZED OFFLINE SIMULATION.

This is what the research page runs, and it is deliberately not a method on a
widget: a pipeline that can only be exercised by clicking is a pipeline that
cannot be tested. `prepare_region` and `run_simulation` take dicts and return
dicts, the worker only moves them between threads, and the end-to-end test
drives the same code the button does.

Three amplitudes, three different meanings
------------------------------------------
    requested   what the user asked for, Hz/m
    capacity    the model's reachable upper bound under this ROI, this
                epsilon, this theta and the +-3 A box. Not an amplifier rating.
    actual      what the simulated currents actually produce after the
                strategy has been applied

They are reported separately and never collapsed. `actual` is recomputed from
the scaled currents rather than obtained by multiplying an old result, because
a uniform scale changes the current vector and every downstream number with it.
"""

from __future__ import annotations

import numpy as np

from core.linear_encoding_metrics import (apply_strategy, check_current_bounds,
                                          electrical_estimate, encoding_metrics,
                                          gram_matrix, to_mt_per_m)
from core.linear_encoding_model import (UnitStatus, build_model,
                                        simulate_currents)
from core.linear_encoding_pareto import (SOLVER_TOL, max_gradient_socp,
                                         position_sigma)
from core.linear_encoding_waveform import (WaveformSpec, check_nyquist,
                                           generate, load_csv_waveform)
from core.spatial_grid import (COVERAGE_THRESHOLD, RAS_AXES, sphere_support,
                               unit_vector, world_coordinates)

__all__ = ["prepare_region", "run_simulation", "MODE_TARGET", "MODE_DIRECT",
           "RegionError", "SimulationError"]

MODE_TARGET = "target_gradient"
MODE_DIRECT = "direct_currents"

#: Fewer valid voxels than this and a linear fit is not worth trusting.
MIN_ROI_VOXELS = 200


class RegionError(RuntimeError):
    """The region could not be built, or is not usable."""


class SimulationError(RuntimeError):
    """The simulation cannot be run as asked. Carries a user-facing message."""

    def __init__(self, message, detail=""):
        super().__init__(message)
        self.detail = detail


# ── loading ─────────────────────────────────────────────────────────
def load_coil_native(coil_name, progress=None):
    """Field maps on the coil's own calibration grid, plus its valid support.

    Native grid on purpose: sending a field map across grids and back lost 36%
    RMS on the 36-channel array, which is far too much to sit underneath a
    capability number. Only a binary ROI is ever moved, and only one way.
    """
    import nibabel as nib

    from core.coil_registry import find_fieldmap_nii, get_coil

    coil = get_coil(coil_name)
    dirs = coil.channel_dirs()
    if not dirs:
        raise RegionError("线圈「%s」没有找到通道场图。" % coil_name)
    first = find_fieldmap_nii(dirs[0])
    if first is None:
        raise RegionError("通道 %s 下没有场图文件。" % dirs[0].name)
    img0 = nib.load(str(first))
    shape, affine = tuple(img0.shape[:3]), np.asarray(img0.affine, dtype=float)

    maps = np.empty(shape + (len(dirs),), dtype=np.float32)
    names = []
    for j, d in enumerate(dirs):
        f = find_fieldmap_nii(d)
        if f is None:
            raise RegionError("通道 %s 下没有场图文件。" % d.name)
        img = nib.load(str(f))
        if tuple(img.shape[:3]) != shape or not np.allclose(img.affine, affine,
                                                            atol=1e-4):
            raise RegionError(
                "通道 %s 的网格与首通道不一致，拒绝混用。" % d.name)
        maps[..., j] = img.get_fdata(dtype=np.float32)
        names.append(d.name)
        if progress:
            progress(int(5 + 25 * (j + 1) / len(dirs)),
                     "载入通道场图 %d/%d" % (j + 1, len(dirs)))

    finite = np.all(np.isfinite(maps), axis=-1)
    support = finite & (np.abs(maps).max(axis=-1) > 0)
    if not support.any():
        raise RegionError("场图中没有任何有效体素。")
    return {
        "coil_name": coil.name, "maps": maps, "affine": affine,
        "shape": shape, "support": support, "channel_names": names,
        "n_channels": len(dirs),
        "current_limit_a": float(coil.current_limit_a),
        "grid_occupancy": float(support.sum() / support.size),
        "_occupancy_note": ("nonzero voxels / rectangular grid voxels; this is "
                            "occupancy, not DSV coverage"),
    }


def energy_centroid(loaded):
    """Where the array actually has field, in RAS metres."""
    pts = world_coordinates(loaded["affine"], loaded["shape"])
    flat = loaded["maps"].reshape(-1, loaded["n_channels"]).astype(np.float64)
    e = (flat ** 2).sum(axis=1)
    e[~loaded["support"].ravel()] = 0.0
    tot = float(e.sum())
    if tot <= 0:
        raise RegionError("场图能量为零，无法定义能量重心。")
    return (e[:, None] * pts).sum(axis=0) / tot


def isocentre(loaded):
    """The calibration grid's own centre, in RAS metres."""
    return world_coordinates(loaded["affine"], loaded["shape"]).mean(axis=0)


def prepare_region(params, progress=None) -> dict:
    """Build the region and say how much of it the calibration covers.

    `params` needs `coil`, `region_type` ('dsv' or 'roi') and either
    (`dsv_cm`, `centre_method`) or `roi_path`.
    """
    def rep(pct, msg):
        if progress:
            progress(pct, msg)

    rep(2, "载入线圈配置")
    loaded = load_coil_native(params["coil"], progress)

    if params.get("region_type", "dsv") == "dsv":
        method = params.get("centre_method", "isocentre")
        rep(60, "定位球心")
        if method == "energy_centroid":
            centre = energy_centroid(loaded)
        elif method == "isocentre":
            centre = isocentre(loaded)
        else:
            raise RegionError("未知的球心定义：%r" % method)
        dsv_m = float(params.get("dsv_cm", 8.0)) / 100.0
        rep(75, "计算 DSV 覆盖率")
        info = sphere_support(loaded["affine"], loaded["shape"], centre, dsv_m,
                              loaded["support"])
        region = dict(info)
        region.update({"region_type": "dsv", "centre_method": method,
                       "centre_ras_m": [float(v) for v in centre]})
    elif params.get("roi_array") is not None:
        # The one-click page already has the ROI as an array on a known grid.
        # Re-reading it from disk would ask the user to find a file the app is
        # holding, and typing a folder path instead of a file is what produced
        # the PermissionError.
        rep(60, "使用一键匀场的 ROI")
        region = _roi_from_array(params["roi_array"], params.get("roi_affine"),
                                 loaded, params.get("roi_label", "一键匀场 ROI"))
    else:
        path = params.get("roi_path") or ""
        if not path:
            raise RegionError("未选择 ROI：请先在一键匀场页完成一次优化，或指定"
                              "一个 ROI 掩膜文件。")
        rep(60, "载入 ROI")
        region = _load_roi(path, loaded)

    n = int(region["mask"].sum())
    if n < MIN_ROI_VOXELS:
        raise RegionError(
            "区域内只有 %d 个有效体素（下限 %d），线性拟合不可信。"
            % (n, MIN_ROI_VOXELS))

    region.update({
        "coil_name": loaded["coil_name"],
        "n_channels": loaded["n_channels"],
        "channel_names": loaded["channel_names"],
        "current_limit_a": loaded["current_limit_a"],
        "grid_occupancy": loaded["grid_occupancy"],
        "shape": list(loaded["shape"]),
        "affine": loaded["affine"].tolist(),
        "voxels": n,
        "coverage_threshold": COVERAGE_THRESHOLD,
    })
    region["_loaded"] = loaded          # kept for the run; not serialised
    rep(100, "完成")
    return region


def _roi_from_array(mask, affine, loaded, label):
    """An ROI already in memory, moved onto the coil grid if it is elsewhere."""
    import numpy as _np

    src = _np.asarray(mask)
    src = src > 0.5 if src.dtype != bool else src
    aff = (loaded["affine"] if affine is None
           else _np.asarray(affine, dtype=float))
    same = (tuple(src.shape[:3]) == loaded["shape"]
            and _np.allclose(aff, loaded["affine"], atol=1e-4))
    if same:
        out = src & loaded["support"]
        method = "none (ROI already on the coil grid)"
    else:
        from core.fsl_replace import _apply_affine

        moved = _apply_affine(src.astype(_np.float32), aff, loaded["shape"],
                              loaded["affine"], _np.eye(4), "nearest") > 0.5
        out = moved & loaded["support"]
        method = "nearest-neighbour, ROI -> coil grid, one way"
    n_src = int(src.sum())
    return {
        "mask": out, "region_type": "roi", "roi_path": "(in memory)",
        "roi_label": label, "resample_method": method,
        "roi_source_shape": list(src.shape[:3]),
        "roi_source_affine": aff.tolist(),
        "roi_source_voxels": n_src,
        "valid_sphere_voxels": int(out.sum()),
        "theoretical_sphere_voxels": n_src,
        "coverage": float(out.sum() / max(n_src, 1)),
        "truncated": False,
        "region_name": label,
    }


def _validate_roi_path(path):
    """A readable NIfTI file, or an error that says what is actually wrong.

    A directory opens with PermissionError on Windows, which surfaced to the
    user as a permissions problem when the real problem was that they had
    typed a folder.
    """
    from pathlib import Path as _P

    p = _P(str(path).strip().strip('"'))
    if not p.exists():
        raise RegionError("ROI 路径不存在：%s" % p)
    if p.is_dir():
        raise RegionError(
            "这是一个目录，不是掩膜文件：%s\n请指向具体的 .nii 或 .nii.gz 文件。" % p)
    if p.suffix.lower() not in (".nii", ".gz") and "".join(p.suffixes[-2:]).lower() != ".nii.gz":
        raise RegionError("ROI 文件后缀不是 .nii/.nii.gz：%s" % p.name)
    try:
        with p.open("rb") as fh:
            fh.read(1)
    except PermissionError:
        raise RegionError("没有读取权限：%s" % p)
    except OSError as e:
        raise RegionError("无法读取 %s：%s" % (p.name, e))
    return p


def _load_roi(path, loaded):
    """Binary ROI onto the coil grid. Nearest neighbour, one way, never back."""
    import nibabel as nib

    path = _validate_roi_path(path)
    try:
        img = nib.load(str(path))
    except Exception as e:
        raise RegionError("%s 不是可读的 NIfTI：%s" % (Path(path).name, e))
    src = np.asanyarray(img.dataobj) > 0.5
    same = (tuple(img.shape[:3]) == loaded["shape"]
            and np.allclose(img.affine, loaded["affine"], atol=1e-4))
    if same:
        mask = src & loaded["support"]
        method = "none (ROI already on the coil grid)"
    else:
        from core.fsl_replace import _apply_affine

        moved = _apply_affine(src.astype(np.float32), np.asarray(img.affine),
                             loaded["shape"], loaded["affine"], np.eye(4),
                             "nearest") > 0.5
        mask = moved & loaded["support"]
        method = "nearest-neighbour, ROI -> coil grid, one way"
    return {
        "mask": mask, "region_type": "roi", "roi_path": str(path),
        "resample_method": method,
        "roi_source_shape": list(img.shape[:3]),
        "roi_source_affine": np.asarray(img.affine).tolist(),
        "roi_source_voxels": int(src.sum()),
        "valid_sphere_voxels": int(mask.sum()),
        "theoretical_sphere_voxels": int(src.sum()),
        "coverage": float(mask.sum() / max(int(src.sum()), 1)),
        "truncated": False,
        "region_name": "被试 ROI",
        "_overlap_note": ("coverage here is ROI voxels surviving on the coil "
                          "grid divided by ROI voxels in the source"),
    }


# ── the run ─────────────────────────────────────────────────────────
def run_simulation(params, region, progress=None) -> dict:
    """Everything from a built region to a result dict. Pure computation."""
    def rep(pct, msg):
        if progress:
            progress(pct, msg)

    if params.get("unit_status") != UnitStatus.USER_CONFIRMED:
        raise SimulationError(
            "未确认场图单位为 Hz/A，拒绝运行正式仿真。",
            "unit_status=%r" % params.get("unit_status"))

    loaded = region.get("_loaded")
    if loaded is None:
        raise SimulationError("请先计算区域与覆盖率。")
    mode = params.get("mode", MODE_TARGET)
    imax = float(region["current_limit_a"])

    if (region.get("truncated") and mode == MODE_TARGET):
        raise SimulationError(
            "该球形区域覆盖率 %.1f%% 低于 %.0f%%，属于「球形区域 ∩ 有效标定范围」，"
            "不能据此给出模型可达上界；请缩小 DSV 或改用直接电流模式。"
            % (100 * region["coverage"], 100 * COVERAGE_THRESHOLD))

    rep(10, "构建编码模型")
    model = build_model(loaded["maps"], region["mask"], loaded["affine"],
                        channel_names=loaded["channel_names"],
                        unit_status=UnitStatus.USER_CONFIRMED)

    i_static = np.asarray(params.get("static_currents")
                          or np.zeros(model.n_channels), dtype=float).ravel()
    if i_static.shape[0] != model.n_channels:
        raise SimulationError(
            "静态电流有 %d 项，线圈有 %d 个通道。"
            % (i_static.shape[0], model.n_channels))
    if np.any(np.abs(i_static) > imax + 1e-9):
        raise SimulationError("静态匀场基线本身已超出 ±%.1f A。" % imax)

    out = {
        "mode": mode, "coil": region["coil_name"],
        "n_channels": model.n_channels,
        "channel_names": model.channel_names,
        "region_name": region["region_name"],
        "region_type": region["region_type"],
        "coverage": region.get("coverage"),
        "truncated": bool(region.get("truncated")),
        "roi_voxels": model.n_voxels,
        "current_limit_a": imax,
        "unit_status": UnitStatus.USER_CONFIRMED,
        "has_static": bool(np.any(np.abs(i_static) > 0)),
        "static_currents_a": i_static.tolist(),
        "static_peak_a": float(np.abs(i_static).max()),
        "spectrum": gram_matrix(model),
        "idealised": True,
    }

    if mode == MODE_TARGET:
        rep(25, "求解模型可达上界")
        _run_target(params, model, i_static, imax, out, rep)
    elif mode == MODE_DIRECT:
        rep(25, "读取电流表")
        _run_direct(params, model, i_static, imax, out, rep)
    else:
        raise SimulationError("未知的仿真模式：%r" % mode)

    rep(100, "完成")
    return out


def _finish(model, t, i_dynamic, i_static, imax, params, out, rep):
    """Strategy, simulation and metrics, shared by both modes."""
    strategy = params.get("strategy", "reject")
    rep(65, "检查 ±%.1f A 电流边界" % imax)
    applied = apply_strategy(t, i_dynamic, i_static, imax, strategy,
                             model.channel_names)
    scaled_dyn = applied["dynamic_currents"]

    rep(78, "合成时变场")
    # Recomputed from the scaled currents, never by scaling an old result: a
    # uniform scale changes the current vector, and every gradient, phase and
    # k number follows from that vector.
    sim = simulate_currents(model, t, scaled_dyn, i_static,
                            {"strategy": strategy,
                             "uniform_scale": applied["scale"]})

    rep(88, "计算指标")
    d = unit_vector(params.get("direction", RAS_AXES["RAS-X"]))
    peak = sim.peak_index()
    m = encoding_metrics(model, sim.i_dynamic[peak], direction=d)
    sigma = position_sigma(model, d)
    par = m["gradient_along_target_hz_per_m"]
    g_dyn_peak = float(np.abs(sim.g_dynamic @ d).max())

    bounds = check_current_bounds(t, sim.i_total, imax, i_static,
                                  model.channel_names)
    at_bound = [c["channel"] for c in bounds["per_channel"] if
                abs(c["peak_abs_a"] - imax) < 1e-6 or c["saturated"]]

    out.update({
        "uniform_scale": applied["scale"],
        "scaled": applied["scaled"],
        "strategy": strategy,
        "requested_peak_dynamic_a": applied["requested_peak_dynamic_a"],
        "feasible_peak_dynamic_a": applied["feasible_peak_dynamic_a"],
        "actual_peak_gradient_hz_per_m": g_dyn_peak,
        "actual_peak_gradient_mt_per_m": float(to_mt_per_m(g_dyn_peak)),
        "gradient_parallel_actual_hz_per_m": par,
        "gradient_parallel_actual_mt_per_m": float(to_mt_per_m(par)),
        "gradient_magnitude_actual_hz_per_m":
            m["achieved_gradient_magnitude_hz_per_m"],
        "direction_error_deg": m["direction_error_deg"],
        "orthogonal_leakage_ratio": (m["orthogonal_leakage_hz_per_m"]
                                     / max(par, 1e-30)),
        "orthogonal_leakage_hz_per_m": m["orthogonal_leakage_hz_per_m"],
        "nonlinear_spatial_rms_hz": m["nonlinear_rms_hz"],
        "nonlinear_spatial_p95_hz": m["nonlinear_p95_hz"],
        "nonlinear_spatial_ratio": (m["nonlinear_rms_hz"]
                                    / max(par * sigma, 1e-30)),
        "constant_offset_hz": m["constant_offset_hz"],
        "position_sigma_m": sigma,
        "peak_index": peak, "peak_time_s": float(t[peak]),
        # dynamic vs total, kept apart everywhere
        "current_peak_dynamic_a": float(np.abs(sim.i_dynamic).max()),
        "current_rms_dynamic_a": float(np.sqrt(np.mean(sim.i_dynamic ** 2))),
        "current_peak_total_a": float(np.abs(sim.i_total).max()),
        "current_rms_total_a": float(np.sqrt(np.mean(sim.i_total ** 2))),
        "k_dynamic_peak_cycles_per_m":
            float(np.linalg.norm(sim.k_dynamic, axis=1).max()),
        "k_dynamic_end_cycles_per_m":
            float(np.linalg.norm(sim.k_dynamic[-1])),
        "k_total_local_peak_cycles_per_m":
            float(np.linalg.norm(sim.k_total_local_model, axis=1).max()),
        "k_total_local_end_cycles_per_m":
            float(np.linalg.norm(sim.k_total_local_model[-1])),
        "k_note": ("k_dynamic 是新增动态 B0 线性分量的积分；k_total_local_model "
                   "含本地线圈 static+dynamic。两者都不包含扫描仪内置成像梯度，"
                   "非线性空间场也不能表示为单一 k 向量。"),
        "channels_at_current_bound": at_bound,
        "bounds": bounds,
        "electrical": electrical_estimate(t, sim.i_total,
                                          channel_names=model.channel_names),
        "solver_tolerances": SOLVER_TOL,
    })
    out["_sim"] = sim
    out["_model"] = model
    out["_t"] = t
    return out


def _run_target(params, model, i_static, imax, out, rep):
    d = unit_vector(params.get("direction", RAS_AXES["RAS-X"]))
    eps = float(params["epsilon_nonlinear"])
    theta = params.get("theta_max_deg", 5.0)

    cap = max_gradient_socp(model, d, eps, imax, i_static, True,
                            theta_max_deg=theta)
    out["capacity"] = {k: v for k, v in cap.items()
                       if not k.startswith("_") and k != "bounds"}
    out["epsilon_nonlinear"] = eps
    out["theta_max_deg"] = theta
    if not cap.get("feasible"):
        raise SimulationError(
            "当前场图模型、区域和约束下未找到非零可行解。"
            "（ε=%.2f，θ≤%s，±%.1f A，区域 %s）"
            % (eps, "无锥" if theta is None else "%.1f°" % theta, imax,
               out["region_name"]),
            cap.get("note", cap.get("status", "")))

    beta_capacity = cap["gradient_parallel_actual_hz_per_m"]
    i_capacity = np.asarray(cap["currents_a"], dtype=float)
    i_per_gradient = i_capacity / beta_capacity          # A per (Hz/m)

    rep(45, "生成波形")
    spec = WaveformSpec(kind=params.get("waveform", "sine"),
                        amplitude=float(params["requested_peak_hz_per_m"]),
                        frequency=float(params.get("frequency_hz", 100.0)),
                        duration_s=float(params.get("duration_s", 0.02)),
                        sample_rate_hz=float(params.get("sample_rate_hz",
                                                        100000.0)),
                        ramp_s=float(params.get("ramp_s", 0.0)))
    t, g_req = generate(spec)
    i_dynamic = g_req[:, None] * i_per_gradient[None, :]

    out.update({
        "waveform": spec.to_dict(),
        "nyquist": check_nyquist(spec.sample_rate_hz, spec.frequency,
                                 spec.kind),
        "requested_peak_gradient_hz_per_m": float(np.abs(g_req).max()),
        "requested_peak_gradient_mt_per_m":
            float(to_mt_per_m(np.abs(g_req).max())),
        "capacity_gradient_hz_per_m": beta_capacity,
        "capacity_gradient_mt_per_m": float(to_mt_per_m(beta_capacity)),
        "capacity_note": ("当前 ROI/DSV、ε、θ 与 ±3 A 静态电流箱约束下的"
                          "模型可达上界，不是功放动态能力。"),
        "i_per_gradient_a_per_hz_per_m": i_per_gradient.tolist(),
        "direction": d.tolist(),
        "requested_waveform_hz_per_m": g_req.tolist(),
    })
    _finish(model, t, i_dynamic, i_static, imax, params, out, rep)
    sim = out["_sim"]
    out["actual_waveform_hz_per_m"] = (sim.g_dynamic @ d).tolist()
    return out


def _run_direct(params, model, i_static, imax, out, rep):
    path = params.get("csv_path") or ""
    if not path:
        raise SimulationError("直接电流模式需要一个 CSV 电流表。")
    t, vals, cols = load_csv_waveform(path, expect_channels=model.n_channels)
    rep(45, "读入 %d 个采样点 × %d 通道" % (t.size, vals.shape[1]))
    out.update({
        "csv_path": str(path), "csv_columns": cols,
        "waveform": {"kind": "custom_csv", "sample_count": int(t.size),
                     "duration_s": float(t[-1] - t[0])},
        "epsilon_nonlinear": None, "theta_max_deg": None,
        "capacity": None,
        "direct_mode_note": ("直接电流模式不优化场形；ε 和 θ 仅用于目标梯度"
                             "模式，此处不作为约束。"),
        "direction": unit_vector(params.get("direction",
                                            RAS_AXES["RAS-X"])).tolist(),
    })
    _finish(model, t, vals, i_static, imax, params, out, rep)
    d = unit_vector(out["direction"])
    out["actual_waveform_hz_per_m"] = (out["_sim"].g_dynamic @ d).tolist()
    return out
