"""Seven files describing one run. Nothing a machine could execute.

IDEALIZED OFFLINE SIMULATION.

Exports carry their own caveats. A CSV of currents that leaves this folder
without the epsilon, the theta, the region and the coverage it was computed
under is a number without a claim attached, and someone will eventually read it
as a hardware specification. Every file therefore repeats the identity of the
run, and no binary frame a device could consume is written at all.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from core.linear_encoding_metrics import GAMMA_HZ_PER_MT, to_mt_per_m

__all__ = ["export_run", "EXPORT_FILES", "IDEALIZED_NOTICE"]

EXPORT_FILES = ("simulation_config.json", "waveform.csv", "currents.csv",
                "gradient_waveform.csv", "k_offset.csv", "metrics.json",
                "report.md")

IDEALIZED_NOTICE = (
    "IDEALIZED OFFLINE SIMULATION — 未与扫描仪 ADC、梯度时序或功放同步；"
    "未验证电感、电压、带宽及梯度耦合；不能据此宣称 EPI 加速或图像改善。")


def _identity(res: dict) -> dict:
    """What every file repeats, so no number travels without its conditions."""
    return {
        "_idealized": IDEALIZED_NOTICE,
        "coil": res.get("coil"),
        "n_channels": res.get("n_channels"),
        "channel_names": res.get("channel_names"),
        "field_unit": "Hz/A",
        "unit_status": res.get("unit_status"),
        "region_name": res.get("region_name"),
        "region_type": res.get("region_type"),
        "coverage": res.get("coverage"),
        "truncated": res.get("truncated"),
        "roi_voxels": res.get("roi_voxels"),
        "mode": res.get("mode"),
        "epsilon_nonlinear": res.get("epsilon_nonlinear"),
        "theta_max_deg": res.get("theta_max_deg"),
        "current_limit_a": res.get("current_limit_a"),
        "strategy": res.get("strategy"),
        "uniform_scale": res.get("uniform_scale"),
        "has_static_baseline": res.get("has_static"),
        "static_currents_a": res.get("static_currents_a"),
        "gamma_hz_per_mt": GAMMA_HZ_PER_MT,
        "unknown_hardware": ["R", "L", "mutual inductance", "Vmax",
                             "amplifier bandwidth", "H(f)",
                             "scanner ADC synchronisation"],
        "gradient_coupling": "not_characterized",
    }


def _write_csv(path, header, rows, ident):
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# %s\n" % IDEALIZED_NOTICE)
        f.write("# coil=%s region=%s coverage=%s eps=%s theta=%s "
                "limit=%s A scale=%s\n"
                % (ident["coil"], ident["region_name"], ident["coverage"],
                   ident["epsilon_nonlinear"], ident["theta_max_deg"],
                   ident["current_limit_a"], ident["uniform_scale"]))
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_run(res: dict, out_dir) -> dict:
    """Write the seven files. Returns {name: path}. No NIfTI, no 4-D field."""
    sim = res.get("_sim")
    t = np.asarray(res.get("_t"))
    if sim is None or t.size == 0:
        raise ValueError("nothing to export: run a simulation first")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ident = _identity(res)
    names = list(res.get("channel_names") or [])
    written = {}

    # 1 ── configuration
    cfg = dict(ident)
    cfg.update({
        "waveform": res.get("waveform"),
        "nyquist": res.get("nyquist"),
        "direction_ras": res.get("direction"),
        "requested_peak_gradient_hz_per_m":
            res.get("requested_peak_gradient_hz_per_m"),
        "capacity_gradient_hz_per_m": res.get("capacity_gradient_hz_per_m"),
        "capacity_note": res.get("capacity_note"),
        "csv_path": res.get("csv_path"),
        "direct_mode_note": res.get("direct_mode_note"),
        "solver_tolerances": res.get("solver_tolerances"),
        "position_sigma_m": res.get("position_sigma_m"),
    })
    p = out / "simulation_config.json"
    p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    written["simulation_config.json"] = str(p)

    # 2 ── the requested and actual scalar gradient waveforms
    req = res.get("requested_waveform_hz_per_m")
    act = res.get("actual_waveform_hz_per_m") or []
    rows = []
    for k in range(t.size):
        rows.append([("%.9g" % t[k]),
                     ("%.6g" % req[k]) if req else "",
                     ("%.6g" % act[k]) if k < len(act) else ""])
    p = out / "waveform.csv"
    _write_csv(p, ["time_s", "requested_gradient_hz_per_m",
                   "actual_parallel_gradient_hz_per_m"], rows, ident)
    written["waveform.csv"] = str(p)

    # 3 ── currents, dynamic and total kept apart
    hdr = (["time_s"] + ["dyn_%s_a" % n for n in names]
           + ["total_%s_a" % n for n in names])
    rows = []
    for k in range(t.size):
        rows.append(["%.9g" % t[k]]
                    + ["%.6g" % v for v in sim.i_dynamic[k]]
                    + ["%.6g" % v for v in sim.i_total[k]])
    p = out / "currents.csv"
    _write_csv(p, hdr, rows, ident)
    written["currents.csv"] = str(p)

    # 4 ── gradient vectors, dynamic and total
    p = out / "gradient_waveform.csv"
    _write_csv(p, ["time_s", "gx_dyn_hz_per_m", "gy_dyn_hz_per_m",
                   "gz_dyn_hz_per_m", "gx_total_hz_per_m",
                   "gy_total_hz_per_m", "gz_total_hz_per_m"],
               [["%.9g" % t[k]] + ["%.6g" % v for v in sim.g_dynamic[k]]
                + ["%.6g" % v for v in sim.g_total[k]]
                for k in range(t.size)], ident)
    written["gradient_waveform.csv"] = str(p)

    # 5 ── idealised k offsets. Named for what they are.
    p = out / "k_offset.csv"
    _write_csv(p, ["time_s", "kx_dynamic_cycles_per_m",
                   "ky_dynamic_cycles_per_m", "kz_dynamic_cycles_per_m",
                   "kx_total_local_cycles_per_m",
                   "ky_total_local_cycles_per_m",
                   "kz_total_local_cycles_per_m"],
               [["%.9g" % t[k]] + ["%.6g" % v for v in sim.k_dynamic[k]]
                + ["%.6g" % v for v in sim.k_total_local_model[k]]
                for k in range(t.size)], ident)
    written["k_offset.csv"] = str(p)

    # 6 ── metrics
    met = dict(ident)
    met.update({k: v for k, v in res.items()
                if not k.startswith("_")
                and k not in ("channel_names", "requested_waveform_hz_per_m",
                              "actual_waveform_hz_per_m",
                              "i_per_gradient_a_per_hz_per_m")})
    p = out / "metrics.json"
    p.write_text(json.dumps(met, indent=2, ensure_ascii=False, default=str),
                 encoding="utf-8")
    written["metrics.json"] = str(p)

    # 7 ── report
    md = ["# 线性编码仿真报告", "", "> " + IDEALIZED_NOTICE, "",
          "| 项 | 值 |", "|---|---|",
          "| 线圈 | %s (%s 通道) |" % (res.get("coil"), res.get("n_channels")),
          "| 场图单位 | Hz/A (%s) |" % res.get("unit_status"),
          "| 区域 | %s，%s 体素 |" % (res.get("region_name"),
                                      res.get("roi_voxels")),
          "| 覆盖率 | %s%s |" % (
              ("%.1f%%" % (100 * res["coverage"])) if res.get("coverage")
              is not None else "—",
              "  **TRUNCATED**" if res.get("truncated") else ""),
          "| 模式 | %s |" % res.get("mode"),
          "| ε / θ | %s / %s |" % (res.get("epsilon_nonlinear"),
                                   res.get("theta_max_deg")),
          "| 电流限幅 | ±%.1f A |" % (res.get("current_limit_a") or 0.0),
          "| 超限策略 | %s，缩放因子 %.4f |" % (res.get("strategy"),
                                               res.get("uniform_scale", 1.0)),
          "| 静态基线 | %s |" % ("有，峰值 %.3f A" % res.get("static_peak_a", 0.0)
                                 if res.get("has_static") else "无"),
          "", "## 三个幅度", "", "| 量 | Hz/m | mT/m |", "|---|---|---|"]
    for label, key in (("请求峰值", "requested_peak_gradient_hz_per_m"),
                       ("模型可达上界", "capacity_gradient_hz_per_m"),
                       ("实际模型峰值", "actual_peak_gradient_hz_per_m")):
        v = res.get(key)
        md.append("| %s | %s | %s |"
                  % (label, "%.1f" % v if v is not None else "—",
                     "%.4f" % to_mt_per_m(v) if v is not None else "—"))
    md += ["",
           "「模型可达上界」是当前区域、ε、θ 与 ±%.1f A 静态电流箱约束下的模型"
           "上界，不是功放动态能力。" % (res.get("current_limit_a") or 0.0), "",
           "## 场形", "", "| 指标 | 值 |", "|---|---|",
           "| 沿目标方向梯度 | %.1f Hz/m |"
           % res.get("gradient_parallel_actual_hz_per_m", float("nan")),
           "| 方向误差 | %.2f° |" % res.get("direction_error_deg", float("nan")),
           "| 正交泄漏比 | %.3f |"
           % res.get("orthogonal_leakage_ratio", float("nan")),
           "| 非线性空间场分量 RMS | %.2f Hz |"
           % res.get("nonlinear_spatial_rms_hz", float("nan")),
           "| 非线性空间场分量 P95 | %.2f Hz |"
           % res.get("nonlinear_spatial_p95_hz", float("nan")),
           "| 非线性空间场分量比 | %.3f |"
           % res.get("nonlinear_spatial_ratio", float("nan")), "",
           "非线性空间场可以携带编码信息，但不能表示为单一笛卡尔 k 向量；"
           "是否有助于成像需要序列和模型化重建验证。", "",
           "## 电流", "", "| 指标 | dynamic | total |", "|---|---|---|",
           "| 峰值 (A) | %.3f | %.3f |"
           % (res.get("current_peak_dynamic_a", 0.0),
              res.get("current_peak_total_a", 0.0)),
           "| RMS (A) | %.3f | %.3f |"
           % (res.get("current_rms_dynamic_a", 0.0),
              res.get("current_rms_total_a", 0.0)),
           "", "达到 ±%.1f A 电流边界的通道：%s"
           % (res.get("current_limit_a") or 0.0,
              ", ".join(res.get("channels_at_current_bound") or []) or "无"),
           "", "## 理想化 k 偏移", "",
           "线性 B0 分量积分得到的理想化 k 偏移；不含扫描仪内置成像梯度。", "",
           "| 量 | 峰值 (cycles/m) | 终点 (cycles/m) |", "|---|---|---|",
           "| dynamic | %.4f | %.4f |"
           % (res.get("k_dynamic_peak_cycles_per_m", 0.0),
              res.get("k_dynamic_end_cycles_per_m", 0.0)),
           "| total local model | %.4f | %.4f |"
           % (res.get("k_total_local_peak_cycles_per_m", 0.0),
              res.get("k_total_local_end_cycles_per_m", 0.0)), "",
           "## 未知项", "",
           "电压、功率与动态可实现性：**UNKNOWN**（缺 R、L、互感、Vmax、"
           "放大器带宽与 H(f)）。梯度耦合：**not_characterized**。", "",
           "求解器：%s / %s。约束违反：箱 %.2e A。"
           % (res.get("capacity", {}).get("solver") if res.get("capacity")
              else "—",
              res.get("capacity", {}).get("status") if res.get("capacity")
              else "—",
              (res.get("bounds") or {}).get("global_peak_abs_a", 0.0)
              - (res.get("current_limit_a") or 0.0)
              if (res.get("bounds") or {}).get("global_peak_abs_a", 0.0)
              > (res.get("current_limit_a") or 0.0) else 0.0), ""]
    p = out / "report.md"
    p.write_text("\n".join(md), encoding="utf-8")
    written["report.md"] = str(p)
    return written
