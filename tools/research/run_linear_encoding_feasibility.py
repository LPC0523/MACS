"""Linear-encoding feasibility: 12ch PFC and 36ch whole-brain, same maths.

    python tools/research/run_linear_encoding_feasibility.py

IDEALIZED OFFLINE ENCODING SIMULATION. No serial port, no upload, no change to
any production default, no packaging.

Writes `results/linear_encoding_feasibility/`:
    operating_points.csv   one row per (coil, ROI, direction, method)
    summary.json           the same data plus grid audits and provenance
    report.md              the tables, with the caveats attached

Every mT/m in every output comes from `to_mt_per_m`, so a table, a CSV and the
JSON cannot disagree about gamma.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core.coil_registry import find_fieldmap_nii, get_coil  # noqa: E402
from core.data_context import same_grid  # noqa: E402
from core.linear_encoding_metrics import (GAMMA_HZ_PER_MT,  # noqa: E402
                                          gram_matrix, sinusoid_phase_terms,
                                          to_mt_per_m)
from core.linear_encoding_model import UnitStatus, build_model  # noqa: E402
from core.linear_encoding_pareto import (EPSILON_LADDER,  # noqa: E402
                                         THETA_LADDER, beta_ray_limited,
                                         max_gradient_socp)
from core.pipeline import _load_coil_maps  # noqa: E402
from core.spatial_grid import RAS_AXES  # noqa: E402

OUT = ROOT / "results" / "linear_encoding_feasibility"
IMAX = 3.0
DIRS = [("RAS-X", RAS_AXES["RAS-X"]), ("RAS-Y", RAS_AXES["RAS-Y"]),
        ("RAS-Z", RAS_AXES["RAS-Z"])]
FREQS = (50.0, 100.0, 200.0, 500.0, 1000.0)
FOV_M = 0.22

ROW_KEYS = ("coil", "n_channels", "subject", "roi", "roi_voxels", "direction",
            "method", "epsilon", "theta_max_deg", "feasible", "shape_reachable",
            "beta_hz_per_m", "beta_mt_per_m",
            "achieved_parallel_hz_per_m", "achieved_parallel_mt_per_m",
            "gradient_gain", "nrmse", "r_squared", "correlation",
            "direction_error_deg", "orthogonal_leakage_hz_per_m",
            "nonlinear_rms_hz", "nonlinear_p95_hz", "nonlinear_max_hz",
            "constant_offset_hz", "linear_fraction",
            "current_peak_a", "current_rms_a", "n_saturated",
            "saturated_channels", "status", "solver",
            "box_violation_a", "shape_constraint_relative_violation",
            "clipping_applied", "currents_a")


def grid_audit(coil, ref_shape, ref_affine, log):
    """Everything that must match, checked rather than assumed."""
    dirs = coil.channel_dirs()
    img = nib.load(str(find_fieldmap_nii(dirs[0])))
    calib_vox = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    ref_vox = np.sqrt((np.asarray(ref_affine)[:3, :3] ** 2).sum(axis=0))
    # Direction cosines say whether two grids are merely the same size or
    # actually the same orientation.
    cos_c = img.affine[:3, :3] / calib_vox
    cos_r = np.asarray(ref_affine)[:3, :3] / ref_vox
    same = bool(same_grid(img.shape[:3], img.affine, tuple(ref_shape),
                          ref_affine))
    audit = {
        "coil": coil.name,
        "n_channels_config": int(coil.channel_count),
        "n_channels_found": len(dirs),
        "current_limit_a": float(coil.current_limit_a),
        "field_unit": "Hz/A",
        "field_unit_status": UnitStatus.USER_CONFIRMED,
        "channel_order": [d.name for d in dirs],
        "calibration_shape": list(img.shape[:3]),
        "calibration_voxel_mm": [round(float(v), 4) for v in calib_vox],
        "calibration_affine": np.asarray(img.affine).tolist(),
        "subject_shape": [int(v) for v in ref_shape],
        "subject_voxel_mm": [round(float(v), 4) for v in ref_vox],
        "subject_affine": np.asarray(ref_affine).tolist(),
        "orientation_cosine_max_diff": float(np.abs(cos_c - cos_r).max()),
        "same_grid": same,
        "resample_required": not same,
        "interpolation": ("none" if same else
                          "trilinear through world coordinates, single pass "
                          "(core.pipeline._load_coil_maps)"),
    }
    log("  grid audit: %s" % coil.name)
    log("    channels        : %d found / %d in config, limit +-%.1f A"
        % (len(dirs), coil.channel_count, coil.current_limit_a))
    log("    calibration     : %s @ %s mm" % (tuple(img.shape[:3]),
                                              np.round(calib_vox, 2)))
    log("    subject         : %s @ %s mm" % (tuple(ref_shape),
                                              np.round(ref_vox, 2)))
    log("    orientation diff: %.2e (direction cosines)"
        % audit["orientation_cosine_max_diff"])
    log("    same_grid       : %s -> %s"
        % (same, "no resampling" if same else "trilinear resample"))
    return audit


def resample_roundtrip_error(coil, ref_shape, ref_affine, maps, log):
    """Resample back to the calibration grid and measure what changed.

    An interpolation error quoted as "trilinear" says nothing about size. This
    sends channel 1 out and back and reports the round-trip discrepancy, which
    bounds the one-way error from above.
    """
    from core.fsl_replace import _apply_affine

    d0 = coil.channel_dirs()[0]
    img = nib.load(str(find_fieldmap_nii(d0)))
    orig = img.get_fdata(dtype=np.float32)
    try:
        back = _apply_affine(maps[..., 0].astype(np.float32), np.asarray(ref_affine),
                             img.shape[:3], img.affine, np.eye(4), "trilinear")
    except Exception as e:
        log("    round trip      : not measured (%s)" % type(e).__name__)
        return {"measured": False, "reason": str(e)[:120]}
    m = np.abs(orig) > 0
    if not m.any():
        return {"measured": False, "reason": "no nonzero calibration voxels"}
    err = back[m] - orig[m]
    out = {"measured": True,
           "rms_error_hz_per_a": float(np.sqrt(np.mean(err ** 2))),
           "max_abs_error_hz_per_a": float(np.abs(err).max()),
           "reference_rms_hz_per_a": float(np.sqrt(np.mean(orig[m] ** 2)))}
    out["relative_rms"] = out["rms_error_hz_per_a"] / out["reference_rms_hz_per_a"]
    log("    round trip      : RMS %.3f Hz/A on a %.1f Hz/A field (%.1f%%)"
        % (out["rms_error_hz_per_a"], out["reference_rms_hz_per_a"],
           100 * out["relative_rms"]))
    return out


def flatten(base, r):
    row = dict(base)
    row.update({k: r.get(k) for k in ROW_KEYS if k in r})
    sat = r.get("saturated_channels") or []
    row["n_saturated"] = len(sat)
    row["saturated_channels"] = ";".join(sat)
    cur = r.get("currents_a")
    row["currents_a"] = (";".join("%.5f" % v for v in cur) if cur else "")
    return row


def analyse_roi(coil, subject, roi_name, maps, mask, affine, names, log, rows):
    m = build_model(maps, mask, affine, channel_names=names,
                    unit_status=UnitStatus.USER_CONFIRMED)
    g = gram_matrix(m)
    base = {"coil": coil.name, "n_channels": m.n_channels, "subject": subject,
            "roi": roi_name, "roi_voxels": m.n_voxels}

    log("")
    log("  %s / %s / %s  (%d voxels)" % (coil.name, subject, roi_name,
                                         m.n_voxels))
    log("    effective rank %d/%d (1e-3), %d (1e-2); cond %.3g"
        % (g["effective_rank_1e-3"], g["n_channels"],
           g["effective_rank_1e-2"], g["condition_number"]))
    sv = np.asarray(g["singular_values_normalised"])
    log("    normalised sv: " + " ".join("%.3f" % v for v in sv[:10])
        + (" ..." if sv.size > 10 else ""))

    log("    %-7s %-10s %10s %10s %6s %6s %6s %8s %8s %6s"
        % ("dir", "method", "deliv Hz/m", "deliv mT/m", "gain", "NRMSE", "R2",
           "dir_err", "nonlin", "sat"))
    per_dir_best = {}
    for dname, d in DIRS:
        ray = beta_ray_limited(m, d, IMAX, None, True)
        rr = dict(ray["metrics"])
        rr.update({
            "method": "beta_min_norm_ray_limit", "epsilon": None,
            "theta_max_deg": None, "feasible": True, "shape_reachable": True,
            "beta_hz_per_m": ray["beta_ray_limited_hz_per_m"],
            "beta_mt_per_m": ray["beta_ray_limited_mt_per_m"],
            "achieved_parallel_hz_per_m": ray["achieved_gradient_hz_per_m"],
            "achieved_parallel_mt_per_m": ray["achieved_gradient_mt_per_m"],
            "gradient_gain": ray["gradient_gain"],
            "currents_a": ray["peak_currents"],
            "saturated_channels": [ray["binding_channel"]],
            "status": "closed-form", "solver": "lstsq",
            "box_violation_a": 0.0,
            "shape_constraint_relative_violation": 0.0,
            "clipping_applied": False,
        })
        rows.append(flatten(dict(base, direction=dname), rr))
        log("    %-7s %-10s %10.0f %10.4f %6.3f %6.3f %6.3f %8.2f %8.1f %6d"
            % (dname, "ray", rr["achieved_parallel_hz_per_m"],
               rr["achieved_parallel_mt_per_m"], rr["gradient_gain"],
               rr["nrmse"], rr["r_squared"], rr["direction_error_deg"],
               rr["nonlinear_rms_hz"], 1))

        best = rr["achieved_parallel_hz_per_m"]
        for eps in EPSILON_LADDER:
            r = max_gradient_socp(m, d, eps, IMAX, None, True)
            rows.append(flatten(dict(base, direction=dname), r))
            if not r.get("feasible"):
                log("    %-7s eps=%-6.2f  -- shape unreachable" % ("", eps))
                continue
            best = max(best, r["achieved_parallel_hz_per_m"])
            log("    %-7s eps=%-6.2f %10.0f %10.4f %6.3f %6.3f %6.3f %8.2f "
                "%8.1f %6d"
                % ("", eps, r["achieved_parallel_hz_per_m"],
                   r["achieved_parallel_mt_per_m"], r["gradient_gain"],
                   r["nrmse"], r["r_squared"], r["direction_error_deg"],
                   r["nonlinear_rms_hz"], len(r["saturated_channels"])))
        per_dir_best[dname] = best
        log("")

    # Direction cone at a fixed epsilon.
    log("    direction cone at eps=0.40:")
    log("    %-7s %8s %10s %8s %10s" % ("dir", "theta", "deliv Hz/m",
                                        "dir_err", "leak Hz/m"))
    for dname, d in DIRS:
        for th in (None,) + THETA_LADDER:
            r = max_gradient_socp(m, d, 0.40, IMAX, None, True,
                                  theta_max_deg=th)
            rows.append(flatten(dict(base, direction=dname), r))
            tag = "none" if th is None else "%.0f deg" % th
            if not r.get("feasible"):
                log("    %-7s %8s   -- infeasible" % (dname, tag))
                continue
            log("    %-7s %8s %10.0f %8.2f %10.1f"
                % (dname, tag, r["achieved_parallel_hz_per_m"],
                   r["direction_error_deg"], r["orthogonal_leakage_hz_per_m"]))
    log("")
    return m, g, per_dir_best


def sequence_table(best, log):
    log("    sequence scale (IDEALIZED): k_half = G/(pi f), cycles across "
        "%.0f cm FOV" % (FOV_M * 100))
    log("    %-7s %10s %8s %8s %8s %8s %8s"
        % ("dir", "deliv Hz/m", *["%g Hz" % f for f in FREQS]))
    out = {}
    for dname, gval in best.items():
        cyc = []
        for f in FREQS:
            p = sinusoid_phase_terms(gval, f, FOV_M / 2)
            cyc.append(p["k_half_cycle_cycles_per_m"] * FOV_M)
        out[dname] = {"delivered_hz_per_m": gval,
                      "cycles_across_fov": dict(zip(map(str, FREQS), cyc)),
                      "fov_m": FOV_M}
        log("    %-7s %10.0f %8.2f %8.2f %8.2f %8.2f %8.2f"
            % (dname, gval, *cyc))
    log("")
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    from common import find_subjects
    subs = {n: (hz, bp, pp) for n, hz, bp, pp in find_subjects(0, True)}

    rows, audits, seq = [], [], {}

    log("IDEALIZED OFFLINE ENCODING SIMULATION")
    log("gamma = 42.577e6 Hz/T = %.0f Hz/mT; every mT/m below is Hz/m / %.0f"
        % (GAMMA_HZ_PER_MT, GAMMA_HZ_PER_MT))
    log("Delivered gradient = achieved_parallel = d.(G i). The SOCP's `beta`")
    log("is a nominal target and runs high, because plain under-delivery")
    log("counts as shape error: with an otherwise exact shape the optimum sits")
    log("at beta = delivered/(1-eps). Tables quote delivered.")
    log("")

    # ── 12ch PFC ────────────────────────────────────────────────────
    coil = get_coil("PFC 12ch")
    sub = "S30_ear"
    hz, bp, pp = subs[sub]
    img = nib.load(str(hz))
    b0shape = img.shape[:3]
    brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
    pfc = (np.asanyarray(nib.load(str(pp)).dataobj) > 0.5) & brain
    log("=" * 76)
    log("12ch PFC   subject %s" % sub)
    log("=" * 76)
    a = grid_audit(coil, b0shape, img.affine, log)
    maps, names = _load_coil_maps(coil, img.affine, b0shape, lambda *x: None)
    a["roundtrip"] = resample_roundtrip_error(coil, b0shape, img.affine, maps,
                                              log)
    a["roi_overlap"] = {"brain_voxels": int(brain.sum()),
                        "pfc_voxels": int(pfc.sum()),
                        "pfc_fraction_of_brain": float(pfc.sum() / brain.sum())}
    audits.append(a)
    for rname, mask in (("PFC", pfc), ("whole_brain", brain)):
        _, _, best = analyse_roi(coil, sub, rname, maps, mask, img.affine,
                                 names, log, rows)
        seq["12ch PFC / %s / %s" % (sub, rname)] = sequence_table(best, log)

    # ── 36ch whole brain ────────────────────────────────────────────
    coil36 = get_coil("36通道婴幼儿线圈_xiamen扫描")
    # I01 is processed into this results folder rather than the batch
    # index, so it is located by path. Substituting a different subject would
    # confound the coil with the anatomy, which is exactly what the comparison
    # is trying to avoid.
    liy = OUT / "i01_nii"
    sub36 = None
    if liy.is_dir():
        hzs = sorted(liy.glob("*_deltaB0_hertz.nii.gz"))
        bms = sorted(liy.glob("*_bet_mask.nii.gz"))
        if hzs and bms:
            sub36 = hzs[0].name[: -len("_deltaB0_hertz.nii.gz")]
            subs[sub36] = (hzs[0], bms[0], None)
    log("=" * 76)
    log("36ch whole-brain")
    log("=" * 76)
    if sub36 is None:
        log("  I01 is not in the processed-subject index, so the 36ch ROI")
        log("  comparison cannot be run from cached data in this pass. The")
        log("  coil-intrinsic DSV analysis stands; the ROI comparison is")
        log("  reported as NOT RUN rather than substituted with another")
        log("  subject, which would confound coil with anatomy.")
        audits.append({"coil": coil36.name, "roi_comparison": "not_run",
                       "reason": "no processed I01 entry in the subject index"})
    else:
        hz, bp, pp = subs[sub36]
        img36 = nib.load(str(hz))
        brain36 = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
        a36 = grid_audit(coil36, img36.shape[:3], img36.affine, log)
        maps36, names36 = _load_coil_maps(coil36, img36.affine,
                                          img36.shape[:3], lambda *x: None)
        a36["roundtrip"] = resample_roundtrip_error(coil36, img36.shape[:3],
                                                    img36.affine, maps36, log)
        a36["roi_overlap"] = {"brain_voxels": int(brain36.sum())}
        audits.append(a36)
        _, _, best = analyse_roi(coil36, sub36, "whole_brain", maps36, brain36,
                                 img36.affine, names36, log, rows)
        seq["36ch / %s / whole_brain" % sub36] = sequence_table(best, log)

    # ── outputs ─────────────────────────────────────────────────────
    with open(OUT / "operating_points.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_KEYS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    payload = {
        "_what": "Linear-encoding feasibility, offline. IDEALIZED.",
        "gamma_hz_per_mt": GAMMA_HZ_PER_MT,
        "conversion": "G[mT/m] = G[Hz/m] / 42577",
        "current_limit_a": IMAX,
        "roi_weights": "uniform",
        "epsilon_ladder": list(EPSILON_LADDER),
        "theta_ladder_deg": list(THETA_LADDER),
        "field_unit_status": UnitStatus.USER_CONFIRMED,
        "beta_vs_delivered": ("beta is nominal; delivered = d.(G i). With an "
                              "otherwise exact shape the optimum has "
                              "beta = delivered/(1-eps), so beta overstates "
                              "the deliverable gradient."),
        "not_shown": ["amplifier bandwidth", "R", "L", "mutual inductance",
                      "Vmax", "H(f)", "scanner/ADC synchronisation",
                      "gradient coupling transfer function"],
        "grid_audits": audits,
        "sequence_scale": seq,
        "operating_points": rows,
    }
    (OUT / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8")

    md = ["# Linear-encoding feasibility (IDEALIZED offline simulation)", "",
          "> This shows only whether a near-linear field is reachable. It does",
          "> not demonstrate MRI acceleration, does not address RF centre",
          "> encoding, does not show MB-EPI is usable, and does not establish",
          "> that any amplifier can follow these waveforms. R, L, mutual",
          "> inductance, Vmax, H(f) and scanner synchronisation are unknown.",
          "", "```"] + lines + ["```", ""]
    (OUT / "report.md").write_text("\n".join(md), encoding="utf-8")

    log("wrote %s (%d operating points)" % (OUT, len(rows)))


if __name__ == "__main__":
    main()
