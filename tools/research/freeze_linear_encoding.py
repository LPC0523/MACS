"""Freeze the linear-encoding numerical definitions and implementation.

    python tools/research/freeze_linear_encoding.py

What is frozen is the mathematics and the code that implements it -- not a
statement about what any hardware can do. The verified spatial ranges below
belong to the field maps that were loaded, identified by hash; a different
calibration gives different ranges and the code recomputes them.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

# tools/<group>/<script>.py -> tools/<group> -> tools -> project root.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core import linear_encoding_pareto as pareto  # noqa: E402
from core.coil_registry import find_fieldmap_nii, get_coil  # noqa: E402
from core.linear_encoding_metrics import (GAMMA_HZ_PER_MT,  # noqa: E402
                                          GAMMA_HZ_PER_T)
from core.linear_encoding_model import UnitStatus  # noqa: E402
from core.linear_encoding_waveform import WAVEFORMS  # noqa: E402
from core.spatial_grid import COVERAGE_THRESHOLD  # noqa: E402

OUT = ROOT / "results" / "preregistration"

TRACKED = [
    "core/spatial_grid.py",
    "core/linear_encoding_waveform.py",
    "core/linear_encoding_model.py",
    "core/linear_encoding_metrics.py",
    "core/linear_encoding_pareto.py",
    "tests/test_linear_encoding_waveform.py",
    "tests/test_linear_encoding_model.py",
    "tests/test_linear_encoding_constraints.py",
    "tests/test_linear_encoding_pareto.py",
    "tests/freeze_linear_encoding.py",
    "config/shim_delta_bounds.json",
]


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def coil_fingerprint(name):
    """Identity of the field maps a result was computed from."""
    try:
        coil = get_coil(name)
    except Exception as e:
        return {"coil": name, "error": "%s: %s" % (type(e).__name__, e)}
    dirs = coil.channel_dirs()
    h = hashlib.sha256()
    per = []
    for d in dirs:
        f = find_fieldmap_nii(d)
        if f is None:
            per.append({"channel": d.name, "file": None})
            continue
        digest = sha256_file(f)
        h.update(d.name.encode())
        h.update(bytes.fromhex(digest))
        per.append({"channel": d.name, "file": f.name, "sha256": digest[:16]})
    return {
        "coil": coil.name,
        "n_channels": len(dirs),
        "current_limit_a": float(coil.current_limit_a),
        "field_unit": "Hz/A",
        "field_unit_status": UnitStatus.USER_CONFIRMED,
        "channel_order": [d.name for d in dirs],
        "fieldmap_set_sha256": h.hexdigest(),
        "per_channel": per,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    record = {
        "_what": ("Frozen numerical definitions and implementation for the "
                  "linear-encoding feasibility simulator."),
        "_status": ("IDEALIZED OFFLINE SIMULATION. Research module. Not wired "
                    "into any production shim path, no serial port, no upload."),
        "_scope": ("What is frozen is the mathematics and the code. It is not "
                   "a general hardware capability claim: every number depends "
                   "on the field maps hashed below, the region, the current "
                   "bounds and both tolerances."),

        # 1 ── the objective
        "objective": {
            "maximise": "beta_actual = d^T G i",
            "note": ("The delivered gradient along d. There is no independent "
                     "nominal beta and none may be reintroduced: an earlier "
                     "version maximised a free beta against a relative fit "
                     "error, which rewarded plain under-delivery -- a field of "
                     "1200 Hz/m sits within 10% of a 1333 Hz/m target -- and "
                     "could prefer a solution delivering 900 at gain 0.7 over "
                     "one delivering 1000 at gain 1.0."),
            "constraints": [
                "d^T G i >= 0",
                "||(I - d d^T) G i||_2 <= tan(theta_max) * d^T G i",
                "sqrt(mean_W[(R i)^2]) <= epsilon_nonlinear * (d^T G i) * sigma_position",
                "lower_j <= i_j <= upper_j",
            ],
            "sigma_position": "sqrt(mean_W[(d^T (r - r0))^2]), metres",
            "reduction": ("Q = R^T W R / sum(W), Cholesky Q = L^T L, so the "
                          "residual constraint is ||L i|| <= eps * beta * sigma. "
                          "The million-row design never reaches CVXPY."),
        },

        # 2 ── the decomposition
        "decomposition": {
            "basis": "X = [1, x-x0, y-y0, z-z0] on the ROI, about its weighted centre",
            "per_channel": "B_j = X M_j + R_j",
            "constant_hz_per_a": "M[0]  -- reported, never counted as encoding",
            "G_hz_per_m_per_a": "M[1:4]  -- the linear part",
            "R_hz_per_a": "the nonlinear residual",
            "why_constant_excluded": ("a uniform field advances every spin's "
                                      "phase equally and carries no position "
                                      "information"),
        },

        # 3 ── units and frames
        "units": {
            "gamma_hz_per_t": GAMMA_HZ_PER_T,
            "gamma_hz_per_mt": GAMMA_HZ_PER_MT,
            "conversion": "G[mT/m] = G[Hz/m] / 42577",
            "single_entry_point": "core.linear_encoding_metrics.to_mt_per_m",
            "field_maps": "Hz/A",
            "field_unit_status": UnitStatus.USER_CONFIRMED,
            "time": "s", "current": "A", "field": "Hz",
            "gradient": "Hz/m", "phase": "rad", "k": "cycles/m",
        },
        "coordinates": {
            "frame": "NIfTI RAS+ world, metres",
            "x": "right(+) / left(-)", "y": "anterior(+) / posterior(-)",
            "z": "superior(+) / inferior(-)",
            "source": "affine only; array axes are never used as directions",
            "scanner_frame": ("NOT AVAILABLE. PatientPosition and "
                              "PhaseEncodingDirection are not read anywhere in "
                              "this project, so RAS-Y must not be presented as "
                              "the scanner phase-encoding direction."),
        },

        # 4 ── tolerances and limits
        "tolerances": {
            "theta_max_deg": {
                "meaning": "half angle of the cone on the gradient direction",
                "required": True,
                "note": ("Not optional. With only a nonlinearity budget the "
                         "optimum tilts hard -- 74 degrees with an orthogonal "
                         "leakage 3.5x the parallel component was measured on "
                         "the 12ch PFC ROI -- because the objective rewards the "
                         "parallel component and the nonlinearity constraint "
                         "does not penalise orthogonal linear leakage."),
                "ui_default_deg": 5.0,
                "swept": list(pareto.THETA_LADDER),
            },
            "epsilon_nonlinear": {
                "meaning": ("nonlinear residual RMS divided by "
                            "(delivered gradient * sigma_position)"),
                "required_in_every_headline": True,
                "swept": list(pareto.EPSILON_LADDER),
            },
            "current_limit_a": 3.0,
            "dynamic_bounds": {
                "symmetric_bipolar": "|i_j| <= Imax - |i_static_j|",
                "asymmetric_unipolar": "[-Imax - i_static, Imax - i_static]",
                "note": ("the symmetric form is the default and is what a "
                         "sinusoid needs; the asymmetric window suits a "
                         "single-sign blip and then the waveform must still be "
                         "checked sample by sample"),
            },
            "post_solve_clipping": False,
            "clipping_note": ("never available. Out of bounds is answered by "
                              "reject (default) or one uniform scale factor."),
            "coverage_threshold": COVERAGE_THRESHOLD,
            "coverage_rule": ("coverage = valid sphere voxels / theoretical "
                              "sphere voxels; below the threshold the region is "
                              "named 'sphere and valid support', is shown as "
                              "TRUNCATED, and no maximum-capability conclusion "
                              "may be drawn from it"),
        },

        # 5 ── solver
        "solver": {
            "backend": ["CLARABEL", "SCS (fallback)"],
            "form": "SOCP, linear objective, second-order cones",
            "violations_measured_on_the_returned_vector": True,
            "reported_violations": ["box_violation_a",
                                    "nonlinear_constraint_relative_violation",
                                    "direction_cone_relative_violation"],
            "monotonicity_tolerance_relative": 1e-4,
        },

        # 6 ── metric names
        "primary_metrics": [
            "gradient_parallel_actual_hz_per_m",
            "gradient_parallel_actual_mt_per_m",
            "gradient_magnitude_actual_hz_per_m",
            "direction_error_deg",
            "orthogonal_leakage_ratio",
            "nonlinear_residual_ratio",
            "nonlinear_residual_rms_hz",
        ],
        "secondary_metrics": {
            "target_fit_nrmse": ("||B_centered i - beta_actual t|| / "
                                 "||beta_actual t||; mixes orthogonal linear "
                                 "leakage with nonlinear residual and must not "
                                 "be used alone"),
        },
        "legacy_debug_only": ["_legacy_requested_beta_hz_per_m"],
        "waveforms": list(WAVEFORMS),

        # 7 ── what was verified
        "verification": {
            "synthetic": {
                "perfect_linear_coil": ("3 A x 400 Hz/m/A delivers 1200.00 Hz/m "
                                        "at epsilon 0.10, 0.30 and 0.50 alike; "
                                        "relaxing epsilon adds no gradient"),
                "adversarial": ("the SOCP matches an exhaustive 25x25 grid "
                                "search over the feasible box and never falls "
                                "below the clean single-channel solution"),
                "exact_recovery": "an exact X-gradient channel is recovered to 1e-9",
                "collinear_detection": "4 collinear channels give effective rank 1",
                "homogeneity": "doubling Imax doubles the gradient, ratios unchanged",
            },
            "monotonicity": {
                "epsilon": "6 rungs, largest relative drop 4.6e-12",
                "theta": "6 rungs, largest relative drop 0.0",
                "tolerance": 1e-4,
            },
            "memory": ("no (Nx, Ny, Nz, Nt) array is allocated; a 1.9 GB 4-D "
                       "field peaks at 1.9 MB via B(r,t) = C(r,:) i(t)"),
            "tests": ["test_linear_encoding_waveform.py",
                      "test_linear_encoding_model.py",
                      "test_linear_encoding_constraints.py",
                      "test_linear_encoding_pareto.py"],
            "production_smoke": "9/9",
        },

        # 8 ── verified spatial ranges, for THESE field maps
        "verified_ranges": {
            "_note": ("Belongs to the hashed field maps, not to the arrays in "
                      "general. No diameter is hard-coded anywhere; the UI "
                      "recomputes coverage from whatever maps are loaded."),
            "12ch PFC": {
                "real_PFC_ROI": "verified (ROI already on the coil grid)",
                "DSV_4cm_energy_centred": "verified, coverage 100%",
                "DSV_8cm_energy_centred": "verified, coverage 98%",
            },
            "36ch whole-brain": {
                "DSV_8cm_isocentre": "verified, coverage 100%",
                "DSV_12cm_isocentre": "verified, coverage 99.9%",
                "DSV_20cm_isocentre": ("NOT VERIFIED, coverage 46% -- "
                                       "TRUNCATED, excluded from conclusions"),
            },
        },

        # 9 ── still unknown
        "unknown_hardware": {
            "resistance_ohm": "unknown", "inductance_h": "unknown",
            "mutual_inductance_h": "unknown", "voltage_limit_v": "unknown",
            "amplifier_bandwidth": "unknown", "transfer_function_hf": "unknown",
            "scanner_adc_synchronisation": "unknown / trigger line not connected",
            "gradient_coupling": "not_characterized",
            "consequence": ("voltage, power and dynamic realisability are "
                            "reported as unknown, never as zero"),
        },
        "does_not_demonstrate": [
            "MRI acceleration of any factor",
            "that RF centre-encoding is solved",
            "that MB-EPI is usable",
            "that any amplifier can follow these waveforms",
            "any improvement in parallel-imaging conditioning",
        ],
    }

    record["field_map_identity"] = [
        coil_fingerprint("PFC 12ch"),
        coil_fingerprint("36通道婴幼儿线圈_xiamen扫描"),
    ]
    files = {}
    for rel in TRACKED:
        p = ROOT / rel
        files[rel] = sha256_file(p) if p.is_file() else None
    record["provenance"] = {
        "git_commit": None,
        "git_note": "SHA-256 per file; see CHANGELOG for the tagged build",
        "file_sha256": files,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }

    (OUT / "linear_encoding_freeze.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Linear-encoding core: frozen numerical definitions", "",
          "> IDEALIZED OFFLINE SIMULATION. Research module, not in any",
          "> production shim path. What is frozen is the mathematics and the",
          "> code -- not a hardware capability claim.", "",
          "## Objective", "",
          "```", "maximise   beta_actual = d^T G i", "",
          "s.t.  d^T G i >= 0",
          "      ||(I - d d^T) G i||        <= tan(theta_max) * d^T G i",
          "      sqrt(mean_W[(R i)^2])      <= epsilon_nl * (d^T G i) * sigma",
          "      lower <= i <= upper", "```", "",
          "No independent nominal beta, and none may be reintroduced.", "",
          "| item | value |", "|---|---|",
          "| gamma | 42.577e6 Hz/T = 42577 Hz/mT |",
          "| field maps | Hz/A, user-confirmed |",
          "| coordinates | NIfTI RAS+, metres, from the affine only |",
          "| scanner frame | NOT AVAILABLE |",
          "| current limit | ±3 A |",
          "| theta_max | required, UI default 5° |",
          "| epsilon_nonlinear | required in every headline number |",
          "| post-solve clipping | never |",
          "| coverage threshold | %.0f%% |" % (100 * COVERAGE_THRESHOLD),
          "| solver | CLARABEL → SCS, SOCP |",
          "| monotonicity tolerance | 1e-4 relative |", "",
          "## Verified ranges, for the hashed field maps", "",
          "| array | region | status |", "|---|---|---|",
          "| 12ch PFC | real PFC ROI | verified |",
          "| 12ch PFC | DSV 4 cm, energy-centred | verified, 100% |",
          "| 12ch PFC | DSV 8 cm, energy-centred | verified, 98% |",
          "| 36ch | DSV 8 cm, isocentre | verified, 100% |",
          "| 36ch | DSV 12 cm, isocentre | verified, 99.9% |",
          "| 36ch | DSV 20 cm, isocentre | **NOT VERIFIED**, 46%, TRUNCATED |",
          "",
          "No diameter is hard-coded. The UI recomputes coverage from the maps",
          "that are loaded, so a larger calibration enlarges the usable DSV by",
          "itself.", "",
          "## Still unknown", "",
          "R, L, mutual inductance, Vmax, amplifier bandwidth, H(f), scanner",
          "ADC synchronisation, gradient coupling. Voltage, power and dynamic",
          "realisability are reported as unknown, never as zero.", "",
          "Full record: `linear_encoding_freeze.json`.", ""]
    (OUT / "linear_encoding_freeze.md").write_text("\n".join(md),
                                                   encoding="utf-8")

    print("frozen: objective = d^T G i, no nominal beta")
    print("  coverage threshold %.0f%%, theta required, epsilon required"
          % (100 * COVERAGE_THRESHOLD))
    for fp in record["field_map_identity"]:
        if "error" in fp:
            print("  !! %s" % fp["error"])
        else:
            print("  %s: %d channels, set sha256 %s"
                  % (fp["coil"], fp["n_channels"],
                     fp["fieldmap_set_sha256"][:16]))
    missing = [r for r, h in files.items() if h is None]
    if missing:
        print("  !! not hashed: %s" % ", ".join(missing))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
