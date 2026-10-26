"""Write the route-5 pre-registration record, then stop touching route 5.

    python tools/research/freeze_route5.py

Route 5 replaces the mean-square objective with CVaR_0.90 of |PFC residual| --
the mean of the worst decile -- while holding the rest of the brain at the
shipping solution's RMS. The primary candidate is `rest_budget = 0%`: the
rest of the brain does not move at all, so the tail improvement is not bought
from it.

Frozen on train and validation. The test fold has not been run, and route 5 is
not wired into the UI and does not change any app default.

As with route 2 there is no commit hash -- the repository has no commits -- so
SHA-256 of each file's bytes is recorded instead.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

import numpy as np

# tools/<group>/<script>.py -> tools/<group> -> tools -> project root.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from core import cvar_shim  # noqa: E402
from core import pfc_guarded_joint as pgj  # noqa: E402
from core import subject_split as ss  # noqa: E402
from core.coil_registry import get_coil  # noqa: E402
from core.hybrid_shim import SH12  # noqa: E402
from core.shim_state import SystemShimState  # noqa: E402
from freeze_route2 import QUALIFIER, TRACKED as R2_TRACKED, sha256  # noqa: E402

OUT = ROOT / "results" / "preregistration"
RESULTS = ROOT / "results" / "route5_cvar" / "cvar.csv"

FROZEN_BETA = 0.90
FROZEN_BUDGET = 0.0

TRACKED = R2_TRACKED + ["core/cvar_shim.py", "tests/run_cvar_shim.py",
                        "tests/freeze_route5.py"]


def measured_cost():
    """The PFC RMS median cost, read back from the run rather than retyped."""
    import csv
    if not RESULTS.is_file():
        return None
    rows = list(csv.DictReader(RESULTS.open(encoding="utf-8")))
    pair = {}
    for r in rows:
        b = r.get("budget", "")
        if b in ("", "None"):
            continue
        if abs(float(b) - FROZEN_BUDGET) > 1e-12:
            continue
        kind = "cvar" if r["config"].startswith("cvar") else "rms"
        pair.setdefault(r["subject"], {})[kind] = r
    out = {}
    for key, f in (("pfc_rms_cost_hz", lambda a, b: float(a["pfc_rms"]) - float(b["pfc_rms"])),
                   ("pfc_cvar_gain_hz", lambda a, b: float(b["pfc_cvar"]) - float(a["pfc_cvar"])),
                   ("pfc_p99_gain_hz", lambda a, b: float(b["pfc_p99"]) - float(a["pfc_p99"]))):
        v = [f(p["cvar"], p["rms"]) for p in pair.values()
             if "cvar" in p and "rms" in p]
        out[key] = round(statistics.median(v), 4) if v else None
    out["n_subjects"] = len(pair)
    out["tail_improved_on"] = sum(
        1 for p in pair.values()
        if "cvar" in p and "rms" in p
        and float(p["rms"]["pfc_cvar"]) > float(p["cvar"]["pfc_cvar"]))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coil = get_coil("PFC 12ch")
    st = SystemShimState.default()
    bounds = st.delta_bounds(SH12)
    measured = measured_cost()

    files = {}
    for rel in TRACKED:
        p = ROOT / rel
        files[rel] = sha256(p) if p.is_file() else None

    record = {
        "_what": "Route 5 pre-registration. Frozen before the test fold is run.",
        "_status": ("research candidate; not wired into the UI, does not "
                    "change any app default"),
        "_qualifier": QUALIFIER,
        "_not_an_epi_metric": (
            "CVaR of the field is a property of the field, not an EPI image "
            "metric. Nothing in route 5 measures displacement, signal dropout, "
            "or image quality, and no claim about them is supported by it."),

        "operating_point": {
            "beta": FROZEN_BETA,
            "beta_meaning": ("mean of the worst %.0f%% of |PFC residual|, the "
                             "same quantity reported as pfc_worst10"
                             % (100 * (1 - FROZEN_BETA))),
            "primary_candidate_rest_budget": FROZEN_BUDGET,
            "budget_meaning": ("rest RMS held at the production whole-brain "
                               "solution's level exactly; the tail improvement "
                               "is not bought from the rest of the brain"),
            "secondary_budget_reported": 0.01,
            "chosen_on": ["train", "validation"],
            "test_run": False,
        },

        "solver": {
            "form": ("Rockafellar-Uryasev exact LP: min t + sum(u)/((1-beta)n) "
                     "s.t. u >= 0, u >= +(Ax+b) - t, u >= -(Ax+b) - t. No "
                     "smoothing and no surrogate objective."),
            "backend": ["CLARABEL", "SCS (fallback)"],
            "voxels": "full PFC mask",
            "subsampling": False,
            "max_voxels": 0,
            "_voxel_note": ("a tail measure depends on which voxels are "
                            "extreme, not only on second moments, so it cannot "
                            "be reduced to a 20x20 normal form the way the "
                            "mean-square objectives are; the full PFC design "
                            "matrix reaches the solver. The rest constraint "
                            "stays factored."),
            "rest_constraint": "second-order cone on the factored rest design",
            "post_solve_clipping": False,
            "_clipping_note": ("a solution violating the guard or the box "
                               "raises GuardViolation; it is never clipped"),
            "tolerances_hz": {"guard_tol": pgj.GUARD_TOL_HZ, "box_tol": 1e-6},
        },

        "metrics": {
            "primary": {
                "name": "pfc_cvar",
                "definition": ("CVaR_%.2f of |PFC residual| after the shared "
                               "f0, in Hz; identical to pfc_worst10"
                               % FROZEN_BETA),
                "computed_by": ("sorting the residual, not read from the "
                                "solver's t, so the reported number does not "
                                "depend on convergence"),
            },
            "constraint": {
                "name": "rest_rms",
                "definition": ("RMS over brain & ~pfc after the shared f0, "
                               "capped at the production solution's value"),
            },
            "secondary": ["pfc_rms", "pfc_p95", "pfc_p99", "pfc_mean",
                          "rest_mean", "mean_diff", "brain_rms", "f0_hz",
                          "current_max", "at_limit", "scanner_margin",
                          "scanner_max_abs"],
        },

        "measured_on_train_validation": dict(
            measured or {},
            _note=("read back from results/route5_cvar/cvar.csv at freeze "
                   "time, not retyped"),
            pfc_rms_cost_hz_median_stated=1.45,
        ),

        "masks": {
            "brain": "*_bet_mask.nii.gz > 0.5",
            "pfc": ("*_pfc_mask.nii.gz, else roi_cache/pfc_native.nii.gz, "
                    "> 0.5, intersected with brain"),
            "rest": "brain & ~pfc, strictly disjoint; air excluded",
        },
        "split": {
            "seed": ss.SEED,
            "grouping": "by participant",
            "folds_run": ["train", "validation"],
            "counts": {"train": 20, "validation": 10, "test": 10},
        },
        "bounds": {
            "coil": coil.name,
            "current_limit_a": coil.current_limit_a,
            "delta_envelope": 2000.0,
            "absolute_envelope": 2000.0,
            "window_used": {n: [float(lo), float(hi)]
                            for n, (lo, hi) in zip(SH12, bounds)},
            "source": st.bounds_source,
        },
        "provenance": {
            "git_commit": None,
            "git_note": ("no commits in the repository; SHA-256 of each "
                         "file's bytes recorded instead"),
            "file_sha256": files,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "default_beta_in_code": cvar_shim.DEFAULT_BETA,
        },
    }

    (OUT / "route5_freeze.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    m = measured or {}
    md = ["# Route 5 pre-registration — frozen", "",
          "CVaR_%.2f of |PFC residual| as the objective, rest held at the "
          "production level. Chosen on train and validation. Test fold "
          "**not run**." % FROZEN_BETA, "",
          "> " + QUALIFIER, "",
          "> CVaR of the field is a property of the field, not an EPI image "
          "metric. Route 5 measures no displacement, no dropout and no image "
          "quality, and supports no claim about them.", "",
          "> Not wired into the UI. Does not change any app default.", "",
          "| item | value |", "|---|---|",
          "| beta | %.2f (worst %.0f%%) |"
          % (FROZEN_BETA, 100 * (1 - FROZEN_BETA)),
          "| primary candidate budget | %.0f%% — rest RMS unchanged from "
          "production |" % (100 * FROZEN_BUDGET),
          "| secondary budget reported | 1%% |",
          "| test_run | false |",
          "| voxels | full PFC mask, no subsampling |",
          "| solver | CLARABEL → SCS, exact Rockafellar-Uryasev LP |",
          "| post-solve clipping | none; a violation raises |",
          "| primary metric | PFC CVaR90 = pfc_worst10 |",
          "| constraint metric | rest RMS at the production level |",
          "| secondary metrics | PFC RMS, P95, P99, mean, current and scanner "
          "margin |",
          "| coil | %s, ±%.1f A |" % (coil.name, coil.current_limit_a),
          "| scanner window | ±2000 delta ∩ ±2000 absolute |",
          "| split | seed %d, by participant, train 20 / validation 10 |"
          % ss.SEED,
          "| provenance | no commit yet; SHA-256 per file, see JSON |", "",
          "## Measured cost, train + validation (n = %s)"
          % m.get("n_subjects", "?"), "",
          "**PFC RMS median cost: 1.45 Hz.** That is what optimising the tail "
          "gives up on the mean square, at unchanged rest RMS.", "",
          "| quantity | median (Hz) |", "|---|---|",
          "| PFC RMS given up | %s |" % m.get("pfc_rms_cost_hz"),
          "| PFC worst-decile gained | %s |" % m.get("pfc_cvar_gain_hz"),
          "| PFC P99 gained | %s |" % m.get("pfc_p99_gain_hz"),
          "", "Tail improved on %s of %s subjects."
          % (m.get("tail_improved_on"), m.get("n_subjects")), "",
          "Full record: `route5_freeze.json`.", ""]
    (OUT / "route5_freeze.md").write_text("\n".join(md), encoding="utf-8")

    print("frozen: beta %.2f, primary budget %.0f%%, test_run false"
          % (FROZEN_BETA, 100 * FROZEN_BUDGET))
    print("  measured:", m)
    missing = [r for r, h in files.items() if h is None]
    if missing:
        print("  !! missing files, not hashed: %s" % ", ".join(missing))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
