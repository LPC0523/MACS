"""Write the route-2 pre-registration record, then stop touching route 2.

    python tools/research/freeze_route2.py

The candidate operating point is `rest_budget = 1%`. Freezing it means the
number was chosen from train and validation only, and is recorded here before
the test fold is ever run. Everything the choice depends on is captured in one
file so that a later test run can be checked against it rather than trusted:
the budget, the solver settings and tolerances, the mask definitions, the exact
participant split, the scanner and current bounds, content hashes of every
module involved, and the metric definitions.

This is a research candidate. It is **not** an app default, and a 1% rest-RMS
budget does not mean the rest of the brain is unaffected in EPI.

There is no commit hash: the repository has no commits yet, so `git rev-parse`
has nothing to return. SHA-256 of each file's bytes is recorded instead, which
pins the same thing and does not depend on the working tree being committed.
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

import nibabel as nib  # noqa: E402

from core import pfc_guarded_joint as pgj  # noqa: E402
from core.coil_registry import get_coil  # noqa: E402
from core.hybrid_shim import SH12  # noqa: E402
from core.shim_metrics import METRIC_KEYS  # noqa: E402
from core.shim_state import SystemShimState  # noqa: E402
from core import subject_split as ss  # noqa: E402
from run_controllability_audit import MIN_PLAUSIBLE_STD_HZ, screen_field_map  # noqa: E402
from common import find_subjects  # noqa: E402

OUT = ROOT / "results" / "preregistration"

#: The frozen candidate. Chosen on train + validation; test not run.
FROZEN_BUDGET = 0.01

TRACKED = [
    "core/pfc_guarded_joint.py",
    "core/controllability_audit.py",
    "core/subject_split.py",
    "core/shim_metrics.py",
    "core/shim_state.py",
    "core/adaptive_joint.py",
    "core/hybrid_shim.py",
    "config/shim_delta_bounds.json",
    "tests/run_pfc_guarded_pareto.py",
    "tests/run_controllability_audit.py",
    "tests/freeze_route2.py",
]

QUALIFIER = (
    "Pareto optimality applies only to regional MSE/RMS under the identical "
    "feasible set; it does not imply Pareto optimality for CVaR, EPI "
    "displacement, signal dropout, or measured EPI image quality.")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    coil = get_coil("PFC 12ch")
    st = SystemShimState.default()
    bounds = st.delta_bounds(SH12)

    found = find_subjects(0, True)
    keep, excluded = [], []
    for name, hz, bp, pp in found:
        b0 = np.asanyarray(nib.load(str(hz)).dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
        ok, diag = screen_field_map(b0, brain)
        (keep if ok else excluded).append((name, diag))
    fold_of = ss.split_subjects([n for n, _ in keep])

    files = {}
    for rel in TRACKED:
        p = ROOT / rel
        files[rel] = sha256(p) if p.is_file() else None

    record = {
        "_what": "Route 2 pre-registration. Frozen before the test fold is run.",
        "_status": "research candidate, not an app default",
        "_qualifier": QUALIFIER,
        "_rest_budget_meaning": (
            "A 1% rest-RMS budget bounds one scalar summary of one region. It "
            "does not mean the rest of the brain is undamaged in EPI: nothing "
            "here measures displacement, dropout or image quality."),

        # 1 ── the frozen operating point
        "operating_point": {
            "rest_budget": FROZEN_BUDGET,
            "definition": (
                "minimise PFC RMS subject to rest RMS <= (1 + budget) * "
                "rest RMS of the production whole-brain solution, under the "
                "identical feasible set"),
            "reference_solution": (
                "core.adaptive_joint.solve_adaptive_joint with uniform weights "
                "and lambda_pfc_low=0, i.e. the shipping whole-brain solve"),
            "chosen_on": ["train", "validation"],
            "test_run": False,
            "pareto_points_reported": list(pgj.DEFAULT_BUDGETS),
        },

        # 2 ── solver, weights, tolerances
        "solver": {
            "candidate": {
                "form": ("epsilon-constraint SOCP on the factored normal form; "
                         "min ||R_p x + u_p||^2 s.t. ||R_r x + u_r|| <= cap, "
                         "box on x"),
                "backend": ["CLARABEL", "SCS (fallback)"],
                "reduction": ("design matrices reduced to their 20x20 Cholesky "
                              "factor before reaching CVXPY; no million-row "
                              "matrix is handed to the solver"),
                "cholesky_jitter_start": 1e-14,
            },
            "reference": {
                "backend": "scipy L-BFGS-B",
                "options": {"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12,
                            "maxls": 60},
                "weight_kwargs": {"base": 1.0, "alpha": 0.0,
                                  "top_fraction": 0.0},
                "lambda_pfc_low": 0.0,
                "_note": ("alpha = 0 and top_fraction = 0 means uniform voxel "
                          "weights: the reference is the plain whole-brain "
                          "mean-square solve, with no adaptive weighting"),
            },
            "tolerances_hz": {
                "guard_tol": pgj.GUARD_TOL_HZ,
                "pareto_selfcheck_tol": 1e-3,
                "box_tol": 1e-6,
            },
            "post_solve_clipping": False,
            "_clipping_note": ("a solution violating its constraints raises "
                               "GuardViolation; it is never clipped"),
        },

        # 3 ── masks and split
        "masks": {
            "brain": "*_bet_mask.nii.gz > 0.5",
            "pfc": ("*_pfc_mask.nii.gz, else roi_cache/pfc_native.nii.gz, "
                    "> 0.5, intersected with brain"),
            "rest": "brain & ~pfc, strictly disjoint from pfc; air excluded",
            "field": "*_deltaB0_hertz.nii.gz, Hz",
        },
        "exclusions": {
            "rule": "in-brain std < %.1f Hz" % MIN_PLAUSIBLE_STD_HZ,
            "reason": ("radian-scale phase written into a file named hertz; "
                       "values reach 0.997 x 3*pi and are identical between "
                       "two different participants"),
            "scans": [{"scan": n, "brain_std_hz": round(d["brain_std_hz"], 3),
                       "max_abs_hz": round(d["max_abs_hz"], 3)}
                      for n, d in excluded],
        },
        "split": {
            "seed": ss.SEED,
            "nominal_target_scans": list(ss.TARGET),
            "grouping": ("by participant, not by scan: repeat sessions and the "
                         "with/without-earshim pairs share anatomy"),
            "scans_found": len(found),
            "scans_audited": len(keep),
            "counts": {f: sum(1 for v in fold_of.values() if v == f)
                       for f in ss.FOLDS},
            "assignment": dict(sorted(fold_of.items())),
        },

        # 4 ── bounds
        "bounds": {
            "coil": coil.name,
            "current_limit_a": coil.current_limit_a,
            "scanner_terms": list(SH12),
            "delta_envelope": 2000.0,
            "absolute_envelope": 2000.0,
            "rule": ("per-term intersection: lo = max(-delta, -abs - baseline), "
                     "hi = min(+delta, +abs - baseline); asymmetric whenever "
                     "the baseline is non-zero"),
            "baseline_used": {n: float(st.baseline.get(n, 0.0)) for n in SH12},
            "window_used": {n: [float(lo), float(hi)]
                            for n, (lo, hi) in zip(SH12, bounds)},
            "source": st.bounds_source,
            "_policy": ("a deliberately conservative deployment envelope inside "
                        "the scanner's real capability; expected to bind, and "
                        "binding is the policy being enforced"),
        },

        # 5 ── provenance
        "provenance": {
            "git_commit": None,
            "git_note": ("the repository has no commits yet, so there is no "
                         "commit to record; SHA-256 of each file's bytes is "
                         "given instead"),
            "file_sha256": files,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
        },

        # 6 ── metrics
        "metrics": {
            "shared_f0": ("one centre frequency for the whole head, equal to "
                          "the negated in-brain mean of the residual, built "
                          "into the design by centring every column on its "
                          "in-brain mean; PFC and rest are never demeaned "
                          "separately"),
            "primary": {
                "name": "pfc_rms",
                "definition": ("root mean square of the residual over the PFC "
                               "mask after the shared f0, in Hz"),
            },
            "constraint": {
                "name": "rest_rms",
                "definition": ("same quantity over brain & ~pfc; capped at "
                               "(1 + budget) x the reference solution's value"),
            },
            "secondary": ["brain_rms", "pfc_p95", "pfc_p99", "pfc_worst10",
                          "pfc_mean", "rest_mean", "mean_diff", "f0_hz"],
            "reported_but_not_optimised": ["pfc_std", "rest_std"],
            "_std_note": ("region-centred standard deviations demean each "
                          "region separately and hide a genuine frequency "
                          "offset between them; kept only for comparison with "
                          "figures produced before the shared f0 was settled"),
            "all_metric_keys": list(METRIC_KEYS),
        },
    }

    (OUT / "route2_freeze.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Route 2 pre-registration — frozen", "",
          "Candidate operating point: **rest_budget = %.0f%%**, chosen on train "
          "and validation. Test fold **not run**." % (100 * FROZEN_BUDGET), "",
          "> %s" % QUALIFIER, "",
          "> This is a research candidate, not an app default, and a 1% "
          "rest-RMS budget does not mean the rest of the brain is unaffected "
          "in EPI.", "",
          "| item | value |", "|---|---|",
          "| budget | %.4g |" % FROZEN_BUDGET,
          "| reference | whole-brain L-BFGS-B solve, uniform weights, "
          "lambda_pfc_low=0 |",
          "| candidate solver | CLARABEL, SCS fallback, factored SOCP |",
          "| guard tolerance | %.0e Hz, no post-solve clipping |"
          % pgj.GUARD_TOL_HZ,
          "| coil | %s, ±%.1f A |" % (coil.name, coil.current_limit_a),
          "| scanner window | ±2000 delta ∩ ±2000 absolute, per term |",
          "| split seed | %d, grouped by participant |" % ss.SEED,
          "| scans | %d found, %d excluded, %d audited |"
          % (len(found), len(excluded), len(keep)),
          "| folds | %s |" % ", ".join(
              "%s %d" % (f, sum(1 for v in fold_of.values() if v == f))
              for f in ss.FOLDS),
          "| primary metric | PFC RMS under the shared f0 |",
          "| constraint metric | rest RMS, capped at (1+budget)×reference |",
          "| provenance | no commit yet; SHA-256 per file, see JSON |", ""]
    md += ["## File hashes", "", "| file | sha256 |", "|---|---|"]
    for rel, h in files.items():
        md.append("| `%s` | `%s` |" % (rel, (h or "MISSING")[:16]))
    md += ["", "Full record: `route2_freeze.json`.", ""]
    (OUT / "route2_freeze.md").write_text("\n".join(md), encoding="utf-8")

    print("frozen: budget %.4g" % FROZEN_BUDGET)
    print("  %d scans, %d excluded, folds %s"
          % (len(found), len(excluded),
             {f: sum(1 for v in fold_of.values() if v == f) for f in ss.FOLDS}))
    missing = [r for r, h in files.items() if h is None]
    if missing:
        print("  !! missing files, not hashed: %s" % ", ".join(missing))
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
