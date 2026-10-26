"""Route 5: does optimising the PFC tail beat optimising its mean square?

    python tools/research/run_cvar_shim.py            # train + validation
    python tools/research/run_cvar_shim.py --limit 4  # smoke test

Route 2 showed the shipping solution is Pareto-optimal in (PFC RMS, rest RMS).
That claim is confined to regional MSE/RMS. This run tests the first thing it
excludes: at the *same* rest budget, is the RMS-optimal solution also optimal
for CVaR_0.9 of |PFC residual| -- the mean of the worst decile?

Budget 0 is the sharpest form of the question: the rest of the brain is held at
exactly the shipping solution's level, so any tail improvement is obtained
without moving rest RMS at all. Budget 1% is the frozen route-2 candidate.

Test fold not run. Writes `results/route5_cvar/`.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# tools/<group>/<script>.py -> tools/<group> -> tools -> project root.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import nibabel as nib  # noqa: E402

from core.adaptive_joint import solve_adaptive_joint  # noqa: E402
from core.coil_registry import get_coil  # noqa: E402
from core.cvar_shim import DEFAULT_BETA, cvar_of, solve_cvar_guarded  # noqa: E402
from core.hybrid_shim import SH12, build_sh_basis  # noqa: E402
from core.pfc_guarded_joint import _build, _residual, solve_pfc_guarded  # noqa: E402
from core.pipeline import _load_coil_maps  # noqa: E402
from core.shim_metrics import metric_row  # noqa: E402
from core.shim_state import SystemShimState  # noqa: E402
from core.subject_split import describe, split_subjects  # noqa: E402
from run_controllability_audit import (FOLD_COLOUR, QUALIFIER,  # noqa: E402
                                       screen_field_map, stat_table, stats)
from common import find_subjects  # noqa: E402

OUT = ROOT / "results" / "route5_cvar"
BUDGETS = (0.0, 0.01)

KEYS = ("subject", "fold", "config", "budget", "beta",
        "pfc_cvar", "pfc_rms", "pfc_p95", "pfc_p99", "pfc_worst10",
        "rest_rms", "brain_rms", "f0_hz", "current_max",
        "rest_cap_hz", "pfc_voxels_total", "pfc_voxels_used",
        "solver", "status", "violation", "seconds")


def fig_tail(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for j, b in enumerate(BUDGETS):
        pairs = {}
        for r in rows:
            if r.get("budget") != b:
                continue
            pairs.setdefault(r["subject"], {})[r["kind"]] = r
        xs, ys, cols = [], [], []
        for name, kk in pairs.items():
            if "rms" in kk and "cvar" in kk:
                xs.append(kk["cvar"]["pfc_rms"] - kk["rms"]["pfc_rms"])
                ys.append(kk["rms"]["pfc_cvar"] - kk["cvar"]["pfc_cvar"])
                cols.append(FOLD_COLOUR[kk["rms"]["fold"]])
        ax.scatter(xs, ys, s=26, alpha=0.8, c=cols,
                   marker="o" if b == 0 else "^",
                   label="budget %.2g%%" % (100 * b))
    ax.axhline(0, c="gray", lw=1)
    ax.axvline(0, c="gray", lw=1)
    ax.set_xlabel("PFC RMS given up by optimising the tail (Hz)")
    ax.set_ylabel("PFC worst-decile gained (Hz)")
    ax.set_title("What a tail objective buys, at the same rest budget")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coil", default="PFC 12ch")
    ap.add_argument("--folds", default="train,validation")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA)
    args = ap.parse_args()

    want = [f.strip() for f in args.folds.split(",") if f.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    coil = get_coil(args.coil)
    imax = coil.current_limit_a
    st = SystemShimState.default()
    bounds = st.delta_bounds(SH12)

    found = find_subjects(args.limit, True)
    keep = []
    for rec in found:
        _, hz, bp, _ = rec
        b0 = np.asanyarray(nib.load(str(hz)).dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
        if screen_field_map(b0, brain)[0]:
            keep.append(rec)
    fold_of = split_subjects([n for n, *_ in keep])
    subjects = [r for r in keep if fold_of[r[0]] in want]

    print("coil %s (±%.1f A), beta %.2f, folds %s -> %d subjects\n"
          % (coil.name, imax, args.beta, want, len(subjects)))
    print(describe(fold_of), "\n")

    rows, failures = [], []
    for k, (name, hz, bp, pp) in enumerate(subjects, 1):
        fold = fold_of[name]
        try:
            img = nib.load(str(hz))
            b0 = np.asanyarray(img.dataobj, dtype=np.float64)
            brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
            pfc = np.asanyarray(nib.load(str(pp)).dataobj) > 0.5
            maps, _ = _load_coil_maps(coil, img.affine, b0.shape[:3],
                                      lambda *a: None)
            basis = build_sh_basis(img.affine, b0.shape[:3])
            sh_maps = [basis[t] for t in SH12]
            d = _build(b0, brain, pfc, sh_maps, maps)
            pm, rm = d["pfc"], d["rest"]

            prod = solve_adaptive_joint(
                b0, brain, sh_maps, maps, imax=imax, lambda_pfc_low=0.0,
                weight_kwargs={"base": 1.0, "alpha": 0.0, "top_fraction": 0.0},
                sh_bounds=bounds, log_cb=None)
            xp = np.r_[prod["scanner_sh_delta"], prod["coil_currents"]]
            pr = metric_row("production", _residual(b0, d, xp), pm, rm, brain)
            pr["pfc_cvar"] = cvar_of(
                _residual(b0, d, xp)[pm] + pr["f0_hz"], args.beta)
            pr.update({"subject": name, "fold": fold, "kind": "production",
                       "budget": None, "beta": args.beta, "violation": 0.0,
                       "current_max": float(np.abs(prod["coil_currents"]).max()),
                       "solver": "scipy", "status": "L-BFGS-B", "seconds": 0.0})
            rows.append(pr)

            line = "[%2d/%d] %-12s %-10s prod tail %7.2f" % (
                k, len(subjects), name, fold, pr["pfc_cvar"])
            for b in BUDGETS:
                t0 = time.time()
                rr = solve_pfc_guarded(b0, brain, pfc, sh_maps, maps, bounds,
                                       imax=imax, reference_solution=xp,
                                       budget=b)
                xr = np.r_[rr["scanner_sh_delta"], rr["coil_currents"]]
                rr["pfc_cvar"] = cvar_of(
                    _residual(b0, d, xr)[pm] + rr["f0_hz"], args.beta)
                rr.update({"subject": name, "fold": fold, "kind": "rms",
                           "beta": args.beta, "seconds": time.time() - t0})
                rows.append(rr)

                t0 = time.time()
                cv = solve_cvar_guarded(b0, brain, pfc, sh_maps, maps, bounds,
                                        imax=imax, reference_solution=xp,
                                        budget=b, beta=args.beta)
                cv.update({"subject": name, "fold": fold, "kind": "cvar",
                           "seconds": time.time() - t0})
                rows.append(cv)
                line += "  | %.2g%%: tail %+6.2f, RMS %+5.2f" % (
                    100 * b, rr["pfc_cvar"] - cv["pfc_cvar"],
                    cv["pfc_rms"] - rr["pfc_rms"])
            print(line)
        except Exception as e:
            failures.append((name, "%s: %s" % (type(e).__name__, e)))
            print("[%2d/%d] %-12s FAILED: %s" % (k, len(subjects), name, e))
            traceback.print_exc()

    with open(OUT / "cvar.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(KEYS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    if rows:
        fig_tail(rows, OUT / "fig_tail_vs_rms.png")

    # paired comparison, per budget
    md = ["# Route 5 — tail (CVaR) against mean square", "",
          "> " + QUALIFIER, "",
          "> CVaR of the field is still not a measurement of an EPI image. "
          "This addresses the first omission in the qualifier above and none "
          "of the others.", "",
          "Coil `%s`, ±%.1f A, β = %.2f (mean of the worst %.0f%% of |PFC "
          "residual|). Folds: %s. Test fold not run."
          % (coil.name, imax, args.beta, 100 * (1 - args.beta),
             ", ".join(want)), ""]
    if failures:
        md += ["## Failures", ""] + ["- `%s`: %s" % f for f in failures] + [""]

    for b in BUDGETS:
        paired = {}
        for r in rows:
            if r.get("budget") == b and r.get("kind") in ("rms", "cvar"):
                paired.setdefault(r["subject"], {})[r["kind"]] = r
        comp = []
        for name, kk in paired.items():
            if "rms" not in kk or "cvar" not in kk:
                continue
            comp.append({
                "subject": name, "fold": kk["rms"]["fold"],
                "tail_gain_hz": kk["rms"]["pfc_cvar"] - kk["cvar"]["pfc_cvar"],
                "tail_gain_pct": 100.0 * (kk["rms"]["pfc_cvar"]
                                          - kk["cvar"]["pfc_cvar"])
                                 / max(kk["rms"]["pfc_cvar"], 1e-12),
                "rms_cost_hz": kk["cvar"]["pfc_rms"] - kk["rms"]["pfc_rms"],
                "p99_gain_hz": kk["rms"]["pfc_p99"] - kk["cvar"]["pfc_p99"],
                "rest_diff_hz": kk["cvar"]["rest_rms"] - kk["rms"]["rest_rms"],
            })
        if not comp:
            continue
        n_pos = sum(1 for c in comp if c["tail_gain_hz"] > 0)
        md += ["## Rest budget %.2g%%" % (100 * b), ""]
        if b == 0:
            md += ["The rest of the brain is held at exactly the shipping "
                   "solution's RMS, so the RMS-optimal solution here *is* "
                   "production. Any tail improvement below is obtained at "
                   "unchanged rest RMS.", ""]
        md += ["PFC worst-decile gained by optimising the tail instead of the "
               "mean square (Hz):", "",
               stat_table(comp, "tail_gain_hz", "min"), "",
               "As a percentage of the RMS solution's worst decile:", "",
               stat_table(comp, "tail_gain_pct", "min"), "",
               "PFC RMS given up to get it (Hz):", "",
               stat_table(comp, "rms_cost_hz", "max"), "",
               "P99 of |PFC residual| gained (Hz):", "",
               stat_table(comp, "p99_gain_hz", "min"), "",
               "The tail improves on %d of %d subjects. Rest RMS differs "
               "between the two by at most %.2e Hz — both sit on the same cap."
               % (n_pos, len(comp),
                  max(abs(c["rest_diff_hz"]) for c in comp)), ""]

    worst_v = max((r.get("violation", 0.0) for r in rows), default=0.0)
    md += ["## Constraints", "",
           "Largest violation over every solve: **%.2e Hz**. No post-solve "
           "clipping." % worst_v, "",
           "![](fig_tail_vs_rms.png)", ""]
    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
