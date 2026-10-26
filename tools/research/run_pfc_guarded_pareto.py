"""Route 2: what a Hz of rest homogeneity buys in the PFC.

    python tools/research/run_pfc_guarded_pareto.py              # train + validation
    python tools/research/run_pfc_guarded_pareto.py --limit 4    # smoke test
    python tools/research/run_pfc_guarded_pareto.py --folds test # only after freezing

Route 1 established that the shipping solution is already Pareto-optimal in
(PFC RMS, rest RMS): at a 0% rest budget there is nothing to collect, and the
solve returns production back. So this run does not look for a gap. It prices
the trade -- for each subject, the PFC reachable at each rest budget, and the
exchange rate that implies.

**Test fold is not run by default.** The operating point on this front is a
choice made from these numbers, which makes it a hyper-parameter; it is
selected on train and validation, frozen, and only then measured on test.
`--folds test` exists so that final run is one deliberate command.

Writes, under `results/route2_pfc_guarded/`:

    pareto.csv      one row per (subject, budget)
    summary.md      exchange rates and prices, median / IQR / P10 / P90 / worst
    fig_*.png       per-subject fronts, exchange-rate distribution

No serial port, no hardware, no change to the production defaults.
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
from core.hybrid_shim import SH12, build_sh_basis  # noqa: E402
from core.pfc_guarded_joint import DEFAULT_BUDGETS, pareto_front  # noqa: E402
from core.pipeline import _load_coil_maps  # noqa: E402
from core.shim_metrics import metric_row  # noqa: E402
from core.shim_state import SystemShimState  # noqa: E402
from core.subject_split import describe, split_subjects  # noqa: E402
from run_controllability_audit import (FOLD_COLOUR, QUALIFIER,  # noqa: E402
                                       screen_field_map, stat_table, stats)
from common import find_subjects  # noqa: E402

OUT = ROOT / "results" / "route2_pfc_guarded"

#: Budget-0 must reproduce production. Anything above this is a real
#: disagreement between the two solvers, not conic slop.
PARETO_TOL_HZ = 1e-3

CSV_KEYS = ("subject", "fold", "budget",
            "pfc_rms", "rest_rms", "brain_rms", "f0_hz",
            "pfc_p95", "pfc_p99", "pfc_worst10", "mean_diff",
            "pfc_gain_hz", "pfc_gain_pct", "rest_cost_hz", "rest_cost_pct",
            "exchange_rate", "beats_baseline",
            "rest_cap_hz", "rest_floor_hz", "rest_floor_margin_hz",
            "current_max", "at_limit", "sh_at_limit",
            "pfc_voxels", "rest_voxels",
            "baseline_pfc_rms", "production_pfc_rms", "production_rest_rms",
            "solver", "status", "violation")


def fig_fronts(rows, path):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    by_sub = {}
    for r in rows:
        by_sub.setdefault(r["subject"], []).append(r)
    for name, rs in by_sub.items():
        rs = sorted(rs, key=lambda r: r["budget"])
        ax.plot([r["rest_cost_hz"] for r in rs],
                [-r["pfc_gain_hz"] for r in rs], "-o", ms=3, lw=1.0,
                alpha=0.7, color=FOLD_COLOUR[rs[0]["fold"]])
    for f in sorted({r["fold"] for r in rows}):
        ax.plot([], [], color=FOLD_COLOUR[f], label=f)
    ax.set_xlabel("rest RMS given up (Hz)")
    ax.set_ylabel("PFC RMS gained (Hz)")
    ax.set_title("The price of PFC homogeneity, per subject")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_exchange(rows, path, budgets):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pos, labels = [], []
    for i, b in enumerate(budgets):
        if b == 0:
            continue
        v = [r["exchange_rate"] for r in rows
             if abs(r["budget"] - b) < 1e-12 and np.isfinite(r["exchange_rate"])]
        if not v:
            continue
        pos.append(v)
        labels.append("%.2g%%" % (100 * b))
    if not pos:
        return
    ax.boxplot(pos, labels=labels, widths=0.6)
    ax.axhline(1.0, ls="--", c="gray", lw=1)
    ax.set_xlabel("rest budget")
    ax.set_ylabel("Hz of PFC gained per Hz of rest given up")
    ax.set_title("Exchange rate (above 1 = more PFC won than rest lost)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coil", default="PFC 12ch")
    ap.add_argument("--folds", default="train,validation",
                    help="comma-separated; 'test' only after the choice is frozen")
    args = ap.parse_args()

    want_folds = [f.strip() for f in args.folds.split(",") if f.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    coil = get_coil(args.coil)
    imax = coil.current_limit_a

    st = SystemShimState.default()
    bounds = st.delta_bounds(SH12)

    found = find_subjects(args.limit, True)
    keep, excluded = [], []
    for rec in found:
        _, hz, bp, _ = rec
        b0 = np.asanyarray(nib.load(str(hz)).dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
        ok, diag = screen_field_map(b0, brain)
        (keep if ok else excluded).append((rec, diag))
    keep = [r for r, _ in keep]
    fold_of = split_subjects([n for n, *_ in keep])
    subjects = [r for r in keep if fold_of[r[0]] in want_folds]

    print("coil %s (±%.1f A); scanner window from %s"
          % (coil.name, imax, st.bounds_source))
    print("%d scans, %d excluded, folds %s -> %d subjects\n"
          % (len(found), len(excluded), want_folds, len(subjects)))
    print(describe(fold_of), "\n")
    if "test" in want_folds:
        print("!! test fold included -- only valid if the operating point is "
              "already frozen\n")

    rows, failures, pareto_err = [], [], []

    for k, (name, hz, bp, pp) in enumerate(subjects, 1):
        fold = fold_of[name]
        t0 = time.time()
        try:
            img = nib.load(str(hz))
            b0 = np.asanyarray(img.dataobj, dtype=np.float64)
            brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
            pfc = np.asanyarray(nib.load(str(pp)).dataobj) > 0.5
            maps, _ = _load_coil_maps(coil, img.affine, b0.shape[:3],
                                      lambda *a: None)
            basis = build_sh_basis(img.affine, b0.shape[:3])
            sh_maps = [basis[t] for t in SH12]

            prod = solve_adaptive_joint(
                b0, brain, sh_maps, maps, imax=imax, lambda_pfc_low=0.0,
                weight_kwargs={"base": 1.0, "alpha": 0.0, "top_fraction": 0.0},
                sh_bounds=bounds, log_cb=None)
            x_prod = np.r_[prod["scanner_sh_delta"], prod["coil_currents"]]

            corr = sum(x_prod[i] * m for i, m in enumerate(sh_maps))
            corr = corr + np.tensordot(maps, prod["coil_currents"],
                                       axes=([3], [0]))
            pm = pfc & brain
            rm = brain & ~pm
            prow = metric_row("production", b0 + corr, pm, rm, brain)
            brow = metric_row("none", b0, pm, rm, brain)

            front = pareto_front(b0, brain, pfc, sh_maps, maps, bounds,
                                 imax=imax, reference_solution=x_prod,
                                 budgets=DEFAULT_BUDGETS)
        except Exception as e:
            failures.append((name, "%s: %s" % (type(e).__name__, e)))
            print("[%2d/%d] %-12s FAILED: %s" % (k, len(subjects), name, e))
            traceback.print_exc()
            continue

        for r in front:
            if "error" in r:
                failures.append((name, "budget %g: %s"
                                 % (r["budget"], r["error"])))
                continue
            gain = prow["pfc_rms"] - r["pfc_rms"]          # Hz won in the PFC
            cost = r["rest_rms"] - prow["rest_rms"]        # Hz lost elsewhere
            r.update({
                "subject": name, "fold": fold,
                "pfc_voxels": int(pm.sum()), "rest_voxels": int(rm.sum()),
                "pfc_gain_hz": gain,
                "pfc_gain_pct": 100.0 * gain / max(prow["pfc_rms"], 1e-12),
                "rest_cost_hz": cost,
                "rest_cost_pct": 100.0 * cost / max(prow["rest_rms"], 1e-12),
                "exchange_rate": (gain / cost) if cost > 1e-9 else float("nan"),
                "beats_baseline": int(r["pfc_rms"] <= brow["pfc_rms"]),
                "baseline_pfc_rms": brow["pfc_rms"],
                "production_pfc_rms": prow["pfc_rms"],
                "production_rest_rms": prow["rest_rms"],
            })
            rows.append(r)

        zero = [r for r in front if r.get("budget") == 0.0 and "error" not in r]
        if zero:
            d = abs(zero[0]["pfc_rms"] - prow["pfc_rms"])
            if d > PARETO_TOL_HZ:
                pareto_err.append((name, d))

        best = [r for r in front if "error" not in r][-1]
        print("[%2d/%d] %-12s %-10s prod PFC %6.2f  @5%% %6.2f "
              "(%+5.2f Hz for %+4.2f Hz rest, rate %4.2f)  %.1fs"
              % (k, len(subjects), name, fold, prow["pfc_rms"],
                 best["pfc_rms"], best["pfc_gain_hz"], best["rest_cost_hz"],
                 best["exchange_rate"], time.time() - t0))

    with open(OUT / "pareto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_KEYS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if rows:
        fig_fronts(rows, OUT / "fig_pareto_fronts.png")
        fig_exchange(rows, OUT / "fig_exchange_rate.png", DEFAULT_BUDGETS)

    # ── summary ─────────────────────────────────────────────────────────
    md = ["# Route 2 — the PFC-rest trade-off", "",
          "> " + QUALIFIER, "",
          "> A rest-RMS budget bounds one scalar summary of one region. It "
          "does not mean the rest of the brain is unaffected in EPI: nothing "
          "here measures displacement, dropout or image quality.", "",
          "Coil `%s`, ±%.1f A. Scanner window: intersection of the ±2000 delta "
          "and absolute envelopes, from `%s`. Folds run: %s. %d subjects."
          % (coil.name, imax, st.bounds_source, ", ".join(want_folds),
             len({r["subject"] for r in rows})), ""]
    if "test" not in want_folds:
        md += ["> Test fold deliberately not run. The operating point on this "
               "front is a choice made from these numbers, so it is a "
               "hyper-parameter: chosen here, frozen, then measured once on "
               "test with `--folds test`.", ""]
    if failures:
        md += ["## Failures", ""] + ["- `%s`: %s" % f for f in failures] + [""]

    md += ["## Self-check: budget 0 must return production", "",
           "Production minimises whole-brain mean square, i.e. the "
           "voxel-count-weighted sum of the two regional objectives, so it is "
           "Pareto-optimal and `min PFC s.t. rest ≤ rest_production` cannot "
           "improve on it. If this disagrees, one of the two solvers is not "
           "at its optimum.", ""]
    md += (["Largest disagreement: **%.2e Hz** (tolerance %.0e). %s"
            % (max(d for _, d in pareto_err), PARETO_TOL_HZ,
               "; ".join("`%s` %.2e" % (n, d) for n, d in pareto_err))]
           if pareto_err else
           ["All subjects agree to within %.0e Hz. Confirmed." % PARETO_TOL_HZ])
    md += [""]

    for b in DEFAULT_BUDGETS:
        sel = [r for r in rows if abs(r["budget"] - b) < 1e-12]
        if not sel:
            continue
        if b == 0:
            md += ["## Budget 0%", "",
                   "PFC gain is zero by construction — see the self-check "
                   "above. Listed for completeness.", ""]
            continue
        n_beat = sum(r["beats_baseline"] for r in sel)
        n_below = sum(1 for r in sel
                      if r["production_pfc_rms"] > r["baseline_pfc_rms"])
        md += ["## Budget %.2g%% — rest allowed to degrade by %.2g%%"
               % (100 * b, 100 * b), "",
               "PFC RMS gained (Hz):", "",
               stat_table(sel, "pfc_gain_hz", "min"), "",
               "Rest RMS given up (Hz):", "",
               stat_table(sel, "rest_cost_hz", "max"), "",
               "Exchange rate — Hz of PFC per Hz of rest:", "",
               stat_table(sel, "exchange_rate", "min"), "",
               "PFC RMS reached (Hz):", "",
               stat_table(sel, "pfc_rms", "max"), "",
               "At this budget %d of %d subjects reach a PFC RMS at or below "
               "their unshimmed baseline (production manages %d)."
               % (n_beat, len(sel), len(sel) - n_below), ""]

    vr = np.median([r["rest_voxels"] / max(r["pfc_voxels"], 1)
                    for r in rows if r.get("pfc_voxels")]) if any(
                        r.get("pfc_voxels") for r in rows) else float("nan")
    md += ["## Reading the exchange rate", "",
           ("The rate is a ratio of two RMS values over regions of very "
            "different size — the rest holds about %.0f times as many voxels "
            "as the PFC on this cohort." % float(vr)) +
           " A rate of 5 does **not** mean five times "
           "more field is fixed than broken. Weighted by voxel count the same "
           "trade runs the other way, and it must: production minimises the "
           "voxel-weighted sum exactly, so every move along this front "
           "increases it. What the front says is that PFC RMS is *cheap* in "
           "rest-RMS terms, because spreading a small degradation over a large "
           "region barely moves that region's RMS.", "",
           "Every row on this front is a PFC–rest trade-off with a stated "
           "price, and none of it is obtained for nothing. What the front "
           "supports is a decision that has to be made on scientific grounds: "
           "for a PFC study, is a PFC voxel worth more than a voxel in the "
           "rest of the head? Production currently answers no, by default, "
           "purely through voxel counts — and never says so.", "",
           "## Tail metrics at each budget", "",
           "The PFC's worst decile is what a task-driven objective would "
           "target; RMS can improve while the tail does not.", ""]
    for b in DEFAULT_BUDGETS:
        sel = [r for r in rows if abs(r["budget"] - b) < 1e-12]
        if not sel or b == 0:
            continue
        md += ["Budget %.2g%% — mean of the worst 10%% of |PFC| (Hz):"
               % (100 * b), "", stat_table(sel, "pfc_worst10", "max"), ""]

    worst_v = max((r["violation"] for r in rows), default=0.0)
    md += ["## Constraints", "",
           "Largest violation of the rest cap or the scanner/current box over "
           "every solve: **%.2e Hz**. Nothing is clipped after solving; a "
           "solution that violated its constraints would raise." % worst_v, "",
           "![](fig_pareto_fronts.png)", "", "![](fig_exchange_rate.png)", ""]

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote %s" % OUT)
    if failures:
        print("%d failures" % len(failures))
    if pareto_err:
        print("!! budget-0 disagreement on %d subjects" % len(pareto_err))


if __name__ == "__main__":
    main()
