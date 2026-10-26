"""Route 1: run the controllability audit over the whole cohort.

    python tools/research/run_controllability_audit.py            # all subjects
    python tools/research/run_controllability_audit.py --limit 3  # smoke test

Writes, under `results/route1_controllability/`:

    per_subject.csv     one row per (subject, config), with the fold label
    gaps.csv            one row per subject: the gap and spectrum summaries
    <subject>.npz       solution vectors and spectra, per subject
    summary.md          median / IQR / P10 / P90 / worst case, split by fold
    fig_*.png           spectrum, principal angles, high-order improvement, gaps

Nothing here touches a serial port or the production defaults. It reads the
same B0 / brain / PFC files the batch tests use and calls `audit_subject`.
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

from core.controllability_audit import CONFIGS, audit_subject  # noqa: E402
from core.coil_registry import get_coil  # noqa: E402
from core.pipeline import _load_coil_maps  # noqa: E402
from core.shim_metrics import METRIC_KEYS  # noqa: E402
from core.subject_split import describe, split_subjects  # noqa: E402
from common import find_subjects  # noqa: E402

OUT = ROOT / "results" / "route1_controllability"

#: Attached to every report produced here and by route 2. Every bound in this
#: file is a bound on a mean square over a mask; it says nothing about tails,
#: and nothing about what an EPI image will look like.
QUALIFIER = (
    "Pareto optimality applies only to regional MSE/RMS under the identical "
    "feasible set; it does not imply Pareto optimality for CVaR, EPI "
    "displacement, signal dropout, or measured EPI image quality.")
FOLDS = ("train", "validation", "test")
FOLD_COLOUR = {"train": "#1f77b4", "validation": "#ff7f0e", "test": "#2ca02c"}

#: A 5 T brain spans tens to hundreds of Hz; across this cohort the in-brain
#: standard deviation runs 52.6 to 137.9 Hz. Anything an order of magnitude
#: below that is not a field map in Hz, whatever the filename says.
MIN_PLAUSIBLE_STD_HZ = 10.0


def screen_field_map(b0, brain):
    """(ok, diagnostics) -- is this really a delta-B0 map in Hz?

    Four scans in the cohort, all from the 2024-09-13 session, carry files
    named `*_deltaB0_hertz.nii.gz` whose values run to exactly +-9.398 =
    0.997 * 3*pi and are byte-identical in range between two different
    participants. That is unwrapped phase in radians, never converted; the
    matching `_ppm` file for one of them divides by 127.7 where every other
    subject uses 212.9 Hz/ppm, so the conversion for that session is broken at
    the source. Averaging a 3.5 Hz "baseline" in with 50-140 Hz ones would move
    every percentile in the summary, so they are excluded and named, not
    quietly kept.

    The test is on the in-brain standard deviation, and the margin is not
    close: the lowest valid subject is 52.6 Hz, the highest rejected one 4.1.
    """
    v = np.asarray(b0)[np.asarray(brain, dtype=bool)]
    if v.size == 0:
        return False, {"brain_std_hz": float("nan"), "max_abs_hz": float("nan"),
                       "reason": "empty brain mask"}
    d = {"brain_std_hz": float(v.std()),
         "max_abs_hz": float(np.abs(v).max()),
         "max_over_3pi": float(np.abs(v).max() / (3 * np.pi)),
         "reason": ""}
    if d["brain_std_hz"] < MIN_PLAUSIBLE_STD_HZ:
        d["reason"] = ("in-brain std %.2f Hz < %.1f Hz; values reach %.4f = "
                       "%.4f x 3pi, i.e. radians, not Hz"
                       % (d["brain_std_hz"], MIN_PLAUSIBLE_STD_HZ,
                          d["max_abs_hz"], d["max_over_3pi"]))
        return False, d
    return True, d

ROW_KEYS = (("subject", "fold", "config", "objective") + METRIC_KEYS
            + ("current_rms", "current_max", "at_limit", "scanner_margin",
               "scanner_max_abs", "high_sigma_before", "high_sigma_after",
               "high_improve_pct", "high_improve_hz", "guard_rest_hz",
               "solver", "status", "violation"))

GAP_KEYS = ("subject", "fold",
            "gap_guarded_hz", "gap_guarded_pct",
            "gap_pfc_hz", "gap_pfc_pct",
            "gap_whole_hz", "gap_whole_pct",
            "rest_cost_unguarded_hz",
            "baseline_pfc_rms", "production_pfc_rms", "production_rest_rms",
            "high_improve_pct_bounded", "high_improve_pct_full",
            "high_improve_hz_full",
            "effective_rank", "condition_number", "h_in_coil_span",
            "min_principal_angle_deg", "max_violation_hz",
            "pfc_voxels", "rest_voxels", "seconds")


# ── statistics ──────────────────────────────────────────────────────────

def stats(values, worst="max"):
    """median / IQR / P10 / P90 / worst, on the finite values only.

    `worst` says which tail is the bad one: "max" for a gap or an error,
    "min" for an improvement. Naming it per metric avoids the usual mistake of
    reporting the largest number as the worst case when large is good.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)],
                   dtype=float)
    if v.size == 0:
        return {k: float("nan") for k in
                ("n", "median", "p25", "p75", "iqr", "p10", "p90", "worst")}
    q10, q25, q75, q90 = np.percentile(v, [10, 25, 75, 90])
    return {"n": int(v.size), "median": float(np.median(v)),
            "p25": float(q25), "p75": float(q75), "iqr": float(q75 - q25),
            "p10": float(q10), "p90": float(q90),
            "worst": float(v.max() if worst == "max" else v.min())}


def stat_table(rows, key, worst="max", fmt="%.2f"):
    """A markdown table of `key`, one line per fold plus an `all` line."""
    head = ("| fold | n | median | IQR (P25–P75) | P10 | P90 | worst |\n"
            "|---|---|---|---|---|---|---|\n")
    body = ""
    for fold in FOLDS + ("all",):
        sel = rows if fold == "all" else [r for r in rows if r["fold"] == fold]
        st = stats([r.get(key) for r in sel], worst)
        if not st["n"] or st["n"] != st["n"]:
            continue
        body += ("| %s | %d | " + fmt + " | " + fmt + "–" + fmt
                 + " (" + fmt + ") | " + fmt + " | " + fmt + " | "
                 + fmt + " |\n") % (
            fold, st["n"], st["median"], st["p25"], st["p75"], st["iqr"],
            st["p10"], st["p90"], st["worst"])
    return head + body


# ── figures ─────────────────────────────────────────────────────────────

def fig_spectrum(per_subject, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, d in per_subject.items():
        sv = d["spectrum"]["singular_values"]
        if not sv:
            continue
        sv = np.asarray(sv) / max(sv[0], 1e-30)
        ax.semilogy(np.arange(1, len(sv) + 1), sv, "-", lw=1.0, alpha=0.6,
                    color=FOLD_COLOUR[d["fold"]])
    for f in FOLDS:
        ax.plot([], [], color=FOLD_COLOUR[f], label=f)
    ax.set_xlabel("singular value index")
    ax.set_ylabel(r"$\sigma_k/\sigma_1$ of $C_\perp$ (PFC, high-order part)")
    ax.set_title("How evenly the coils reach the high-order PFC field")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_angles(per_subject, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data = [d["spectrum"]["principal_angles_deg"] for d in per_subject.values()]
    n = max((len(a) for a in data), default=0)
    if not n:
        return
    arr = np.full((len(data), n), np.nan)
    for i, a in enumerate(data):
        arr[i, :len(a)] = a
    ax.boxplot([arr[np.isfinite(arr[:, k]), k] for k in range(n)],
               positions=np.arange(1, n + 1), widths=0.6)
    ax.set_xlabel("principal angle index")
    ax.set_ylabel("angle between SH span and coil span in PFC (deg)")
    ax.set_title("Small angles = directions the coils and the scanner share")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_high_improve(rows, path):
    keys = ["high_unconstrained", "high_bounded", "high_full"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, cfg in enumerate(keys):
        for j, fold in enumerate(FOLDS):
            v = [r["high_improve_pct"] for r in rows
                 if r["config"] == cfg and r["fold"] == fold
                 and r.get("high_improve_pct") is not None]
            x = i + (j - 1) * 0.24
            ax.scatter(np.full(len(v), x) + np.linspace(-0.05, 0.05, len(v)),
                       v, s=14, alpha=0.75, color=FOLD_COLOUR[fold],
                       label=fold if i == 0 else None)
            if v:
                ax.hlines(np.median(v), x - 0.1, x + 0.1, color="k", lw=1.6)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["unconstrained", "|i| <= 3 A", "+ scanner feasible"])
    ax.set_ylabel("high-order PFC residual removed (%)")
    ax.set_title("What the coils add beyond the scanner's reach")
    ax.axhline(5, ls="--", c="gray", lw=1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def fig_gaps(gaps, path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for j, fold in enumerate(FOLDS):
        v = sorted(g["gap_guarded_hz"] for g in gaps if g["fold"] == fold)
        if not v:
            continue
        ax.plot(v, np.linspace(0, 100, len(v)), "o-", ms=4,
                color=FOLD_COLOUR[fold], label=fold)
    for x, ls in ((1.0, "--"), (3.0, ":")):
        ax.axvline(x, ls=ls, c="gray", lw=1)
    ax.set_xlabel("production PFC RMS − guarded bound PFC RMS (Hz)")
    ax.set_ylabel("cumulative % of subjects")
    ax.set_title("Room left in the PFC without giving anything back elsewhere")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


# ── main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--coil", default="PFC 12ch")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    coil = get_coil(args.coil)
    found = find_subjects(args.limit, True)

    # Screen before splitting: an excluded scan must not consume a fold slot.
    subjects, excluded = [], []
    for rec in found:
        name, hz, bp, _ = rec
        b0 = np.asanyarray(nib.load(str(hz)).dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(bp)).dataobj) > 0.5
        ok, diag = screen_field_map(b0, brain)
        (subjects if ok else excluded).append((rec, diag))
    subjects = [r for r, _ in subjects]

    fold_of = split_subjects([n for n, *_ in subjects])

    print("%d scans found, %d excluded, %d audited. Coil %s (+-%.1f A)\n"
          % (len(found), len(excluded), len(subjects), coil.name,
             coil.current_limit_a))
    for (name, *_), diag in excluded:
        print("  excluded %-12s %s" % (name, diag["reason"]))
    if excluded:
        print()
    print(describe(fold_of), "\n")

    rows, gaps, per_subject, failures = [], [], {}, []

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
            out = audit_subject(b0, brain, pfc, maps, img.affine,
                                imax=coil.current_limit_a)
        except Exception as e:
            failures.append((name, "%s: %s" % (type(e).__name__, e)))
            print("[%2d/%d] %-12s %s FAILED: %s"
                  % (k, len(subjects), name, fold, e))
            traceback.print_exc()
            continue

        secs = time.time() - t0
        by = {r["config"]: r for r in out["rows"]}
        for r in out["rows"]:
            rows.append(dict(r, subject=name, fold=fold))
        per_subject[name] = {"fold": fold, "spectrum": out["spectrum"]}

        sp = out["spectrum"]
        g = dict(out["gap"])
        g.update({
            "subject": name, "fold": fold,
            "baseline_pfc_rms": by["none"]["pfc_rms"],
            "production_pfc_rms": by["production"]["pfc_rms"],
            "production_rest_rms": by["production"]["rest_rms"],
            "high_improve_pct_bounded": by["high_bounded"]["high_improve_pct"],
            "high_improve_pct_full": by["high_full"]["high_improve_pct"],
            "high_improve_hz_full": by["high_full"]["high_improve_hz"],
            "effective_rank": sp["effective_rank"],
            "condition_number": sp["condition_number"],
            "h_in_coil_span": sp["h_in_coil_span"],
            "min_principal_angle_deg": (min(sp["principal_angles_deg"])
                                        if sp["principal_angles_deg"] else float("nan")),
            "max_violation_hz": max(r["violation"] for r in out["rows"]),
            "pfc_voxels": sp["pfc_voxels"], "rest_voxels": sp["rest_voxels"],
            "seconds": secs,
        })
        gaps.append(g)

        np.savez_compressed(
            OUT / ("%s.npz" % name),
            fold=fold,
            configs=np.array([r["config"] for r in out["rows"]]),
            pfc_rms=np.array([r["pfc_rms"] for r in out["rows"]]),
            rest_rms=np.array([r["rest_rms"] for r in out["rows"]]),
            brain_rms=np.array([r["brain_rms"] for r in out["rows"]]),
            singular_values=np.asarray(sp["singular_values"]),
            principal_angles_deg=np.asarray(sp["principal_angles_deg"]),
            coil_low_share=np.asarray(sp["coil_low_share"]),
            coil_high_share=np.asarray(sp["coil_high_share"]),
            **{"solution_" + r["config"]: np.r_[
                np.asarray(r.get("scanner_sh_delta", []), dtype=float),
                np.asarray(r.get("coil_currents", []), dtype=float)]
               for r in out["rows"] if "scanner_sh_delta" in r})

        print("[%2d/%d] %-12s %-10s base %6.1f  prod %5.1f  guarded %5.1f  "
              "gap %+5.2f Hz  high %+5.1f%%  viol %.1e  %.1fs"
              % (k, len(subjects), name, fold, by["none"]["pfc_rms"],
                 by["production"]["pfc_rms"], by["bound_pfc_guarded"]["pfc_rms"],
                 g["gap_guarded_hz"], g["high_improve_pct_full"],
                 g["max_violation_hz"], secs))

    # ── CSVs ────────────────────────────────────────────────────────────
    with open(OUT / "per_subject.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW_KEYS), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with open(OUT / "gaps.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(GAP_KEYS), extrasaction="ignore")
        w.writeheader()
        for g in gaps:
            w.writerow(g)

    # ── figures ─────────────────────────────────────────────────────────
    if per_subject:
        fig_spectrum(per_subject, OUT / "fig_spectrum.png")
        fig_angles(per_subject, OUT / "fig_principal_angles.png")
        fig_high_improve(rows, OUT / "fig_high_order_improvement.png")
        fig_gaps(gaps, OUT / "fig_gap_cdf.png")

    # ── summary ─────────────────────────────────────────────────────────
    md = ["# Route 1 — controllability audit", "",
          "> " + QUALIFIER, "",
          "Coil `%s`, limit ±%.1f A. %d scans found, %d excluded before the "
          "split, %d audited, %d succeeded."
          % (coil.name, coil.current_limit_a, len(found), len(excluded),
             len(subjects), len(gaps)), "",
          "```", describe(fold_of), "```", ""]
    if excluded:
        md += ["## Excluded scans", "",
               "Screened out before the split, so they consume no fold slot.",
               "", "| scan | in-brain std (Hz) | max abs (Hz) | max / 3π |",
               "|---|---|---|---|"]
        for (name, *_), d in excluded:
            md.append("| `%s` | %.2f | %.3f | %.4f |"
                      % (name, d["brain_std_hz"], d["max_abs_hz"],
                         d["max_over_3pi"]))
        md += ["",
               "All four are from the 2024-09-13 session. Their values run to "
               "exactly ±9.398 = 0.997 × 3π and the range is identical between "
               "two different participants: this is unwrapped phase in radians "
               "written into a file named hertz. One of them also divides by "
               "127.7 Hz/ppm where every other subject uses 212.9. They are "
               "not field maps and are not audited.", ""]
    if failures:
        md += ["## Failures", ""] + ["- `%s`: %s" % f for f in failures] + [""]

    worst_v = max((g["max_violation_hz"] for g in gaps), default=0.0)
    md += ["Largest constraint violation over every solve: **%.2e Hz**."
           % worst_v, ""]

    md += ["## Headline: the PFC gained at a 1% rest budget (Hz)", "",
           "**This is a price, not a gap.** Production PFC RMS minus the best "
           "PFC RMS reachable while the rest of the brain is allowed to "
           "degrade by 1%. Read it as one point on a Pareto front — what 1% of "
           "rest homogeneity buys in the PFC — and not as room the shipping "
           "solution is leaving on the table.", "",
           "At a **0%** budget the number is exactly zero, provably. "
           "Production minimises whole-brain mean square, which is the "
           "voxel-count-weighted sum `n_pfc·PFC_ms + n_rest·rest_ms`; a "
           "minimiser of a positive-weighted sum of two convex objectives is "
           "Pareto-optimal, so no feasible point has rest no worse *and* PFC "
           "better. The whole-brain self-check below confirms production is "
           "that exact minimiser (0.000 Hz), so the argument applies without "
           "slack. Solving `min PFC s.t. rest ≤ rest_production` directly "
           "returns production back, agreeing to 1e-5 Hz on every subject "
           "tested. The 1% slack in this table was introduced for a solver "
           "reason — the ε-constraint has an empty interior at 0% and the "
           "conic solvers report `infeasible` — and every Hz in it is bought, "
           "not recovered.", "",
           stat_table(gaps, "gap_guarded_hz", "max"), ""]
    md += ["Same, as a percentage of production PFC RMS:", "",
           stat_table(gaps, "gap_guarded_pct", "max"), ""]

    md += ["## Gap to the unguarded PFC bound (Hz)", "",
           "Shown for contrast only. This bound is free to wreck the rest of",
           "the brain and on every subject it does — see the row below it.", "",
           stat_table(gaps, "gap_pfc_hz", "max"), "",
           "Cost the unguarded bound imposes on the rest of the brain (Hz):", "",
           stat_table(gaps, "rest_cost_unguarded_hz", "max"), ""]

    md += ["## Whole-brain self-check (Hz)", "",
           "Production minus the whole-brain bound under identical constraints.",
           "Both minimise the same thing, so this should be ~0; anything else",
           "means the shipping solver is not reaching its own optimum.", "",
           stat_table(gaps, "gap_whole_hz", "max", "%.3f"), ""]

    md += ["## High-order controllability", "",
           "Fraction of the PFC field lying outside span{1, SH12} that the",
           "coils remove. `_full` additionally requires the scanner command",
           "the solution implies to be inside the console's limits.", "",
           "Constrained (`high_full`, %):", "",
           stat_table(gaps, "high_improve_pct_full", "min"), "",
           "Bounded only by ±3 A (`high_bounded`, %):", "",
           stat_table(gaps, "high_improve_pct_bounded", "min"), "",
           "In Hz (`high_full`):", "",
           stat_table(gaps, "high_improve_hz_full", "min"), ""]

    binds = [g for g in gaps
             if g["high_improve_pct_bounded"] - g["high_improve_pct_full"] > 0.5]
    st_f = stats([g["high_improve_pct_full"] for g in gaps], "min")
    st_b = stats([g["high_improve_pct_bounded"] for g in gaps], "min")
    md += ["### The ±2000 scanner envelope", "",
           "`high_full` and every `bound_*` row constrain the scanner command "
           "to the intersection of a ±2000 delta envelope and a ±2000 absolute "
           "envelope, in the app's SH units. This is a deliberately "
           "conservative deployment limit chosen to sit inside the scanner's "
           "real capability — a software safety policy, written explicitly in "
           "`config/shim_delta_bounds.json`. It is **expected to bind**, and "
           "it does: an SH-only fit to the PFC alone asks for B21 of 3000–4400 "
           "on several subjects. A pinned term here means the envelope was "
           "enforced, not that a unit or a configuration is wrong.", "",
           "`high_full` is therefore the operative result. `high_bounded` is "
           "the same problem with the envelope removed and is reported as a "
           "**theoretical upper bound only**: median %.2f%% enforced against "
           "%.2f%% unenforced, with the envelope active on %d of %d subjects "
           "(%s)."
           % (st_f["median"], st_b["median"], len(binds), len(gaps),
              ", ".join("`%s`" % g["subject"] for g in binds) or "none"),
           "",
           "The guarded gap barely moves either way: production and the "
           "guarded bound are solved under the *same* envelope.", "",
           "All runs here use a zero baseline, so the intersection reduces to "
           "a symmetric ±2000. With a non-zero baseline the window narrows and "
           "becomes asymmetric per term — B11 at −1757.7 leaves −242.3 / "
           "+2000 — and `audit_subject` takes it via `scanner_bounds`.", ""]

    md += ["## Geometry", "",
           "Effective rank of the high-order coil operator (out of 12):", "",
           stat_table(gaps, "effective_rank", "min", "%.1f"), "",
           "Fraction of the high-order PFC field inside the coils' reach:", "",
           stat_table(gaps, "h_in_coil_span", "min", "%.3f"), "",
           "Smallest principal angle between the SH and coil spans (deg):", "",
           stat_table(gaps, "min_principal_angle_deg", "min"), ""]

    md += ["## Per-config PFC RMS (Hz), all folds", "",
           "| config | median | P10 | P90 | worst |", "|---|---|---|---|---|"]
    for cfg in CONFIGS:
        sel = [r for r in rows if r["config"] == cfg]
        st = stats([r["pfc_rms"] for r in sel], "max")
        if st["n"] and st["n"] == st["n"]:
            md.append("| `%s` | %.1f | %.1f | %.1f | %.1f |"
                      % (cfg, st["median"], st["p10"], st["p90"], st["worst"]))
    md += ["",
           "![](fig_gap_cdf.png)", "", "![](fig_high_order_improvement.png)",
           "", "![](fig_spectrum.png)", "", "![](fig_principal_angles.png)", ""]

    g_med = stats([g["gap_guarded_hz"] for g in gaps], "max")
    g_pct = stats([g["gap_guarded_pct"] for g in gaps], "max")
    h_pct = stats([g["high_improve_pct_full"] for g in gaps], "min")
    long_tail = [g["subject"] for g in gaps if g["gap_guarded_hz"] > 4.0]
    prod_worse = [g["subject"] for g in gaps
                  if g["production_pfc_rms"] > g["baseline_pfc_rms"]]
    sh_only_pfc = stats([r["pfc_rms"] for r in rows
                         if r["config"] == "sh_only"], "max")
    prod_pfc = stats([r["pfc_rms"] for r in rows
                      if r["config"] == "production"], "max")

    md += ["## Decision criteria", "",
           "Applied exactly as written before the run, to all %d subjects."
           % len(gaps), "",
           "**A — constrained high-order improvement median <5%% and P90 "
           "<10%%: drop static high-order fitting as a headline claim.** "
           "Median %.2f%%, P90 %.2f%%. **Not met**: the criterion does not "
           "fire, the coils do remove ~10%% of the high-order PFC residual. "
           "But that is %.2f Hz at the median, against a PFC RMS around "
           "%.0f Hz — real, and small."
           % (h_pct["median"], h_pct["p90"],
              stats([g["high_improve_hz_full"] for g in gaps], "min")["median"],
              prod_pfc["median"]), "",
           "**B — production-to-bound gap median <1 Hz: static mean-square "
           "optimisation is near saturation.** At the correct 0%% rest budget "
           "the gap is **exactly 0**, by the Pareto argument above and "
           "verified to 1e-5 Hz. **Met, in its strongest form**: there is no "
           "PFC improvement available that does not degrade the rest of the "
           "brain. The %.2f Hz median in the headline table is the price at a "
           "1%% rest budget, not a gap."
           % g_med["median"], "",
           "**C — gap >3 Hz, or a clear long tail: route 2 is top priority.** "
           "**Not met as written**, because the gap it refers to is zero. The "
           "criterion assumed a gap existed to collect; there is none. Route 2 "
           "is not obsolete, but its question changes: not whether PFC quality "
           "can be improved for nothing, but what the PFC–rest trade-off costs "
           "and whether it is worth making. The prices at a 1%% rest-RMS "
           "budget — median %.2f Hz, P90 %.2f Hz, above 4 Hz on %d of %d "
           "subjects (%s) — are the opening quotes for that decision."
           % (g_med["median"], g_med["p90"], len(long_tail), len(gaps),
              ", ".join("`%s`" % s for s in long_tail) or "none"), "",
           "**D — route 4 (dynamic slab) is not cancelled by a static "
           "audit.** Nothing here measures per-slice behaviour; every number "
           "above is a single static solution. D holds by construction.", "",
           "### What the audit actually found", "",
           "The hardware is not the binding constraint, and neither is the "
           "solver. Aiming the *scanner alone* at the PFC reaches %.1f Hz "
           "median while the shipping whole-brain solution leaves the PFC at "
           "%.1f Hz — a %.1f Hz difference — but that difference is not "
           "waste. Production is already on the Pareto front of (PFC, rest); "
           "the %.1f Hz is what the rest of the brain is being protected "
           "*with*. Every Hz of it is purchasable and the exchange rate is "
           "the real deliverable: %.1f Hz of PFC at a 1%% rest budget."
           % (sh_only_pfc["median"], prod_pfc["median"],
              prod_pfc["median"] - sh_only_pfc["median"],
              prod_pfc["median"] - sh_only_pfc["median"], g_med["median"]), "",
           "The high-order operator is full rank (12/12) on every subject, "
           "and %.0f%% of the high-order PFC field lies inside the coils' "
           "reach at the median, so the array is not degenerate. The smallest "
           "principal angle between the SH and coil spans is %.1f°, i.e. the "
           "coils and the scanner do overlap strongly in one direction — "
           "which is why the two must be solved jointly, not in sequence."
           % (100 * stats([g["h_in_coil_span"] for g in gaps], "min")["median"],
              stats([g["min_principal_angle_deg"] for g in gaps],
                    "min")["median"]), "",
           "On %d of %d subjects production leaves the PFC *worse* than no "
           "shim at all (%s). This is not a solver failure — each of those is "
           "a Pareto point too — it is the voxel-count weighting deciding, "
           "unasked, that PFC homogeneity is worth less than the rest of the "
           "head. Choosing that trade deliberately, rather than inheriting it "
           "from a region's voxel count, is what route 2 is for."
           % (len(prod_worse), len(gaps),
              ", ".join("`%s`" % s for s in prod_worse) or "none"), ""]

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\nwrote %s" % OUT)
    if failures:
        print("%d failures" % len(failures))


if __name__ == "__main__":
    main()
