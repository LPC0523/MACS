"""One shared centre frequency, then regional RMS. The metric everything reports.

Every region is measured after subtracting the *same* f0. Demeaning the PFC and
the rest separately -- which an ROI-centred standard deviation does implicitly --
hides the one thing that most needs watching: a solution can look excellent in
both regions while sitting at two different frequencies, and the offset between
them is a real frequency shift that the scanner cannot remove for both at once.
Prescan locks one f0 for the whole acquisition.

So `f0` is a single free variable, shared, and `PFC RMS` / `rest RMS` are
computed from `r = b + S s + C i + f0`. The regional means and their difference
are always reported alongside, so that offset is visible rather than absorbed.

`roi_centered_std` is kept only to compare against figures produced before this
was settled. It is a secondary number and must never be the one optimised.
"""

from __future__ import annotations

import numpy as np

__all__ = ["optimal_f0", "regional_metrics", "metric_row", "METRIC_KEYS"]

METRIC_KEYS = (
    "f0_hz",
    "pfc_rms", "rest_rms", "brain_rms",
    "pfc_mean", "rest_mean", "mean_diff",
    "pfc_p95", "pfc_p99", "pfc_worst10",
    "pfc_std", "rest_std",           # secondary, historical compatibility only
)


def optimal_f0(residual, brain_mask) -> float:
    """The single f0 that minimises whole-brain RMS of `residual + f0`.

    That is just the negated in-brain mean. It is computed over the brain rather
    than over the PFC because prescan sets one frequency for the head, not for a
    region; choosing it on the PFC would silently trade the rest of the brain
    away through a channel the optimiser was never asked about.
    """
    v = np.asarray(residual)[np.asarray(brain_mask, dtype=bool)]
    return float(-np.mean(v)) if v.size else 0.0


def regional_metrics(residual, pfc_mask, rest_mask, brain_mask=None,
                     f0: float = None) -> dict:
    """Metrics for one solution, all sharing a single f0.

    `residual` is the field before f0 is applied. Pass `f0` to score a solution
    whose f0 was a decision variable; leave it None to use the optimal one.
    """
    r = np.asarray(residual, dtype=float)
    pfc = np.asarray(pfc_mask, dtype=bool)
    rest = np.asarray(rest_mask, dtype=bool)
    brain = pfc | rest if brain_mask is None else np.asarray(brain_mask, dtype=bool)

    if (pfc & rest).any():
        raise ValueError("pfc_mask and rest_mask overlap; they must be disjoint")

    if f0 is None:
        f0 = optimal_f0(r, brain)
    rf = r + f0

    p, q = rf[pfc], rf[rest]

    def rms(v):
        return float(np.sqrt(np.mean(v ** 2))) if v.size else float("nan")

    def mean(v):
        return float(np.mean(v)) if v.size else float("nan")

    absp = np.abs(p)
    out = {
        "f0_hz": float(f0),
        "pfc_rms": rms(p),
        "rest_rms": rms(q),
        "brain_rms": rms(rf[brain]),
        "pfc_mean": mean(p),
        "rest_mean": mean(q),
        "pfc_p95": float(np.percentile(absp, 95)) if p.size else float("nan"),
        "pfc_p99": float(np.percentile(absp, 99)) if p.size else float("nan"),
        # Mean of the worst decile -- the quantity a CVaR objective targets.
        "pfc_worst10": (float(np.mean(np.sort(absp)[int(0.9 * absp.size):]))
                        if p.size else float("nan")),
        # Secondary. Each region demeaned on its own, which is what the earlier
        # numbers did; reported only so old figures remain comparable.
        "pfc_std": float(np.std(p)) if p.size else float("nan"),
        "rest_std": float(np.std(q)) if q.size else float("nan"),
    }
    out["mean_diff"] = out["pfc_mean"] - out["rest_mean"]
    return out


def metric_row(name: str, residual, pfc_mask, rest_mask, brain_mask=None,
               f0: float = None, **extra) -> dict:
    """`regional_metrics` plus identifying columns, ready for a CSV row."""
    row = {"config": name}
    row.update(regional_metrics(residual, pfc_mask, rest_mask, brain_mask, f0))
    row.update(extra)
    return row
