"""Are the Slepian modes unstable, or only their individual eigenvectors?

Mode 0's current vector agrees at |cos| 0.59 on average across the train fold
and 0.004 at worst. Two different things produce that:

  * near-degenerate eigenvalues, where any rotation inside a subspace is an
    equally valid eigenvector, so the individual vectors are arbitrary but the
    subspace they span is not;
  * genuinely different optima, where the subspaces differ too.

Only the second would rule out shipping a fixed mode set. Principal angles
between the K-dimensional spans separate them.

    D:\\python.exe tools/research/pfc_modes/phase1b_subspace.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import OUT_DIR, Progress, write_json  # noqa: E402


def subspace_affinity(A, B, k):
    """Mean cos^2 of the principal angles between the two k-dim spans.

    1.0 = same subspace, 0.0 = orthogonal. Averaged over the k angles so it is
    comparable across k.
    """
    Qa, _ = np.linalg.qr(A[:, :k])
    Qb, _ = np.linalg.qr(B[:, :k])
    sv = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    sv = np.clip(sv, 0.0, 1.0)
    return float(np.mean(sv ** 2)), float(np.min(sv) ** 2)


def main():
    P = Progress("Phase 1b — subspace stability of the Slepian modes")
    npz = np.load(OUT_DIR / "slepian_modes.npz")
    names = sorted({k.split("__")[0] for k in npz.files})
    P.log(f"{len(names)} scans")

    cur = {n: np.asarray(npz[f"{n}__currents"], dtype=float).T for n in names}
    lam = {n: np.asarray(npz[f"{n}__lambda"], dtype=float) for n in names}

    P.log("eigenvalue gaps (degeneracy would make single vectors arbitrary)")
    gaps = []
    for n in names:
        L = lam[n]
        g = [round(float(L[i] - L[i + 1]), 4) for i in range(min(4, len(L) - 1))]
        gaps.append({"subject": n, **{f"gap{i}_{i+1}": v for i, v in enumerate(g)},
                     **{f"lambda{i}": round(float(L[i]), 4)
                        for i in range(min(5, len(L)))}})
    g01 = np.array([r["gap0_1"] for r in gaps])
    g12 = np.array([r["gap1_2"] for r in gaps])
    P.log(f"    lambda0-lambda1: mean {g01.mean():.3f}  min {g01.min():.3f}")
    P.log(f"    lambda1-lambda2: mean {g12.mean():.3f}  min {g12.min():.3f}")

    P.log("pairwise subspace affinity (mean cos^2 of principal angles)")
    rows = []
    for k in (1, 2, 3, 4, 6):
        vals, mins = [], []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                m, lo = subspace_affinity(cur[names[i]], cur[names[j]], k)
                vals.append(m); mins.append(lo)
        vals = np.array(vals); mins = np.array(mins)
        rows.append({"k": k, "mean_affinity": round(float(vals.mean()), 4),
                     "min_affinity": round(float(vals.min()), 4),
                     "worst_single_angle_cos2": round(float(mins.min()), 4)})
        P.log(f"    k={k}: mean {vals.mean():.3f}  worst pair {vals.min():.3f}  "
              f"worst single direction {mins.min():.3f}")

    # A random 12-dim baseline: what affinity would two unrelated k-dim
    # subspaces score? Without this, 0.5 looks meaningful when it may be chance.
    rng = np.random.default_rng(0)
    P.log("chance level from random subspaces in R^12")
    chance = {}
    for k in (1, 2, 3, 4, 6):
        vals = []
        for _ in range(400):
            A = rng.normal(size=(12, k)); B = rng.normal(size=(12, k))
            vals.append(subspace_affinity(A, B, k)[0])
        chance[k] = float(np.mean(vals))
        P.log(f"    k={k}: chance {chance[k]:.3f}")
    for r in rows:
        r["chance_affinity"] = round(chance[r["k"]], 4)
        r["above_chance"] = round(r["mean_affinity"] - chance[r["k"]], 4)

    # Consensus subspace: the leading directions of all scans stacked. If a
    # fixed mode set is shippable at all, this is what it would be.
    P.log("consensus subspace from all scans (SVD of stacked leading modes)")
    cons = {}
    for k in (1, 2, 3, 4):
        M = np.concatenate([cur[n][:, :k] / np.linalg.norm(cur[n][:, :k], axis=0)
                            for n in names], axis=1)
        U, sv, _ = np.linalg.svd(M, full_matrices=False)
        energy = sv ** 2 / np.sum(sv ** 2)
        caps = [subspace_affinity(U[:, :k], cur[n], k)[0] for n in names]
        cons[k] = {"consensus_energy_in_first_k": round(float(energy[:k].sum()), 4),
                   "mean_capture": round(float(np.mean(caps)), 4),
                   "min_capture": round(float(np.min(caps)), 4),
                   "worst_subject": names[int(np.argmin(caps))],
                   "consensus_basis": np.round(U[:, :k], 5).tolist()}
        P.log(f"    k={k}: consensus captures each scan's span at "
              f"mean {cons[k]['mean_capture']:.3f}, "
              f"min {cons[k]['min_capture']:.3f} ({cons[k]['worst_subject']})")

    with open(OUT_DIR / "subspace_stability.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    write_json(OUT_DIR / "subspace_stability.json",
               {"n_scans": len(names), "pairwise": rows,
                "eigenvalue_gaps": gaps, "consensus": cons})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
