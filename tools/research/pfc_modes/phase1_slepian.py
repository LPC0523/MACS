"""Phase 1 + w_P + Slepian: what can this array do that the harmonics cannot?

    D:\\python.exe tools/research/pfc_modes/phase1_slepian.py [--fold train] [--limit N]

The whole-brain projection first: P8 is the field the console can already
produce, so only its complement is the array's to contribute.

    r_n      = (I - P8) b_n          target residual, per scan
    C_perp   = (I - P8) C            the array's deliverable-independent part

The projection is done over the *whole brain*, never inside the PFC. Projecting
within the PFC would discard exactly the property under study: a coil field
that looks low-order over the target but falls away outside it is, globally, not
low-order at all, and that difference is the array's entire reason to exist.

Then the Slepian problem, in the array's own reachable space:

    C_perpᵀ W_P C_perp v  =  λ  C_perpᵀ W_B C_perp v

λ is the fraction of the mode's field energy that lands near the PFC. This is a
spatial-concentration problem, not a PCA -- the modes are ordered by where their
energy goes, not by how much variance they explain.

Solved by whitening against B and truncating its near-null directions, because
B is rank-deficient by construction: C_perp has at most 12 columns and the
harmonics have already taken 8 directions out of them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (            # noqa: E402
    SH8, IMAX, OUT_DIR, Progress, basis_with_a00, cohort, coil_maps_for,
    load_subject, sh_matrix, triplane_png, write_json)

#: Frozen before any result was seen. 5 and 10 mm are sensitivity checks only
#: and never select anything.
TAPER_MM = 7.5
TAPER_SENSITIVITY = (5.0, 10.0)

#: Relative tolerance for discarding near-null directions of B when whitening.
WHITEN_RTOL = 1e-9


def taper_weight(pfc, affine, taper_mm):
    """1 inside the PFC, cosine roll-off to 0 over `taper_mm` outside it.

    A hard mask would make the concentration ratio depend on the mask's exact
    boundary, which is a registration artefact rather than physics; it also
    rewards modes with a discontinuity at the edge. The roll-off is in
    millimetres, so it means the same thing on every voxel size in the cohort.
    """
    from scipy.ndimage import distance_transform_edt

    spacing = np.sqrt((np.asarray(affine)[:3, :3] ** 2).sum(axis=0))
    d = distance_transform_edt(~pfc, sampling=spacing)
    w = np.zeros(pfc.shape, dtype=np.float64)
    w[pfc] = 1.0
    band = (~pfc) & (d <= taper_mm)
    w[band] = 0.5 * (1.0 + np.cos(np.pi * d[band] / taper_mm))
    return w


def whitened_gevp(A, B, rtol=WHITEN_RTOL):
    """Solve A v = lambda B v for symmetric A, PSD and singular B.

    scipy.linalg.eigh(A, B) requires B positive definite; here B is not, so the
    problem is restricted to B's range first. Returns (eigenvalues descending,
    eigenvectors as columns, rank kept).
    """
    B = 0.5 * (B + B.T)
    A = 0.5 * (A + A.T)
    mu, U = np.linalg.eigh(B)
    keep = mu > rtol * max(float(mu.max()), 0.0)
    if not keep.any():
        raise ValueError("B has no usable range; the array contributes nothing")
    T = U[:, keep] / np.sqrt(mu[keep])
    Aw = T.T @ A @ T
    Aw = 0.5 * (Aw + Aw.T)
    lam, V = np.linalg.eigh(Aw)
    order = np.argsort(lam)[::-1]
    return lam[order], T @ V[:, order], int(keep.sum())


def analyse(s, maps, basis, P, taper_mm=TAPER_MM, n_modes=8):
    brain, pfc = s["brain"], s["pfc"]
    nb = int(brain.sum())
    C = maps.reshape(-1, maps.shape[-1])[brain.ravel(), :].astype(np.float64)
    A8 = sh_matrix(basis, SH8, brain)
    b = s["b0"][brain].astype(np.float64)

    # W_B counts every brain voxel once and W_P is the tapered weight in [0, 1]
    # on the same voxels. Deliberately NOT normalised to sum 1: with w_P <= w_B
    # pointwise, lambda = sum(w_P f^2) / sum(w_B f^2) lands in [0, 1] and reads
    # as "the fraction of this mode's field energy that sits near the PFC",
    # which is what a Slepian concentration eigenvalue means. Normalising both
    # to probabilities instead gives an energy-*density* ratio, which exceeds 1
    # whenever the mode is at all concentrated and is not comparable with the
    # region's own size.
    wB = np.ones(nb)

    Q, _ = np.linalg.qr(A8)                    # orthonormal basis of the SH span
    P8C = Q @ (Q.T @ C)
    C_perp = C - P8C
    r = b - Q @ (Q.T @ b)

    audit = {
        "reconstruction_max_abs": float(np.max(np.abs(C - (P8C + C_perp)))),
        "coil_field_max_abs": float(np.max(np.abs(C))),
        "sh_orthogonality_max_abs": float(np.max(np.abs(A8.T @ (wB[:, None] * C_perp)))),
        "sh_scale": float(np.max(np.abs(A8.T @ (wB[:, None] * C)))),
        "residual_orthogonality_max_abs": float(np.max(np.abs(A8.T @ (wB * r)))),
        "b0_rms": float(np.sqrt(np.mean(b ** 2))),
        "residual_rms": float(np.sqrt(np.mean(r ** 2))),
    }
    audit["reconstruction_rel"] = audit["reconstruction_max_abs"] / max(audit["coil_field_max_abs"], 1e-12)
    audit["sh_orthogonality_rel"] = audit["sh_orthogonality_max_abs"] / max(audit["sh_scale"], 1e-12)

    wP_vol = taper_weight(pfc, s["affine"], taper_mm)
    wP = wP_vol[brain]
    if wP.sum() <= 0:
        raise ValueError("tapered PFC weight is empty inside the brain")
    # The eigenvalue a spatially uniform field would score: the region's share
    # of the brain. Any lambda must be read against this, not against 1.
    lam_chance = float(wP.sum() / nb)

    # Slepian: concentrate field energy near the PFC, within what the array can
    # still do after the harmonics have taken their part.
    Amat = C_perp.T @ (wP[:, None] * C_perp)
    Bmat = C_perp.T @ (wB[:, None] * C_perp)
    lam, V, rank = whitened_gevp(Amat, Bmat)

    # Normalise each mode to unit whole-brain RMS field, so the current vector
    # printed alongside it is "amps per Hz of field", comparable across modes.
    modes = []
    for k in range(min(n_modes, V.shape[1])):
        v = V[:, k]
        f = C_perp @ v
        # Unit whole-brain RMS, so the current vector reads as amps per Hz.
        rms_b = float(np.sqrt(np.mean(f ** 2)))
        if rms_b <= 0:
            continue
        v = v / rms_b
        f = f / rms_b
        f_full = C @ v                      # the field the coils actually emit
        e_p = float(np.sum(wP * f ** 2) / np.sum(wB * f ** 2))
        rest_sel = wP_vol[brain] < 1e-6
        e_rest = (float(np.mean(f[rest_sel] ** 2)) if rest_sel.any() else 0.0)
        e_pfc_density = float(np.sum(wP * f ** 2) / max(wP.sum(), 1e-12))
        # How much of the emitted field is still SH-reproducible: the mode is
        # built from C_perp, but the hardware emits C.
        sh_share = float(np.linalg.norm(Q.T @ f_full)
                         / max(np.linalg.norm(f_full), 1e-12))
        # Alignment with this scan's own target residual.
        denom = float(np.sqrt(np.sum(wP * r ** 2)) * np.sqrt(np.sum(wP * f ** 2)))
        align = float(np.sum(wP * r * f) / denom) if denom > 0 else 0.0
        peak = float(np.max(np.abs(v)))
        modes.append({
            "mode": k, "lambda_concentration": round(float(lam[k]), 6),
            "lambda_chance_level": round(lam_chance, 6),
            "concentration_gain_vs_chance": round(
                float(lam[k]) / max(lam_chance, 1e-12), 3),
            "pfc_energy_fraction": round(e_p, 4),
            "pfc_energy_density": round(e_pfc_density, 6),
            "rest_energy_density": round(e_rest, 6),
            "density_ratio_pfc_over_rest": round(
                e_pfc_density / max(e_rest, 1e-12), 3),
            "sh_reproducible_share_of_emitted": round(sh_share, 4),
            "align_with_own_residual": round(align, 4),
            "current_rms_per_hz": round(float(np.sqrt(np.mean(v ** 2))), 5),
            "current_peak_per_hz": round(peak, 5),
            # Field in Hz this mode can reach before a channel hits +-3 A.
            "max_field_hz_at_3A": round(float(IMAX / peak) if peak > 0 else 0.0, 2),
            "currents": np.round(v, 6).tolist(),
        })

    return {"audit": audit, "rank_kept": rank,
            "eigenvalues": np.round(lam, 6).tolist(),
            "lambda_chance_level": round(lam_chance, 6),
            "modes": modes, "wP_volume": wP_vol,
            "C_perp_energy_share_pct": round(
                100.0 * float(np.sum(C_perp ** 2))
                / max(float(np.sum(C ** 2)), 1e-12), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()

    P = Progress("Phase 1 + Slepian — hardware-reachable PFC-concentrated modes")
    entries, folds = cohort()
    want = [w.strip() for w in args.subjects.split(",") if w.strip()]
    sel = [e for e in entries
           if (e[0] in want) if want] or [e for e in entries
                                          if folds[e[0]] == args.fold]
    if args.limit:
        sel = sel[:args.limit]
    P.log(f"{len(sel)} scans  (fold={args.fold if not want else 'explicit'})")

    per_subject, mode_rows, audit_rows = {}, [], []
    for e in sel:
        s = load_subject(e)
        maps, _ = coil_maps_for(s["affine"], s["shape"])
        basis = basis_with_a00(s["affine"], s["shape"])
        res = analyse(s, maps, basis, P)
        a = res["audit"]
        P.log(f"  {s['name']:12s} rank {res['rank_kept']:2d}  "
              f"C_perp/C {res['C_perp_energy_share_pct']:6.2f}%  "
              f"chance {res['lambda_chance_level']:.3f}  "
              f"lambda {np.round(res['eigenvalues'][:4], 3)}  "
              f"recon {a['reconstruction_rel']:.1e}  orth {a['sh_orthogonality_rel']:.1e}")
        per_subject[s["name"]] = {
            "fold": folds[s["name"]], "audit": a, "rank_kept": res["rank_kept"],
            "eigenvalues": res["eigenvalues"], "modes": res["modes"],
            "C_perp_energy_share_pct": res["C_perp_energy_share_pct"]}
        audit_rows.append({"subject": s["name"], "fold": folds[s["name"]],
                           "rank_kept": res["rank_kept"],
                           "C_perp_energy_share_pct": res["C_perp_energy_share_pct"],
                           **{k: (round(v, 12) if isinstance(v, float) else v)
                              for k, v in a.items()}})
        for m in res["modes"]:
            mode_rows.append({"subject": s["name"], "fold": folds[s["name"]],
                              **{k: v for k, v in m.items() if k != "currents"}})

        if s["name"] == sel[0][0]:
            triplane_png(OUT_DIR / "figures" / f"wP_taper_{s['name']}.png",
                         res["wP_volume"], [(s["pfc"], "white")],
                         title=f"{s['name']}  tapered w_P ({TAPER_MM} mm) with "
                               f"hard PFC outline", clim=(0, 1), cmap="jet")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in (("slepian_summary.csv", mode_rows),
                       ("projection_audit.csv", audit_rows)):
        if rows:
            keys = sorted({k for r in rows for k in r})
            with open(OUT_DIR / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader(); w.writerows(rows)

    np.savez_compressed(
        OUT_DIR / "slepian_modes.npz",
        **{f"{n}__currents": np.array([m["currents"] for m in d["modes"]])
           for n, d in per_subject.items()},
        **{f"{n}__lambda": np.array(d["eigenvalues"]) for n, d in per_subject.items()})

    # Cross-subject stability of the current vectors: the modes live on
    # different grids, but a 12-vector of amps is comparable directly, and that
    # is what would actually be shipped.
    names = sorted(per_subject)
    stability = {}
    if len(names) > 1:
        for k in range(3):
            V = []
            for n in names:
                ms = per_subject[n]["modes"]
                if k < len(ms):
                    v = np.array(ms[k]["currents"], dtype=float)
                    nv = np.linalg.norm(v)
                    if nv > 0:
                        V.append(v / nv)
            if len(V) > 1:
                V = np.array(V)
                Gm = np.abs(V @ V.T)
                iu = np.triu_indices(len(V), 1)
                stability[f"mode{k}"] = {
                    "n": len(V),
                    "mean_abs_cos": round(float(Gm[iu].mean()), 4),
                    "min_abs_cos": round(float(Gm[iu].min()), 4)}
                P.log(f"  mode {k} current-vector agreement across scans: "
                      f"mean |cos| {stability[f'mode{k}']['mean_abs_cos']:.3f}  "
                      f"min {stability[f'mode{k}']['min_abs_cos']:.3f}")

    write_json(OUT_DIR / "slepian_per_subject.json",
               {"taper_mm": TAPER_MM, "whiten_rtol": WHITEN_RTOL,
                "n_scans": len(per_subject), "fold": args.fold,
                "current_vector_stability": stability,
                "per_subject": per_subject})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
