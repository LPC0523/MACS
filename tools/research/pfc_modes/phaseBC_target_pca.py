"""Phase B + C/D -- SH9 high-order residual, carried to MNI, then PCA.

    D:\\python.exe tools/research/pfc_modes/phaseBC_target_pca.py

The correction to the previous round: A00 is the centre frequency, not a
high-order capability. Anything claiming to measure "what the harmonics cannot
shape" must therefore project onto **SH9** = A00 + 3 first + 5 second order.
The SH8 numbers describe a control-interface gap (no f0 on this page), not the
array's field-shaping ability.

Per scan, in native physical space:

    b_H = (I - P_SH9) b          the residual no scanner harmonic can remove
    C_H = (I - P_SH9) C          the part of the coil field likewise
    C   = S9 K + C_H             K is 9x12; K[0] is A00, K[1:] the 8 shim terms

`b_H` is then carried into MNI through the Phase A2 transform, and the PCA is
taken over the train fold inside a smoothly tapered PFC weight. The modes are
*targets* -- what field shape is worth cancelling -- and place no restriction on
which currents may be used later.
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (            # noqa: E402
    SH9, OUT_DIR, Progress, basis_with_a00, cohort, coil_maps_for,
    load_subject, sh_matrix, triplane_png, write_json)
from tools.research.pfc_modes.phase1_slepian import TAPER_MM, taper_weight  # noqa: E402
from core.paths import find_atlases                       # noqa: E402

TX_DIR = OUT_DIR / "transforms"
#: MNI working resolution. The residual is a smooth field; 2 mm keeps the whole
#: cohort in memory at once and is far finer than anything the harmonics or the
#: array can shape.
MNI_MM = 2.0
N_MODES = 10


def mni_reference(mni_path, mm):
    """A `mm`-isotropic grid covering the atlas, as (shape, affine)."""
    img = nib.load(str(mni_path))
    zoom = np.sqrt((img.affine[:3, :3] ** 2).sum(axis=0))
    shape = np.maximum(np.round(np.asarray(img.shape[:3]) * zoom / mm).astype(int), 1)
    aff = img.affine.copy()
    aff[:3, :3] = img.affine[:3, :3] @ np.diag(mm / zoom)
    return tuple(int(v) for v in shape), aff


def project_sh9(b0, C, brain, basis):
    """(b_H, C_H, K) over the brain mask, plain least squares on SH9."""
    S = sh_matrix(basis, SH9, brain)
    b = b0[brain].astype(np.float64)
    Cb = C.reshape(-1, C.shape[-1])[brain.ravel(), :].astype(np.float64)
    Q, _ = np.linalg.qr(S)
    b_H = b - Q @ (Q.T @ b)
    C_H = Cb - Q @ (Q.T @ Cb)
    K, *_ = np.linalg.lstsq(S, Cb, rcond=None)      # 9 x 12
    return b_H, C_H, K, S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="",
                    help="suffix for output files, so a rebuilt target does "
                         "not overwrite the one earlier results were computed "
                         "against")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    P = Progress("Phase B + C/D — SH9 residual -> MNI -> target PCA")
    import ants
    from core import ants_io

    mni_path, pfc_path = find_atlases()
    shape_m, aff_m = mni_reference(mni_path, MNI_MM)
    P.log(f"MNI working grid {shape_m} at {MNI_MM} mm")

    ref_img = nib.Nifti1Image(np.zeros(shape_m, np.float32), aff_m)
    tmp = Path(tempfile.mkdtemp(prefix="pfcmodes_"))
    ref_path = tmp / "mni_ref.nii.gz"
    nib.save(ref_img, str(ref_path))
    ref_ants = ants_io.image_read(ref_path)

    # PFC atlas and its taper, defined once in MNI.
    pfc_img = nib.load(str(pfc_path))
    from core.data_context import resample_to_target
    pfc_m = resample_to_target(
        (np.asarray(pfc_img.get_fdata()) > 0.5).astype(np.float32),
        pfc_img.affine, aff_m, shape_m, order=0) > 0.5
    wP_m = taper_weight(pfc_m, aff_m, TAPER_MM)
    P.log(f"MNI PFC {int(pfc_m.sum()):,} voxels, tapered weight sum "
          f"{wP_m.sum():.0f}")

    entries, folds = cohort()
    sel = [e for e in entries if folds[e[0]] == args.fold]
    if args.limit:
        sel = sel[:args.limit]
    P.log(f"{len(sel)} scans in fold '{args.fold}'")

    fields, brains, rows, kmats = [], [], [], {}
    for name, hz, brain_p, pfc_p in sel:
        tx = TX_DIR / name / "native_to_mni_0GenericAffine.mat"
        if not tx.is_file():
            P.log(f"  {name:12s} no transform, skipped"); continue
        s = load_subject((name, hz, brain_p, pfc_p))
        maps, _ = coil_maps_for(s["affine"], s["shape"])
        basis = basis_with_a00(s["affine"], s["shape"])
        b_H, C_H, K, S = project_sh9(s["b0"], maps, s["brain"], basis)

        b = s["b0"][s["brain"]]
        Cb = maps.reshape(-1, maps.shape[-1])[s["brain"].ravel(), :]
        rows.append({
            "subject": name, "fold": folds[name],
            "b_rms": round(float(np.sqrt(np.mean(b ** 2))), 3),
            "b_H_rms": round(float(np.sqrt(np.mean(b_H ** 2))), 3),
            "b_H_share_pct": round(100.0 * float(np.sum(b_H ** 2) / np.sum(b ** 2)), 3),
            "C_H_share_pct": round(100.0 * float(np.sum(C_H ** 2) / np.sum(Cb ** 2)), 4),
            "K_A00_absmax_hz_per_a": round(float(np.max(np.abs(K[0]))), 3),
            "K_SH_absmax": round(float(np.max(np.abs(K[1:]))), 3),
            "pfc_b_H_rms": round(float(np.sqrt(np.mean(
                (s["b0"] - 0)[s["pfc"]] * 0 + 0))), 3) if False else None,
        })
        kmats[name] = K

        # Carry b_H into MNI. Written as a volume first: zero outside the
        # brain, with the brain mask carried alongside so voxels that no scan
        # covers can be excluded rather than counted as zero field.
        vol = np.zeros(s["shape"], np.float32)
        vol[s["brain"]] = b_H
        f_native = tmp / f"{name}_bH.nii.gz"
        m_native = tmp / f"{name}_brain.nii.gz"
        nib.save(nib.Nifti1Image(vol, s["affine"]), str(f_native))
        nib.save(nib.Nifti1Image(s["brain"].astype(np.float32), s["affine"]),
                 str(m_native))

        w_f = ants.apply_transforms(fixed=ref_ants,
                                    moving=ants_io.image_read(f_native),
                                    transformlist=[str(tx)],
                                    interpolator="linear")
        w_m = ants.apply_transforms(fixed=ref_ants,
                                    moving=ants_io.image_read(m_native),
                                    transformlist=[str(tx)],
                                    interpolator="nearestNeighbor")
        fields.append(np.asarray(w_f.numpy(), np.float64))
        brains.append(np.asarray(w_m.numpy()) > 0.5)
        P.log(f"  {name:12s} b_H {rows[-1]['b_H_rms']:7.2f} Hz "
              f"({rows[-1]['b_H_share_pct']:5.2f}% of b)  "
              f"C_H {rows[-1]['C_H_share_pct']:6.3f}%  -> MNI")

    if len(fields) < 3:
        raise SystemExit("too few scans carried to MNI")

    F = np.stack(fields, 0)
    B = np.stack(brains, 0)
    covered = B.all(axis=0)
    region = covered & (wP_m > 0)
    P.log(f"voxels covered by every scan: {int(covered.sum()):,}; "
          f"inside the tapered PFC: {int(region.sum()):,}")
    if region.sum() < 500:
        raise SystemExit("shared PFC region too small in MNI")

    w = wP_m[region]
    X = F[:, region]                       # (n_scans, n_voxels)
    sw = np.sqrt(w)

    # Mean mode kept explicitly: it is the part of the residual every subject
    # shares, and discarding it silently would hide the largest correctable
    # component.
    mean_mode = X.mean(axis=0)
    Xc = X - mean_mode

    def pca(M, tag):
        A = M * sw[None, :]
        U, sv, Vt = np.linalg.svd(A, full_matrices=False)
        var = sv ** 2 / max(float(np.sum(sv ** 2)), 1e-12)
        modes = Vt / np.maximum(sw[None, :], 1e-12)   # back to field units
        return {"tag": tag, "sv": sv, "var": var, "modes": modes, "coef": U * sv}

    P.log("PCA over the tapered PFC, amplitude preserved")
    raw = pca(X, "with_mean")
    cen = pca(Xc, "mean_removed")

    mean_w = float(np.sum(w * mean_mode ** 2) / np.sum(w))
    tot_w = float(np.sum(w[None, :] * X ** 2) / (X.shape[0] * np.sum(w)))
    P.log(f"  mean mode carries {100 * mean_w / max(tot_w, 1e-12):.1f}% of the "
          f"weighted residual energy")

    mode_rows = []
    for k in range(min(N_MODES, cen["modes"].shape[0])):
        for src, name_ in ((raw, "with_mean"), (cen, "mean_removed")):
            if k >= src["modes"].shape[0]:
                continue
            m = np.zeros(shape_m)
            m[region] = src["modes"][k]
            e_p = float(np.sum(w * src["modes"][k] ** 2))
            rest = covered & (wP_m <= 0)
            mode_rows.append({
                "basis": name_, "mode": k,
                "explained_variance": round(float(src["var"][k]), 5),
                "cumulative_variance": round(float(np.sum(src["var"][:k + 1])), 5),
                "pfc_weighted_energy": round(e_p, 6),
                "coef_mean_abs": round(float(np.mean(np.abs(src["coef"][:, k]))), 4),
                "coef_sd": round(float(np.std(src["coef"][:, k])), 4),
                # Sign consistency across scans: a mode every subject expresses
                # the same way is one worth targeting.
                "coef_sign_agreement": round(float(
                    max((src["coef"][:, k] > 0).mean(),
                        (src["coef"][:, k] < 0).mean())), 4),
            })
    for r in [r for r in mode_rows if r["basis"] == "mean_removed"][:6]:
        P.log(f"  mode {r['mode']}: var {r['explained_variance']:.3f}  "
              f"cum {r['cumulative_variance']:.3f}  "
              f"sign agreement {r['coef_sign_agreement']:.2f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DIR / f"target_pca_modes{tag}.npz",
        region=region, wP=wP_m, covered=covered, affine=aff_m,
        shape=np.array(shape_m), mean_mode=mean_mode,
        modes_with_mean=raw["modes"][:N_MODES],
        modes_mean_removed=cen["modes"][:N_MODES],
        var_with_mean=raw["var"][:N_MODES], var_mean_removed=cen["var"][:N_MODES],
        coef_with_mean=raw["coef"][:, :N_MODES],
        coef_mean_removed=cen["coef"][:, :N_MODES],
        subjects=np.array([r["subject"] for r in rows]))

    for nm, arr in (("mean", mean_mode), ("mode0", cen["modes"][0]),
                     ("mode1", cen["modes"][1])):
        v = np.zeros(shape_m); v[region] = arr
        triplane_png(OUT_DIR / "figures" / f"target_{nm}{tag}.png", v,
                     [(pfc_m, "white")], title=f"MNI target {nm}{tag}",
                     clim=(-np.percentile(np.abs(arr), 98),
                           np.percentile(np.abs(arr), 98)))

    for fn, data in ((f"sh9_residual_per_subject{tag}.csv", rows),
                     (f"target_pca_summary{tag}.csv", mode_rows)):
        keys = sorted({k for r in data for k in r})
        with open(OUT_DIR / fn, "w", newline="", encoding="utf-8") as f:
            w_ = csv.DictWriter(f, fieldnames=keys)
            w_.writeheader(); w_.writerows(data)
    write_json(OUT_DIR / f"target_pca_detail{tag}.json", {
        "fold": args.fold, "n_scans": len(rows), "mni_mm": MNI_MM,
        "taper_mm": TAPER_MM, "sh9": SH9,
        "region_voxels": int(region.sum()),
        "mean_mode_energy_share": round(mean_w / max(tot_w, 1e-12), 5),
        "per_subject": rows, "modes": mode_rows})
    P.log(f"written -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
