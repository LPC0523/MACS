"""Slice-by-slice (dynamic) shimming.

Static shimming finds one current vector for the whole head. It is a
compromise: a coil setting that flattens the frontal lobe is not the setting
that flattens the temporal lobes, and one vector has to serve both. Dynamic
shimming updates the currents as the sequence steps through slices, so each
slice gets the setting that suits it.

Computationally that is the same problem solved N times over smaller regions.
This module therefore contains no solver of its own: `_optimize_single_slice`
reshapes one slice into the (nx, ny, 1) volume that `optimize_currents` already
takes, and calls it. Two implementations of the objective would be two things
to keep in step, and any drift between them would show up as "the static and
dynamic answers disagree" with no way to tell which is right.

Slices whose brain mask is nearly empty are skipped rather than solved. With a
handful of voxels the least-squares problem is under-determined, and the
solver happily returns large currents that fit the noise -- currents that then
get written to the amplifier for a slice containing almost no brain.
"""

from __future__ import annotations

import numpy as np

from core.shim_optimize import optimize_currents, evaluate_masks

__all__ = ["optimize_per_slice", "MIN_SLICE_VOXELS"]

#: Below this many in-mask voxels a slice is left at zero current. Twelve
#: channels need more than a dozen voxels before the fit means anything.
MIN_SLICE_VOXELS = 10


def _optimize_single_slice(b0_slice, mask_slice, coil_slice,
                           lambda_penalty, current_limit, outside_slice=None):
    """One slice through the existing solver.

    `optimize_currents` works on volumes, so the slice is presented as a
    volume of depth one. Nothing about the objective, the bounds or the solver
    changes -- this is a reshape, not a second implementation.
    """
    b0_v = b0_slice[:, :, None]
    mask_v = mask_slice[:, :, None].astype(np.float32)
    coil_v = coil_slice[:, :, None, :]
    out_v = None if outside_slice is None else outside_slice[:, :, None].astype(np.float32)
    return optimize_currents(b0_v, mask_v, coil_v,
                             lambda_penalty=lambda_penalty,
                             current_limit=current_limit,
                             log_cb=None, outside_mask=out_v)


def optimize_per_slice(b0: np.ndarray, brain_mask: np.ndarray,
                       coil_maps: np.ndarray,
                       lambda_penalty: float = 0.5,
                       current_limit: float = 3.0,
                       axis: int = 2,
                       log_cb=None) -> dict:
    """Solve each slice independently along `axis` (Z by default).

    Returns
    -------
    currents   : (n_slices, n_channels) amps, zero rows for skipped slices
    residual   : (nx, ny, nz) field after each slice's own currents
    rmse_before / rmse_after : (n_slices,) in-mask RMSE per slice, NaN when skipped
    voxels     : (n_slices,) in-mask voxel count
    solved     : (n_slices,) bool
    """
    log = log_cb or (lambda *_: None)

    if axis != 2:
        b0 = np.moveaxis(b0, axis, 2)
        brain_mask = np.moveaxis(brain_mask, axis, 2)
        coil_maps = np.moveaxis(coil_maps, axis, 2)

    n_slices = b0.shape[2]
    n_ch = coil_maps.shape[-1]

    currents = np.zeros((n_slices, n_ch), dtype=np.float64)
    rmse_before = np.full(n_slices, np.nan)
    rmse_after = np.full(n_slices, np.nan)
    voxels = np.zeros(n_slices, dtype=np.int64)
    solved = np.zeros(n_slices, dtype=bool)
    at_limit = np.zeros(n_slices, dtype=np.int64)
    residual = b0.astype(np.float32).copy()

    mask3 = brain_mask > 0.5
    log(f"逐层优化：{n_slices} 层 × {n_ch} 通道")

    report_every = max(1, n_slices // 10)
    for z in range(n_slices):
        m = mask3[:, :, z]
        voxels[z] = int(m.sum())
        if voxels[z] < MIN_SLICE_VOXELS:
            continue        # currents stay zero; residual stays as measured

        # Per slice the ROI is the whole in-slice brain, so `rest` is empty and
        # both rest terms are skipped -- exactly as for whole-brain shimming.
        sol = _optimize_single_slice(b0[:, :, z], m, coil_maps[:, :, z, :],
                                     lambda_penalty, current_limit,
                                     outside_slice=m)
        currents[z] = sol["currents"]
        at_limit[z] = sol["channels_at_limit"]
        solved[z] = True

        res_z = sol["residual"][:, :, 0]
        residual[:, :, z] = res_z
        rmse_before[z] = float(np.sqrt(np.mean(b0[:, :, z][m] ** 2)))
        rmse_after[z] = float(np.sqrt(np.mean(res_z[m] ** 2)))

        if z % report_every == 0 or z == n_slices - 1:
            log(f"  层 {z + 1}/{n_slices}  "
                f"{voxels[z]:,} 体素  RMSE {rmse_before[z]:.1f} → {rmse_after[z]:.1f} Hz")

    n_solved = int(solved.sum())
    n_skipped = n_slices - n_solved
    if n_skipped:
        log(f"  跳过 {n_skipped} 层（脑内体素少于 {MIN_SLICE_VOXELS}，电流置 0）")

    if axis != 2:
        residual = np.moveaxis(residual, 2, axis)

    # Whole-brain figures, so dynamic and static can be compared on one number.
    mask_orig = brain_mask > 0.5 if axis == 2 else np.moveaxis(mask3, 2, axis)
    b0_orig = b0 if axis == 2 else np.moveaxis(b0, 2, axis)
    stats = evaluate_masks(b0_orig, residual, mask_orig.astype(np.float32))

    solved_after = rmse_after[solved]
    log(f"完成。已优化 {n_solved} 层，逐层 RMSE 均值 "
        f"{np.mean(solved_after):.1f} Hz（全脑 RMSE "
        f"{stats['roi_rmse_before']:.1f} → {stats['roi_rmse_after']:.1f} Hz）")

    return {
        "currents": currents,
        "residual": residual,
        "rmse_before": rmse_before,
        "rmse_after": rmse_after,
        "voxels": voxels,
        "solved": solved,
        "channels_at_limit": at_limit,
        "stats": stats,
        "n_slices": n_slices,
        "n_channels": n_ch,
        "axis": axis,
        "mean_rmse_after": float(np.mean(solved_after)) if n_solved else float("nan"),
        "mean_rmse_before": (float(np.mean(rmse_before[solved])) if n_solved
                             else float("nan")),
    }


def to_csv(currents: np.ndarray, path, channel_names=None) -> str:
    """Current table as CSV: one row per slice, one column per channel.

    Written for the amplifier's own import, so the header names the columns and
    nothing else appears -- no units row, no comment lines that a fixed-format
    parser would have to be told to skip.
    """
    import csv
    from pathlib import Path

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n_slices, n_ch = currents.shape
    names = list(channel_names or []) or [f"ch{i + 1}" for i in range(n_ch)]

    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["slice"] + names)
        for z in range(n_slices):
            wr.writerow([z + 1] + [f"{v:.6f}" for v in currents[z]])
    return str(p)
