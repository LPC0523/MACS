"""What counts as "outside the ROI", and why it must not be the field of view.

    D:\\python.exe tests\\test_rest_region.py

Three properties, each of which was wrong or unverified at some point:

  1  whole-brain ROI leaves `rest` empty, and both rest terms are skipped
     rather than evaluated -- `np.mean` of an empty slice is NaN, which
     propagates into the objective and drives the solver to near-zero current
     while reporting success.

  2  a PFC ROI leaves `rest` non-empty and equal to `brain & ~roi` exactly --
     not "every voxel outside the ROI", which is mostly air.

  3  padding the volume with air does not change the solution. This is the
     reason the mask matters: with `rest` defined as everything outside the
     ROI, the penalty is dominated by air, and re-cropping the same head gives
     different currents. Nothing about the anatomy or the coils changed, so
     nothing about the answer should.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np

warnings.filterwarnings("ignore")

from core.shim_optimize import optimize_currents

FAILURES = []


def check(name: str, ok: bool, detail: str = ""):
    print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def make_case(seed: int = 0, pad: int = 0):
    """A small synthetic head, optionally surrounded by extra air."""
    rng = np.random.default_rng(seed)
    nx = ny = 24
    nz = 18
    brain = np.zeros((nx, ny, nz), bool)
    brain[5:19, 5:19, 4:14] = True
    roi = np.zeros_like(brain)
    roi[5:11, 5:19, 4:14] = True          # anterior third
    roi &= brain

    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx),
                             indexing="ij")
    field = (0.9 * xx.T + 0.4 * (yy.T ** 2) / ny).astype(np.float64)
    b0 = field - field.mean() + rng.normal(0, 3, (nx, ny, nz))
    maps = np.stack([rng.normal(0, 6, (nx, ny, nz)) + 0.5 * xx.T
                     for _ in range(6)], axis=-1)

    if pad:
        def grow(a, fill=0):
            out = np.full([s + 2 * pad for s in a.shape[:3]] + list(a.shape[3:]),
                          fill, dtype=a.dtype)
            out[pad:-pad, pad:-pad, pad:-pad] = a
            return out
        b0, brain, roi, maps = grow(b0), grow(brain), grow(roi), grow(maps)
    return b0, brain, roi, maps


def main() -> int:
    b0, brain, roi, maps = make_case()

    print("1. 全脑 ROI：rest 为空，两个 rest 项跳过")
    r = optimize_currents(b0, brain.astype(np.float32), maps,
                          lambda_penalty=0.5, current_limit=3.0,
                          outside_mask=brain, log_cb=None)
    check("rest 体素数为 0", r["rest_voxels"] == 0, f"rest={r['rest_voxels']}")
    check("电流有限且非 NaN", np.all(np.isfinite(r["currents"])))
    check("电流没有被 NaN 压成 0",
          float(np.max(np.abs(r["currents"]))) > 1e-3,
          f"max|I|={np.max(np.abs(r['currents'])):.3f} A")
    check("统计量非 NaN", np.isfinite(r["stats"]["roi_rmse_after"]),
          f"RMSE={r['stats']['roi_rmse_after']:.2f}")

    print("\n2. PFC ROI：rest 恰为 brain & ~roi")
    r2 = optimize_currents(b0, roi.astype(np.float32), maps,
                           lambda_penalty=0.5, current_limit=3.0,
                           outside_mask=brain, log_cb=None)
    expect = int((brain & ~roi).sum())
    check("rest == brain & ~roi", r2["rest_voxels"] == expect,
          f"{r2['rest_voxels']} vs {expect}")
    check("rest 非空", r2["rest_voxels"] > 0)
    # The legacy definition would have swept in every air voxel as well.
    legacy = int((~roi).sum())
    check("rest 明显小于「ROI 之外的全部体素」", r2["rest_voxels"] < legacy,
          f"{r2['rest_voxels']} < {legacy}")

    print("\n3. 改变 FOV：brain-only 的解不受影响")
    b0p, brainp, roip, mapsp = make_case(pad=6)
    r3 = optimize_currents(b0p, roip.astype(np.float32), mapsp,
                           lambda_penalty=0.5, current_limit=3.0,
                           outside_mask=brainp, log_cb=None)
    d_brain = float(np.max(np.abs(r2["currents"] - r3["currents"])))
    check("padding 后电流不变（brain-only）", d_brain < 1e-6,
          f"max|ΔI|={d_brain:.2e} A")

    # And the contrast: with the legacy definition it does change.
    r4 = optimize_currents(b0, roi.astype(np.float32), maps,
                           lambda_penalty=0.5, current_limit=3.0,
                           outside_mask=None, log_cb=None)
    r5 = optimize_currents(b0p, roip.astype(np.float32), mapsp,
                           lambda_penalty=0.5, current_limit=3.0,
                           outside_mask=None, log_cb=None)
    d_legacy = float(np.max(np.abs(r4["currents"] - r5["currents"])))
    check("对照：旧定义（含脑外）确实受 FOV 影响", d_legacy > 1e-4,
          f"max|ΔI|={d_legacy:.3f} A")

    print()
    if FAILURES:
        print(f"FAILED: {', '.join(FAILURES)}")
        return 1
    print("all passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
