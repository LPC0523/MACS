"""Choose the current ridge from the batch, not from one subject.

    D:\\python.exe tests\\run_ridge_sweep.py [--coil "PFC 12ch"] [-n 12] [--roi pfc]

Without a ridge the solve may use large, partly cancelling currents whenever two
channels overlap -- the objective cannot distinguish that from a small solution,
but the amplifier and its heat can. How much current the last Hz of homogeneity
costs is a property of the subject and the array, so the setting is measured
rather than guessed, and measured on every subject available rather than on a
convenient one.

For each subject the ladder is swept and the chosen ridge is the *largest*
(smallest currents) whose ROI RMSE stays within `--tol` Hz of the best any ridge
achieved. The default worth freezing is then the one that most subjects agree
on and that no subject is badly served by.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import nibabel as nib

warnings.filterwarnings("ignore")

from core.coil_registry import get_coil
from core.pipeline import _load_coil_maps
from core.shim_optimize import RIDGE_LADDER, RIDGE_TOL_HZ, sweep_ridge

_ROOTS = [_HERE.parent.parent.parent / "2nd_order_SH",
          _HERE.parent.parent / "test_dic"]


def find_subjects(limit: int, need_pfc: bool):
    found, seen = [], set()
    for root in _ROOTS:
        if not root.is_dir():
            continue
        for hz in sorted(root.rglob("*_deltaB0_hertz.nii.gz")):
            name = hz.name[: -len("_deltaB0_hertz.nii.gz")]
            if name in seen:
                continue
            brain = hz.parent / f"{name}_bet_mask.nii.gz"
            if not brain.is_file():
                continue
            pfc = next((p for p in (hz.parent / f"{name}_pfc_mask.nii.gz",
                                    hz.parent / "roi_cache" / "pfc_native.nii.gz")
                        if p.is_file()), None)
            if need_pfc and pfc is None:
                continue
            seen.add(name)
            found.append((name, hz, brain, pfc))
            if limit and len(found) >= limit:
                return found
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coil", default="PFC 12ch")
    ap.add_argument("-n", "--n-subjects", type=int, default=12)
    ap.add_argument("--roi", choices=["whole_brain", "pfc"], default="pfc")
    ap.add_argument("--tol", type=float, default=RIDGE_TOL_HZ)
    ap.add_argument("--lam", type=float, default=0.5, help="λ_protect")
    args = ap.parse_args()

    coil = get_coil(args.coil)
    if coil is None:
        raise SystemExit(f"找不到线圈配置：{args.coil}")
    imax = coil.current_limit_a

    subjects = find_subjects(args.n_subjects, need_pfc=(args.roi == "pfc"))
    if not subjects:
        raise SystemExit("没有找到可用被试")

    print(f"线圈 {coil.name}（±{imax:g} A）　ROI={args.roi}　"
          f"λ_protect={args.lam}　容差 {args.tol:g} Hz　被试 {len(subjects)} 名")
    print(f"ladder: {', '.join(f'{x:g}' for x in RIDGE_LADDER)}\n", flush=True)

    hdr = (f"{'被试':<12}{'λ 选定':>9}{'无ridge':>9}{'最优':>8}{'退化':>8}"
           f"{'电流RMS':>16}{'最大电流':>16}{'饱和':>12}")
    print(hdr)
    print("-" * 100)

    rows = []
    for name, hz, brain_p, pfc_p in subjects:
        img = nib.load(str(hz))
        b0 = np.asanyarray(img.dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(brain_p)).dataobj) > 0.5
        if brain.sum() < 20000:
            continue
        if args.roi == "pfc":
            roi = np.asanyarray(nib.load(str(pfc_p)).dataobj) > 0.5
            if roi.shape != brain.shape:
                continue
            roi &= brain
            if roi.sum() < 500:
                continue
        else:
            roi = brain
        try:
            maps, _ = _load_coil_maps(coil, img.affine, b0.shape[:3], lambda *a: None)
        except Exception:
            continue

        sw = sweep_ridge(b0, roi.astype(np.float32), maps,
                         tol_hz=args.tol, log_cb=None,
                         lambda_penalty=args.lam, current_limit=imax,
                         outside_mask=brain)
        c, nr = sw["chosen"], sw["no_ridge"]
        rows.append((name, c["lambda_ridge"], nr, c, sw))
        print(f"{name:<12}{c['lambda_ridge']:>9g}{nr['roi_rmse']:>9.2f}"
              f"{sw['best_rmse']:>8.2f}{sw['degradation_hz']:>+8.2f}"
              f"{nr['current_rms']:>7.2f}→{c['current_rms']:<8.2f}"
              f"{nr['current_max']:>7.2f}→{c['current_max']:<8.2f}"
              f"{nr['at_limit']:>5}→{c['at_limit']:<6}", flush=True)

    if not rows:
        raise SystemExit("没有任何被试跑通")

    n = len(rows)
    votes = Counter(r[1] for r in rows)
    print(f"\n## 各被试选出的 λ_ridge 分布\n")
    for lam, cnt in sorted(votes.items()):
        print(f"  λ={lam:<8g} {cnt}/{n} 名被试")

    # A default has to be safe for everyone, not just typical. Report the worst
    # degradation each rung would inflict, then take the largest rung that stays
    # inside the tolerance -- and say so plainly when none does, rather than
    # falling back to a value that was never measured.
    worst_by_lam = []
    for lam in RIDGE_LADDER:
        worst = max(
            next(t for t in sw["trials"] if t["lambda_ridge"] == lam)["roi_rmse"]
            - sw["best_rmse"]
            for _, _, _, _, sw in rows)
        worst_by_lam.append((lam, worst))

    print("\n## 每个 λ 对最差被试的退化\n")
    for lam, worst in worst_by_lam:
        mark = "✓" if worst <= args.tol else " "
        print(f"  {mark} λ={lam:<8g} 最差退化 {worst:+.2f} Hz")

    safe = [(lam, w) for lam, w in worst_by_lam if w <= args.tol]
    rec = max(safe, key=lambda x: x[0]) if safe else None

    def mean_at(lam, key):
        vals = []
        for _, _, _, _, sw in rows:
            t = next(t for t in sw["trials"] if t["lambda_ridge"] == lam)
            vals.append(t[key])
        return float(np.mean(vals))

    print(f"\n## 结论\n")
    if rec is None:
        best_lam, best_worst = min(worst_by_lam, key=lambda x: x[1])
        print(f"**没有任何 λ 能让全部 {n} 名被试的退化都 ≤ {args.tol:g} Hz。**")
        print(f"最接近的是 λ={best_lam:g}（最差退化 {best_worst:+.2f} Hz）。")
        print("ROI RMSE 不是被最小化的量（目标里还有 rest 保护项），"
              "所以它对 λ 不单调，个别被试会在某个 λ 上跳变。")
        print("冻结默认值前需要先决定：放宽容差、按多数被试选、还是保持 λ_ridge=0。")
        return 0
    print(f"对全部 {n} 名被试都满足「退化 ≤ {args.tol:g} Hz」的最大 λ_ridge = "
          f"**{rec[0]:g}**（最差退化 {rec[1]:+.2f} Hz）")
    print(f"  电流 RMS  {mean_at(0.0, 'current_rms'):.2f} → "
          f"{mean_at(rec[0], 'current_rms'):.2f} A")
    print(f"  最大电流  {mean_at(0.0, 'current_max'):.2f} → "
          f"{mean_at(rec[0], 'current_max'):.2f} A")
    print(f"  饱和通道  {mean_at(0.0, 'at_limit'):.1f} → "
          f"{mean_at(rec[0], 'at_limit'):.1f}")
    print(f"\n把 core/shim_optimize.py 的 DEFAULT_RIDGE 设为 {rec[0]:g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
