"""PFC-weighted joint shimming across subjects: is lambda_rest=0.5 a good default?

    D:\\python.exe tests\\run_pfc_weighted_batch.py [--coil "PFC 12ch"] [-n 8]

Weighting the PFC buys homogeneity there by spending it elsewhere. One subject
cannot say whether the trade is worth taking -- the PFC's share of the brain,
how bad it starts, and how well the array happens to cover it all vary. This
runs the same four settings on every subject that already has a field map, a
brain mask and a PFC mask on disk, so the answer rests on the spread rather
than on one favourable case.

Four settings per subject:

    lambda_rest = 0      whole-brain equal weight (what the app shipped)
    lambda_rest = 0.05   PFC dominant
    lambda_rest = 0.2
    lambda_rest = 0.5    the proposed default

All figures are demeaned spread (std), matching the UI -- the constant
component is the centre frequency and is not something this solver moves.

"Net" is PFC gain minus whole-brain cost, both in Hz. Positive means the trade
returned more than it consumed; it is the only column that answers the question
directly.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import nibabel as nib

warnings.filterwarnings("ignore")

from core.adaptive_joint import solve_adaptive_joint
from core.coil_registry import get_coil
from core.hybrid_shim import SH12, build_sh_basis
from core.pipeline import _load_coil_maps

UNIFORM = {"base": 1.0, "alpha": 0.0, "top_fraction": 0.0}
LAMBDAS = [0.0, 0.05, 0.2, 0.5]

#: Search roots, most likely first.
_ROOTS = [_HERE.parent.parent.parent / "2nd_order_SH",
          _HERE.parent.parent / "test_dic"]


def find_subjects(limit: int) -> list[tuple[str, Path, Path, Path]]:
    """(name, b0, brain mask, pfc mask) for every subject that has all three."""
    found: list[tuple[str, Path, Path, Path]] = []
    seen: set[str] = set()
    for root in _ROOTS:
        if not root.is_dir():
            continue
        for hz in sorted(root.rglob("*_deltaB0_hertz.nii.gz")):
            name = hz.name[: -len("_deltaB0_hertz.nii.gz")]
            if name in seen:
                continue
            brain = hz.parent / f"{name}_bet_mask.nii.gz"
            pfc_candidates = [hz.parent / f"{name}_pfc_mask.nii.gz",
                              hz.parent / "roi_cache" / "pfc_native.nii.gz"]
            pfc = next((p for p in pfc_candidates if p.is_file()), None)
            if brain.is_file() and pfc is not None:
                seen.add(name)
                found.append((name, hz, brain, pfc))
            if len(found) >= limit:
                return found
    return found


def run_one(b0, brain, pfc, basis_list, maps, imax, lam):
    kw = dict(imax=imax, lambda_pfc_low=0.0, log_cb=lambda *a: None)
    if lam > 0:
        r = solve_adaptive_joint(b0, brain, basis_list, maps,
                                 pfc_mask=pfc, lambda_rest=lam, **kw)
    else:
        r = solve_adaptive_joint(b0, brain, basis_list, maps,
                                 weight_kwargs=UNIFORM, **kw)
    v = r["residual"]
    rest = brain & ~pfc
    return {
        "pfc_before": float(np.std(b0[pfc])),
        "pfc_after": float(np.std(v[pfc])),
        "whole_before": float(np.std(b0[brain])),
        "whole_after": float(np.std(v[brain])),
        "rest_before": float(np.std(b0[rest])),
        "rest_after": float(np.std(v[rest])),
        "n_sat": r["channels_at_limit"],
        "mean_i": float(np.mean(np.abs(r["coil_currents"]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coil", default="PFC 12ch")
    ap.add_argument("-n", "--n-subjects", type=int, default=8)
    args = ap.parse_args()

    coil = get_coil(args.coil)
    if coil is None:
        raise SystemExit(f"找不到线圈配置：{args.coil}")
    imax = coil.current_limit_a

    subjects = find_subjects(args.n_subjects)
    if not subjects:
        raise SystemExit("没有找到同时具备 deltaB0 / bet_mask / pfc_mask 的被试")
    print(f"线圈 {coil.name} ({coil.channel_count}ch, ±{imax:g} A)　"
          f"被试 {len(subjects)} 名\n", flush=True)

    all_rows: dict[str, dict[float, dict]] = {}
    meta: dict[str, tuple[int, float]] = {}

    for name, hz, brain_p, pfc_p in subjects:
        img = nib.load(str(hz))
        b0 = np.asanyarray(img.dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(brain_p)).dataobj) > 0.5
        pfc = np.asanyarray(nib.load(str(pfc_p)).dataobj) > 0.5
        if pfc.shape != brain.shape:
            print(f"[跳过] {name}：PFC mask 尺寸 {pfc.shape} 与脑掩膜 "
                  f"{brain.shape} 不一致", flush=True)
            continue
        pfc = pfc & brain
        if pfc.sum() < 500 or brain.sum() < 20000:
            print(f"[跳过] {name}：PFC {int(pfc.sum())} / 脑 {int(brain.sum())} 体素过少",
                  flush=True)
            continue

        try:
            maps, _ = _load_coil_maps(coil, img.affine, b0.shape[:3],
                                      lambda *a: None)
        except Exception as e:
            print(f"[跳过] {name}：线圈重采样失败 {type(e).__name__}", flush=True)
            continue

        basis = build_sh_basis(img.affine, b0.shape[:3])
        basis_list = [basis[n] for n in SH12]
        meta[name] = (int(pfc.sum()), 100.0 * pfc.sum() / brain.sum())
        all_rows[name] = {lam: run_one(b0, brain, pfc, basis_list, maps, imax, lam)
                          for lam in LAMBDAS}
        r0, r5 = all_rows[name][0.0], all_rows[name][0.5]
        print(f"  {name:<12} PFC {r0['pfc_before']:.0f}Hz "
              f"({meta[name][1]:.1f}%)　λ=0: PFC {r0['pfc_after']:.0f} 全脑 "
              f"{r0['whole_after']:.0f}　λ=0.5: PFC {r5['pfc_after']:.0f} 全脑 "
              f"{r5['whole_after']:.0f}", flush=True)

    if not all_rows:
        raise SystemExit("没有任何被试跑通")

    # ── per-subject matrix ────────────────────────────────────────────
    print("\n## 结果矩阵\n")
    print("| 被试 | PFC 占比 | λ | PFC 前 | PFC 后 | 全脑前 | 全脑后 | 非PFC 前 "
          "| 非PFC 后 | PFC 改善 | 全脑退化 | 净收益 | 饱和 | \\|I\\| |")
    print("|" + "---|" * 14)
    for name, per_lam in all_rows.items():
        for lam in LAMBDAS:
            r = per_lam[lam]
            gain = r["pfc_before"] - r["pfc_after"]
            cost = r["whole_after"] - r["whole_before"]
            print(f"| {name} | {meta[name][1]:.1f}% | {lam:.2f} "
                  f"| {r['pfc_before']:.1f} | {r['pfc_after']:.1f} "
                  f"| {r['whole_before']:.1f} | {r['whole_after']:.1f} "
                  f"| {r['rest_before']:.1f} | {r['rest_after']:.1f} "
                  f"| {gain:+.1f} | {cost:+.1f} | {gain - cost:+.1f} "
                  f"| {r['n_sat']}/{coil.channel_count} | {r['mean_i']:.2f} |")

    # ── summary per lambda ────────────────────────────────────────────
    print("\n## 汇总（每个 λ 跨被试平均）\n")
    print("| λ | 平均 PFC 改善 | 平均全脑退化 | 平均净收益 | 净收益为正 | 兑换率 |")
    print("|---|---|---|---|---|---|")
    summary = {}
    for lam in LAMBDAS:
        gains = [all_rows[n][lam]["pfc_before"] - all_rows[n][lam]["pfc_after"]
                 for n in all_rows]
        costs = [all_rows[n][lam]["whole_after"] - all_rows[n][lam]["whole_before"]
                 for n in all_rows]
        nets = [g - c for g, c in zip(gains, costs)]
        pos = sum(1 for x in nets if x > 0)
        g, c = float(np.mean(gains)), float(np.mean(costs))
        rate = f"{c / g:.1f} : 1" if g > 0.05 else "—"
        summary[lam] = (g, c, float(np.mean(nets)), pos, len(nets))
        print(f"| {lam:.2f} | {g:+.1f} Hz | {c:+.1f} Hz | {np.mean(nets):+.1f} Hz "
              f"| {pos}/{len(nets)} | {rate} |")

    # ── conclusion ────────────────────────────────────────────────────
    print("\n## 结论\n")
    n_tot = len(all_rows)
    g5, c5, net5, pos5, _ = summary[0.5]
    best_lam = max((l for l in LAMBDAS if l > 0), key=lambda l: summary[l][2])
    gb, cb, netb, posb, _ = summary[best_lam]

    print(f"λ=0.5：平均 PFC 改善 {g5:+.1f} Hz，全脑退化 {c5:+.1f} Hz，"
          f"净收益 {net5:+.1f} Hz，{pos5}/{n_tot} 名被试净收益为正。")
    if best_lam != 0.5:
        print(f"净收益最高的是 λ={best_lam:g}（{netb:+.1f} Hz，{posb}/{n_tot} 为正）。")

    if net5 > 0 and pos5 >= 0.7 * n_tot:
        print(f"\n**推荐 λ=0.5 作为默认值**：多数被试净收益为正，"
              f"且全脑退化幅度（平均 {c5:+.1f} Hz）远小于 PFC 收益。")
    elif netb > 0 and posb >= 0.7 * n_tot:
        print(f"\n**推荐 λ={best_lam:g} 作为默认值**：净收益 {netb:+.1f} Hz、"
              f"{posb}/{n_tot} 名被试为正，优于 λ=0.5 的 {net5:+.1f} Hz。")
    else:
        print(f"\n**没有 λ 能在多数被试上取得正净收益**（最好的 λ={best_lam:g} "
              f"只有 {posb}/{n_tot}）。PFC 加权应作为可选项而非默认，"
              "由使用者按当次研究是否只关心前额叶来决定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
