"""Three shim configurations compared on one subject. Standalone -- not imported by the app.

    D:\\python.exe tests\\run_two_stage.py [--subject S28] [--coil "PFC 12ch"]

What is being asked: does keeping first order out of the design give the coils
more to do? First order and a surface array produce the same kind of field over
a local region, and the solver prefers the harmonics because three global
coefficients cost less than twelve local currents.

  1  1st + 2nd order harmonics + coils   -- what the app ships
  2  2nd order only + coils, joint       -- first order simply absent
  3  two-stage                           -- coils shape the target first, then
                                            2nd order cleans up

All figures are demeaned spread (std), matching the UI: the constant component
is the centre frequency and is not something this optimiser moves.

"Coil contribution" is the honest measure of whether the coils earned their
place -- the residual with the coils switched off, minus the residual with them
on, holding the harmonic solution fixed. A configuration can have a fine RMSE
and still be one where the coils did nothing.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np
import nibabel as nib

warnings.filterwarnings("ignore")

from core.adaptive_joint import solve_adaptive_joint, solve_two_stage
from core.coil_registry import get_coil
from core.hybrid_shim import SH2, SH12, build_sh_basis
from core.pipeline import _load_coil_maps

UNIFORM = {"base": 1.0, "alpha": 0.0, "top_fraction": 0.0}


def find_subject(name: str) -> tuple[Path, Path]:
    roots = [_HERE.parent.parent / "test_dic",
             _HERE.parent.parent.parent / "2nd_order_SH"]
    for root in roots:
        if not root.is_dir():
            continue
        for hz in root.rglob(f"{name}_deltaB0_hertz.nii.gz"):
            mask = hz.parent / f"{name}_bet_mask.nii.gz"
            if mask.is_file():
                return hz, mask
    raise SystemExit(f"找不到被试 {name} 的 deltaB0 + bet_mask 配对")


def coil_gain(b0, mask, basis, names, theta, coil_maps, currents) -> float:
    """How much spread the coils removed, with the harmonic part held fixed."""
    sh = np.zeros_like(b0)
    for k, n in enumerate(names):
        sh += theta[k] * basis[n]
    with_coils = b0 + sh + np.tensordot(coil_maps, currents, axes=([3], [0]))
    return float(np.std((b0 + sh)[mask]) - np.std(with_coils[mask]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", default="S28")
    ap.add_argument("--coil", default="PFC 12ch")
    ap.add_argument("--epsilon", type=float, default=0.05)
    ap.add_argument("--grad", action="store_true",
                    help="λ_grad 梯度正则化扫参（单被试）")
    ap.add_argument("--batch-nc", action="store_true",
                    help="批量对比：全脑 joint vs 两阶段（无保护约束）")
    ap.add_argument("--sweep", action="store_true",
                    help="扫 ε，出表格 + 图，回答『放宽保护约束能换到多少 PFC 改善』")
    ap.add_argument("--pfc-mask", default="",
                    help="目标区 mask；留空则两阶段的目标区退化为全脑")
    args = ap.parse_args()

    hz_path, mask_path = find_subject(args.subject)
    img = nib.load(str(hz_path))
    b0 = np.asanyarray(img.dataobj, dtype=np.float64)
    mask = np.asanyarray(nib.load(str(mask_path)).dataobj) > 0.5

    coil = get_coil(args.coil)
    if coil is None:
        raise SystemExit(f"找不到线圈配置：{args.coil}")
    imax = coil.current_limit_a
    maps, ch_names = _load_coil_maps(coil, img.affine, b0.shape[:3], lambda *a: None)
    basis = build_sh_basis(img.affine, b0.shape[:3])
    n_ch = len(ch_names)

    print(f"被试 {args.subject} · 线圈 {coil.name} ({n_ch}ch, ±{imax:g} A) · "
          f"脑内 {int(mask.sum()):,} 体素")
    print(f"匀场前 std {float(np.std(b0[mask])):.1f} Hz\n")

    if args.grad:
        if not args.pfc_mask:
            raise SystemExit("--grad 需要 --pfc-mask")
        pfc = (np.asanyarray(nib.load(args.pfc_mask).dataobj) > 0.5) & mask
        vs = tuple(float(v) for v in nib.affines.voxel_sizes(img.affine))
        print(f"体素尺寸 {vs[0]:.2f} × {vs[1]:.2f} × {vs[2]:.2f} mm")
        run_grad_sweep(b0, mask, pfc, basis, maps, imax, n_ch, voxel_size=vs)
        return 0

    if args.batch_nc:
        return run_batch_nc(coil, limit=0)

    if args.sweep:
        if not args.pfc_mask:
            raise SystemExit("--sweep 需要 --pfc-mask：目标区退化为全脑时保护约束没有意义")
        pfc = np.asanyarray(nib.load(args.pfc_mask).dataobj) > 0.5
        print(f"目标区 mask：{args.pfc_mask}（{int((pfc & mask).sum()):,} 体素，"
              f"占全脑 {100 * (pfc & mask).sum() / mask.sum():.1f}%）")
        run_sweep(b0, mask, pfc, basis, maps, imax,
                  str(_HERE / "pfc_epsilon_sweep.png"), default_eps=args.epsilon)
        return 0

    rows = []

    # 1 -- what the app ships: all eight harmonics plus the coils.
    r1 = solve_adaptive_joint(b0, mask, [basis[n] for n in SH12], maps, imax=imax,
                              weight_kwargs=UNIFORM, lambda_pfc_low=0.0,
                              log_cb=lambda *a: None)
    rows.append((
        "1+2 阶 + 线圈", float(np.std(b0[mask])),
        float(np.std(r1["residual"][mask])), r1["channels_at_limit"],
        coil_gain(b0, mask, basis, SH12, r1["scanner_sh_delta"], maps,
                  r1["coil_currents"]),
        float(np.mean(np.abs(r1["coil_currents"]))),
    ))

    # 2 -- first order simply not in the design.
    r2 = solve_adaptive_joint(b0, mask, [basis[n] for n in SH2], maps, imax=imax,
                              weight_kwargs=UNIFORM, lambda_pfc_low=0.0,
                              log_cb=lambda *a: None)
    rows.append((
        "仅 2 阶 + 线圈", float(np.std(b0[mask])),
        float(np.std(r2["residual"][mask])), r2["channels_at_limit"],
        coil_gain(b0, mask, basis, SH2, r2["scanner_sh_delta"], maps,
                  r2["coil_currents"]),
        float(np.mean(np.abs(r2["coil_currents"]))),
    ))

    # 3 -- coils first over the target region, harmonics after.
    pfc = None
    if args.pfc_mask:
        pfc = np.asanyarray(nib.load(args.pfc_mask).dataobj) > 0.5
        print(f"目标区 mask：{args.pfc_mask}（{int((pfc & mask).sum()):,} 体素）")
    r3 = solve_two_stage(b0, mask, basis, maps, pfc_mask=pfc, imax=imax,
                         epsilon=args.epsilon, log_cb=print)
    rows.append((
        "两阶段", r3["rmse_before"], r3["rmse_after"], r3["channels_at_limit"],
        coil_gain(b0, mask, basis, r3["sh2_names"], r3["theta_2nd"], maps,
                  r3["coil_currents"]),
        float(np.mean(np.abs(r3["coil_currents"]))),
    ))

    print()
    head = (f"{'配置':<16}{'std 前':>9}{'std 后':>9}{'改善':>8}"
            f"{'饱和':>8}{'线圈贡献':>10}{'|I| 均值':>10}")
    print(head)
    print("-" * len(head.encode("gbk", "replace")))
    for tag, before, after, sat, gain, imean in rows:
        drop = (1 - after / max(before, 1e-9)) * 100
        print(f"{tag:<16}{before:>9.1f}{after:>9.1f}{drop:>7.0f}%"
              f"{sat:>6}/{n_ch}{gain:>10.1f}{imean:>10.2f}")

    print("\n分阶段明细（两阶段）")
    print(f"  Stage 1 目标区 std {r3['stage1_rmse_before']:.1f} → "
          f"{r3['stage1_rmse_after']:.1f} Hz  (电流范围 ±{r3['current_limit_used']:.2f} A)")
    print(f"  Stage 2 全脑   std {r3['stage2_rmse_before']:.1f} → "
          f"{r3['stage2_rmse_after']:.1f} Hz")
    print(f"  二阶系数 " + "  ".join(
        f"{n}={v:+.0f}" for n, v in zip(r3["sh2_names"], r3["theta_2nd"])))
    return 0




# ─────────────────────────────────────────────────────────────────────────
#  epsilon sweep: what does protecting the rest of the brain cost the PFC?
# ─────────────────────────────────────────────────────────────────────────
# Stage 1 asks the coils to flatten the PFC, and `epsilon` caps how much
# whole-brain spread may grow while they do it. The PFC is ~8% of the brain, so
# the two goals are genuinely opposed and no solver can decide the trade for
# you. This sweep measures the exchange rate so the choice can be made on
# numbers instead of on a default.

EPSILONS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
            0.70, 0.80, 0.90, 1.00, 1.25, 1.50]

#: What counts as clinically acceptable, for the shaded region in the figure.
ACCEPT_DEG_PCT = 15.0
ACCEPT_GAIN_HZ = 5.0

#: Below this, a "gain" is not worth any whole-brain cost -- it is inside the
#: run-to-run spread of the field maps themselves.
MEANINGFUL_GAIN_HZ = 3.0


def _sweep_row(b0, brain, pfc, basis, maps, imax, eps):
    r = solve_two_stage(b0, brain, basis, maps, pfc_mask=pfc, imax=imax,
                        epsilon=eps, log_cb=lambda *a: None)
    resid = r["residual"]
    pfc_in = pfc & brain
    pfc_before = float(np.std(b0[pfc_in]))
    pfc_after = float(np.std(resid[pfc_in]))
    whole_before = float(np.std(b0[brain]))
    whole_after = float(np.std(resid[brain]))
    cur = r["coil_currents"]
    return {
        "eps": eps,
        "pfc_before": pfc_before,
        "pfc_after": pfc_after,
        "pfc_gain": pfc_before - pfc_after,
        "whole_before": whole_before,
        "whole_after": whole_after,
        "whole_deg_pct": (whole_after / whole_before - 1.0) * 100.0,
        "coil_gain": coil_gain(b0, brain, basis, r["sh2_names"], r["theta_2nd"],
                               maps, cur),
        "n_sat": r["channels_at_limit"],
        "mean_current": float(np.mean(np.abs(cur))),
        "imax_eff": r["current_limit_used"],
        "cur_range": f"{cur.min():+.2f} ~ {cur.max():+.2f}",
    }


def _plot_sweep(rows, out_path, default_eps=0.05):
    """Pareto front: what each unit of whole-brain degradation buys the PFC.

    Plotted as gain against cost rather than both against epsilon, because
    epsilon is a knob and these two are the quantities anyone actually trades.
    A point that sits up and to the left is a good deal; the shape of the
    frontier says whether such a point exists at all.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False

    deg = [r["whole_deg_pct"] for r in rows]
    gain = [r["pfc_gain"] for r in rows]

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=150)

    # Clinically acceptable quadrant, drawn first so the points sit on top.
    ax.axvspan(min(min(deg) - 5, -8), ACCEPT_DEG_PCT, ymin=0, ymax=1,
               color="#2da44e", alpha=0.06, lw=0)
    ax.axvline(ACCEPT_DEG_PCT, color="#2da44e", ls="--", lw=1.4)
    ax.axhline(ACCEPT_GAIN_HZ, color="#2da44e", ls="--", lw=1.4)
    ax.annotate(f"临床可接受区\n全脑退化 < {ACCEPT_DEG_PCT:g}%　PFC 改善 > {ACCEPT_GAIN_HZ:g} Hz",
                xy=(ACCEPT_DEG_PCT, ACCEPT_GAIN_HZ), xytext=(-8, 8),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=9, color="#2da44e")

    ax.axhline(0, color="#999999", lw=0.8)
    ax.plot(deg, gain, "-", color="#1f6feb", lw=1.8, zorder=2)
    ax.scatter(deg, gain, s=70, color="#1f6feb", zorder=3, label="两阶段解")

    for r in rows:
        ax.annotate(f"ε={r['eps']:.2f}",
                    xy=(r["whole_deg_pct"], r["pfc_gain"]), xytext=(7, -3),
                    textcoords="offset points", fontsize=9, color="#24292f")

    d0 = next((r for r in rows if abs(r["eps"] - default_eps) < 1e-9), None)
    if d0:
        ax.scatter([d0["whole_deg_pct"]], [d0["pfc_gain"]], s=170,
                   facecolors="none", edgecolors="#d1242f", lw=2, zorder=4)
        ax.annotate(f"当前默认 ε={default_eps:g}（保守）",
                    xy=(d0["whole_deg_pct"], d0["pfc_gain"]), xytext=(10, -22),
                    textcoords="offset points", fontsize=9, color="#d1242f",
                    arrowprops=dict(arrowstyle="-", color="#d1242f", lw=1))

    # The break-even line: above it the PFC gains more Hz than the whole brain
    # gives up, below it the trade is net negative in absolute terms.
    whole_before = rows[0]["whole_before"]
    xs = np.linspace(min(deg) - 3, max(deg) + 5, 50)
    ax.plot(xs, whole_before * xs / 100.0, ":", color="#8250df", lw=1.4,
            label="等量兑换线（换得 = 让出，单位 Hz）")

    ax.set_xlabel("全脑 std 退化 (%)　→ 代价")
    ax.set_ylabel("PFC std 改善 (Hz)　↑ 收益")
    ax.set_title("PFC 优先策略的 Pareto 前沿", fontsize=13)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def run_sweep(b0, brain, pfc, basis, maps, imax, out_png, default_eps=0.05):
    rows = [_sweep_row(b0, brain, pfc, basis, maps, imax, e) for e in EPSILONS]

    print("\n## ε 扫参结果\n")
    print("| ε | PFC 前 | PFC 后 | PFC 改善 | 全脑前 | 全脑后 | 全脑退化 "
          "| 兑换率 | 线圈贡献 | 饱和 | \\|I\\| 均值 | 有效上限 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        # Exchange rate: Hz of whole-brain spread given up per Hz gained in the
        # PFC. Only meaningful where the PFC actually gained something.
        cost = r["whole_after"] - r["whole_before"]
        rate = (f"{cost / r['pfc_gain']:.1f} : 1" if r["pfc_gain"] > 0.05 else "—")
        print(f"| {r['eps']:.2f} | {r['pfc_before']:.1f} | {r['pfc_after']:.1f} "
              f"| {r['pfc_gain']:+.1f} | {r['whole_before']:.1f} | {r['whole_after']:.1f} "
              f"| {r['whole_deg_pct']:+.1f}% | {rate} | {r['coil_gain']:.1f} | {r['n_sat']} "
              f"| {r['mean_current']:.2f} | ±{r['imax_eff']:.2f} |")

    path = _plot_sweep(rows, out_png, default_eps)
    print(f"\n图已保存：{path}")

    best = max(rows, key=lambda r: r["pfc_gain"])
    print("\n## 结论\n")
    if best["pfc_gain"] < MEANINGFUL_GAIN_HZ:
        print(f"所有 ε 下 PFC 改善均不足 {MEANINGFUL_GAIN_HZ:g} Hz"
              f"（最大 {best['pfc_gain']:+.1f} Hz @ ε={best['eps']:.2f}，"
              f"代价是全脑退化 {best['whole_deg_pct']:+.1f}%）——"
              "放宽保护约束换不到有意义的 PFC 收益，"
              "瓶颈在线圈本身的空间自由度，应转向线圈设计优化而非求解器调参。")
    else:
        # The knee: the largest epsilon whose whole-brain cost is still under
        # the PFC gain in absolute Hz -- i.e. you gain more than you give away.
        viable = [r for r in rows
                  if r["pfc_gain"] >= MEANINGFUL_GAIN_HZ
                  and (r["whole_after"] - r["whole_before"]) < r["pfc_gain"]]
        if viable:
            k = max(viable, key=lambda r: r["pfc_gain"])
            print(f"ε={k['eps']:.2f} 是可接受的平衡点：PFC std 改善 "
                  f"{k['pfc_gain']:+.1f} Hz（{k['pfc_before']:.1f} → {k['pfc_after']:.1f}），"
                  f"代价是全脑退化 {k['whole_deg_pct']:+.1f}%"
                  f"（{k['whole_before']:.1f} → {k['whole_after']:.1f} Hz），"
                  "换来的比让出的多。")
        else:
            print(f"最大 PFC 改善 {best['pfc_gain']:+.1f} Hz @ ε={best['eps']:.2f}，"
                  f"但同时全脑退化 {best['whole_deg_pct']:+.1f}%"
                  f"（{best['whole_before']:.1f} → {best['whole_after']:.1f} Hz）—— "
                  "让出的绝对量超过换来的，没有净收益为正的 ε。")
    return rows


# ─────────────────────────────────────────────────────────────────────────
#  Batch: unconstrained two-stage vs whole-brain joint
# ─────────────────────────────────────────────────────────────────────────
# "Unconstrained" means stage 1 lets the coils chase the PFC with their full
# range and no whole-brain protection at all -- reachable through the existing
# `solve_two_stage` by setting `epsilon` so large the tolerance can never be
# breached, so no second implementation of the same two stages exists.
#
#   stage 1   min ||b0_PFC + L_PFC i||^2       s.t. |i| <= imax
#   stage 2   min ||b0 + L i + G_2nd theta||^2 over the whole brain
#
# The question is not whether the PFC improves -- with no constraint it must --
# but what the rest of the head pays for it, and whether that is a trade worth
# making. Both halves are therefore reported side by side.

#: Large enough that the whole-brain tolerance is never binding.
NO_CONSTRAINT_EPS = 1e9

_SUBJECT_ROOTS = [_HERE.parent.parent.parent / "2nd_order_SH",
                  _HERE.parent.parent / "test_dic"]


def find_all_subjects(limit: int = 0) -> list:
    """(name, b0, brain mask, pfc mask) for every subject that has all three."""
    found, seen = [], set()
    for root in _SUBJECT_ROOTS:
        if not root.is_dir():
            continue
        for hz in sorted(root.rglob("*_deltaB0_hertz.nii.gz")):
            name = hz.name[: -len("_deltaB0_hertz.nii.gz")]
            if name in seen:
                continue
            brain = hz.parent / f"{name}_bet_mask.nii.gz"
            pfc = next((p for p in (hz.parent / f"{name}_pfc_mask.nii.gz",
                                    hz.parent / "roi_cache" / "pfc_native.nii.gz")
                        if p.is_file()), None)
            if brain.is_file() and pfc is not None:
                seen.add(name)
                found.append((name, hz, brain, pfc))
                if limit and len(found) >= limit:
                    return found
    return found


def run_batch_nc(coil, limit: int = 0) -> int:
    imax = coil.current_limit_a
    subjects = find_all_subjects(limit)
    if not subjects:
        raise SystemExit("没有找到同时具备 deltaB0 / bet_mask / pfc_mask 的被试")

    print(f"线圈 {coil.name} ({coil.channel_count}ch, ±{imax:g} A)　"
          f"候选被试 {len(subjects)} 名\n", flush=True)

    rows = []
    for name, hz, brain_p, pfc_p in subjects:
        img = nib.load(str(hz))
        b0 = np.asanyarray(img.dataobj, dtype=np.float64)
        brain = np.asanyarray(nib.load(str(brain_p)).dataobj) > 0.5
        pfc = np.asanyarray(nib.load(str(pfc_p)).dataobj) > 0.5
        if pfc.shape != brain.shape:
            print(f"[跳过] {name}: PFC mask {pfc.shape} != 脑掩膜 {brain.shape}",
                  flush=True)
            continue
        pfc &= brain
        if pfc.sum() < 500 or brain.sum() < 20000:
            print(f"[跳过] {name}: PFC {int(pfc.sum())} / 脑 {int(brain.sum())} 体素过少",
                  flush=True)
            continue
        try:
            maps, _ = _load_coil_maps(coil, img.affine, b0.shape[:3], lambda *a: None)
        except Exception as e:
            print(f"[跳过] {name}: 线圈重采样失败 {type(e).__name__}", flush=True)
            continue

        basis = build_sh_basis(img.affine, b0.shape[:3])

        # 1 -- whole-brain joint, the shipping baseline
        j = solve_adaptive_joint(b0, brain, [basis[n] for n in SH12], maps,
                                 imax=imax, weight_kwargs=UNIFORM,
                                 lambda_pfc_low=0.0, log_cb=lambda *a: None)
        jr = j["residual"]

        # 2 -- two-stage, no whole-brain protection
        t = solve_two_stage(b0, brain, basis, maps, pfc_mask=pfc, imax=imax,
                            epsilon=NO_CONSTRAINT_EPS, log_cb=lambda *a: None)
        tr = t["residual"]

        rows.append({
            "name": name,
            "pfc_before": float(np.std(b0[pfc])),
            "whole_before": float(np.std(b0[brain])),
            "j_pfc": float(np.std(jr[pfc])),
            "j_whole": float(np.std(jr[brain])),
            "j_sat": j["channels_at_limit"],
            "t_pfc": float(np.std(tr[pfc])),
            "t_whole": float(np.std(tr[brain])),
            "t_sat": t["channels_at_limit"],
        })
        r = rows[-1]
        print(f"  {name:<12} PFC {r['pfc_before']:.0f} → joint {r['j_pfc']:.0f} / "
              f"两阶段 {r['t_pfc']:.0f}　全脑 {r['whole_before']:.0f} → "
              f"joint {r['j_whole']:.0f} / 两阶段 {r['t_whole']:.0f}", flush=True)

    if not rows:
        raise SystemExit("没有任何被试跑通")

    print("\n## 全脑 joint vs 两阶段（无保护约束）\n")
    print("| 被试 | PFC 前 | joint PFC 后 | 两阶段 PFC 后 | 全脑前 | joint 全脑后 "
          "| 两阶段 全脑后 | 饱和 joint/两阶段 |")
    print("|" + "---|" * 8)
    for r in rows:
        print(f"| {r['name']} | {r['pfc_before']:.1f} | {r['j_pfc']:.1f} "
              f"| {r['t_pfc']:.1f} | {r['whole_before']:.1f} | {r['j_whole']:.1f} "
              f"| {r['t_whole']:.1f} | {r['j_sat']}/{coil.channel_count} → "
              f"{r['t_sat']}/{coil.channel_count} |")

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print(f"| **平均** | **{avg('pfc_before'):.1f}** | **{avg('j_pfc'):.1f}** "
          f"| **{avg('t_pfc'):.1f}** | **{avg('whole_before'):.1f}** "
          f"| **{avg('j_whole'):.1f}** | **{avg('t_whole'):.1f}** | |")

    # ── verdict ───────────────────────────────────────────────────────
    n = len(rows)
    pfc_better = sum(1 for r in rows if r["t_pfc"] < r["j_pfc"])
    whole_ok = sum(1 for r in rows if r["t_whole"] <= r["j_whole"] * 1.02)
    both = sum(1 for r in rows
               if r["t_pfc"] < r["j_pfc"] and r["t_whole"] <= r["j_whole"] * 1.02)
    d_pfc = avg("j_pfc") - avg("t_pfc")
    d_whole = avg("t_whole") - avg("j_whole")

    print("\n## 结论\n")
    print(f"PFC 优于 joint：{pfc_better}/{n} 名　全脑不劣于 joint（放宽 2%）："
          f"{whole_ok}/{n} 名　两者同时满足：{both}/{n} 名")
    print(f"平均：PFC {avg('j_pfc'):.1f} → {avg('t_pfc'):.1f} Hz（{-d_pfc:+.1f}），"
          f"全脑 {avg('j_whole'):.1f} → {avg('t_whole'):.1f} Hz（{d_whole:+.1f}）")

    if both >= 0.7 * n:
        print(f"\n**成功**：多数被试上 PFC 改善且全脑不劣于全脑 joint，"
              "两阶段无约束可以取代现有基线。")
    elif pfc_better >= 0.7 * n and d_whole > 0:
        ratio = d_whole / d_pfc if d_pfc > 0.05 else float("inf")
        print(f"\n**兑换不划算**：PFC 确实改善（平均 {d_pfc:+.1f} Hz），"
              f"但全脑同时退化 {d_whole:+.1f} Hz，兑换率 {ratio:.1f} : 1。"
              "只有当本次研究完全不关心前额叶以外区域时才值得。")
    else:
        print(f"\n**不成立**：PFC 只在 {pfc_better}/{n} 名被试上优于全脑 joint，"
              "无保护约束的两阶段没有稳定优势。")
    return 0


# ─────────────────────────────────────────────────────────────────────────
#  Gradient regularisation vs value penalty
# ─────────────────────────────────────────────────────────────────────────
# `lambda_rest` penalises the residual *field* outside the target, which
# opposes the coils everywhere at once. `lambda_grad` penalises its spatial
# *slope* instead, leaving a smooth offset outside the target alone while still
# forbidding a sharp edge at the boundary -- the edge being what shows up as
# dropout. The two can be combined.

GRAD_CONFIGS = [
    ("基线：全脑 joint", dict()),
    ("λ_rest=0.5", dict(lambda_rest=0.5)),
    ("λ_grad=0.01", dict(lambda_rest=0.0, lambda_grad=0.01)),
    ("λ_grad=0.05", dict(lambda_rest=0.0, lambda_grad=0.05)),
    ("λ_grad=0.1", dict(lambda_rest=0.0, lambda_grad=0.1)),
    ("λ_rest=0.2 + λ_grad=0.05", dict(lambda_rest=0.2, lambda_grad=0.05)),
]


def run_grad_sweep(b0, brain, pfc, basis, maps, imax, n_ch, voxel_size=None):
    rest = brain & ~pfc

    def grad_energy(v):
        """Mean squared slope outside the target -- what lambda_grad acts on."""
        gs = np.gradient(v) if voxel_size is None else np.gradient(v, *voxel_size)
        return float(np.mean(sum(g[rest] ** 2 for g in gs)))

    print(f"PFC {int(pfc.sum()):,} 体素（{100 * pfc.sum() / brain.sum():.1f}%）　"
          f"非 PFC {int(rest.sum()):,} 体素")
    print()
    head = (f"{'配置':<26}{'PFC 前→后':>16}{'非PFC 前→后':>16}"
            f"{'全脑 前→后':>16}{'边界梯度':>10}{'饱和':>8}{'|I|':>7}")
    print(head)
    print("-" * 100)
    print(f"{'（匀场前）':<26}{np.std(b0[pfc]):>9.1f}{'':>7}"
          f"{np.std(b0[rest]):>9.1f}{'':>7}{np.std(b0[brain]):>9.1f}")

    base_grad = None
    rows = []
    for tag, kw in GRAD_CONFIGS:
        opts = dict(imax=imax, lambda_pfc_low=0.0, log_cb=lambda *a: None)
        if kw:
            opts.update(pfc_mask=pfc, voxel_size=voxel_size, **kw)
        else:
            opts.update(weight_kwargs=UNIFORM)
        r = solve_adaptive_joint(b0, brain, [basis[n] for n in SH12], maps, **opts)
        v = r["residual"]
        ge = grad_energy(v)
        if base_grad is None:
            base_grad = ge
        rows.append((tag, float(np.std(v[pfc])), float(np.std(v[rest])),
                     float(np.std(v[brain])), ge / base_grad,
                     r["channels_at_limit"],
                     float(np.mean(np.abs(r["coil_currents"])))))
        t = rows[-1]
        print(f"{tag:<26}{np.std(b0[pfc]):>7.1f}→{t[1]:>7.1f} "
              f"{np.std(b0[rest]):>7.1f}→{t[2]:>7.1f} "
              f"{np.std(b0[brain]):>7.1f}→{t[3]:>7.1f} "
              f"{t[4]:>9.2f}×{t[5]:>6}/{n_ch}{t[6]:>7.2f}", flush=True)

    base = rows[0]
    print()
    print("相对基线（全脑 joint）：")
    for tag, pfc_a, rest_a, whole_a, ge, sat, mi in rows[1:]:
        print(f"  {tag:<26} PFC {base[1] - pfc_a:+6.1f} Hz　"
              f"非PFC {rest_a - base[2]:+6.1f} Hz　"
              f"全脑 {whole_a - base[3]:+6.1f} Hz　边界梯度 {ge:.2f}×")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
