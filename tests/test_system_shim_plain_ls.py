"""The system-shim page solves plain least squares, and says so honestly.

    D:\\python.exe tests/test_system_shim_plain_ls.py

Two halves, because the change had two halves. The solve must be the plain
overdetermined system -- no free constant column, no ridge, no reweighting --
and the page must report the residual as it will actually be, including the
uniform offset that the eight harmonics cannot remove.

The UI half runs in a subprocess on the native Qt platform and ends with
os._exit(0), for the reasons in tests/smoke_production.py.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def run_py(body, timeout=600):
    src = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, %r)
        try:
        %s
        except BaseException:
            import traceback; traceback.print_exc()
            sys.stdout.flush(); os._exit(17)
        sys.stdout.flush(); os._exit(0)
    """) % (str(ROOT), textwrap.indent(textwrap.dedent(body), "    "))
    # encoding is explicit: this test prints Chinese, and the default on a
    # Chinese Windows is GBK, which cannot decode it.
    r = subprocess.run([sys.executable, "-X", "utf8", "-u", "-c", src],
                       capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    return r


def _phantom():
    """A B0 map built from known harmonics plus a deliberate uniform offset.

    The offset is the whole point: it is what centring used to absorb for free.
    """
    from core.hybrid_shim import build_sh_basis, SH2

    shape = (24, 24, 20)
    affine = np.diag([3.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [-36.0, -36.0, -40.0]      # roughly centred on the isocentre

    basis = build_sh_basis(affine, shape)
    rng = np.random.default_rng(0)
    mask = np.zeros(shape, bool)
    mask[4:20, 4:20, 3:17] = True

    truth = {"A20": 130.0, "A21": -420.0, "B21": 260.0,
             "A22": 75.0, "B22": -50.0}
    b0 = np.full(shape, 35.0)                   # uniform offset, in Hz
    for n, c in truth.items():
        b0 = b0 + c * basis[n]
    b0 = b0 + rng.normal(0.0, 2.0, shape)       # a little noise
    return b0, mask, basis, affine, shape, list(SH2)


def test_solve():
    from core.hybrid_shim import joint_optimize

    b0, mask, basis, affine, shape, terms = _phantom()
    A = np.stack([basis[n][mask] for n in terms], axis=-1).astype(np.float64)
    b = b0[mask].astype(np.float64)

    print("1. the solve is plain least squares")
    w_ref = np.linalg.lstsq(A, -b, rcond=None)[0]
    delta_sh, i_pfc, residual, stats, res, scanner_delta, _ = joint_optimize(
        b0, basis, mask.astype(float), np.zeros(shape),
        np.zeros((*shape, 0)), 0, 3.0,
        0.0, 0.0, 0.0,
        sh_terms=terms, sh_bounds=[(-2000.0, 2000.0)] * len(terms))

    # Relative, not absolute: L-BFGS-B stops at its own tolerance, and the
    # coefficients here are in the hundreds, so 1e-5 absolute is 1e-7 relative
    # -- convergence, not a different answer.
    scale = float(np.max(np.abs(w_ref)))
    worst = float(np.max(np.abs(np.asarray(delta_sh) - w_ref))) / scale
    ok(worst < 1e-6, f"matches np.linalg.lstsq  max|diff|/scale={worst:.2e}")
    ok(bool(res.success), "solver converged")

    print("2. the residual keeps its uniform offset")
    r_ref = b + A @ w_ref
    got = float(np.mean(residual[mask]))
    ok(abs(got - float(np.mean(r_ref))) < 1e-6,
       f"residual mean unshifted  {got:+.3f} Hz")
    ok(abs(stats["centre_frequency_hz"] - got) < 1e-6,
       "reported centre frequency equals the residual mean")
    ok(abs(stats["brain_rmse_post"] - float(np.sqrt(np.mean(r_ref ** 2)))) < 1e-6,
       f"reported RMSE is the true RMSE  {stats['brain_rmse_post']:.3f} Hz")
    # The offset is real: with it in, RMSE must exceed the demeaned spread.
    ok(stats["brain_rmse_post"] > stats["brain_std_post"] + 1.0,
       f"RMSE {stats['brain_rmse_post']:.2f} > std "
       f"{stats['brain_std_post']:.2f} Hz, i.e. the offset was not hidden")

    print("3. centring is still available, and still different")
    from core.adaptive_joint import solve_adaptive_joint

    common = dict(sh_bounds=[(-2000.0, 2000.0)] * len(terms), imax=3.0,
                  lambda_current=0.0, lambda_sh=0.0, lambda_pfc_low=0.0,
                  weight_kwargs={"base": 1.0, "alpha": 0.0, "top_fraction": 0.0})
    basis_list = [basis[n] for n in terms]
    plain = solve_adaptive_joint(b0, mask, basis_list, None,
                                 centre=False, **common)
    cent = solve_adaptive_joint(b0, mask, basis_list, None,
                                centre=True, **common)
    ok(float(np.max(np.abs(plain["model_sh_delta"] - w_ref))) / scale < 1e-6,
       "centre=False reproduces the plain solution")
    gap = float(np.max(np.abs(cent["model_sh_delta"] - w_ref)))
    ok(gap > 1.0,
       f"centre=True is a genuinely different solve  max|diff|={gap:.1f}")
    # Default must stay True so the frozen research baselines are unchanged.
    dflt = solve_adaptive_joint(b0, mask, basis_list, None, **common)
    ok(float(np.max(np.abs(dflt["model_sh_delta"]
                           - cent["model_sh_delta"]))) < 1e-9,
       "centre defaults to True (research baselines unaffected)")

    print("4. no hidden ridge: lam_sh=0 really means unregularised")
    ridged = solve_adaptive_joint(b0, mask, basis_list, None, centre=False,
                                  **{**common, "lambda_sh": 0.5})
    shrunk = float(np.linalg.norm(ridged["model_sh_delta"])
                   < np.linalg.norm(plain["model_sh_delta"]))
    ok(bool(shrunk), "a non-zero lam_sh does shrink the coefficients")


def test_ui():
    print("5. the table carries the centre-frequency row")
    r = run_py("""
        import numpy as np
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication([])

        from widgets.system_shim_widget import SystemShimWidget
        from core.hybrid_shim import SH2

        w = SystemShimWidget({})
        terms = list(SH2)
        delta = [10.0, -20.0, 30.0, -40.0, 50.0]
        w._fill_sh_table(delta, terms, delta, None, 20.31)

        rows = w.sh_table.rowCount()
        print("ROWS", rows, len(terms))
        last = [w.sh_table.item(rows - 1, c).text() for c in range(3)]
        print("LAST", "|".join(last))
        print("TIP", w.sh_table.item(rows - 1, 2).toolTip().replace(chr(10), " "))

        # and without an offset the row must not appear
        w._fill_sh_table(delta, terms, delta, None, None)
        print("ROWS_NONE", w.sh_table.rowCount())

        # With the field map's own transmit frequency available, the row must
        # stop being an increment: an operator holding a bare "+20.3" has no
        # way to know which f0 it attaches to, and the console's current one is
        # the wrong answer whenever a prescan ran in between.
        w2 = SystemShimWidget({"imaging_frequency_hz": 123253430.0})
        w2._fill_sh_table(delta, terms, delta, None, 20.31)
        rows = w2.sh_table.rowCount()
        last = [w2.sh_table.item(rows - 1, c).text() for c in range(3)]
        print("ABS", "|".join(last))
        print("ABSTIP", w2.sh_table.item(rows - 1, 2).toolTip().replace(chr(10), " "))
    """)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr)
        ok(False, "UI subprocess failed")
        return

    out = {}
    for line in r.stdout.splitlines():
        k, _, v = line.partition(" ")
        out[k] = v

    ok(out.get("ROWS", "").startswith("6 5"),
       f"one row more than the harmonics  ({out.get('ROWS')})")
    last = out.get("LAST", "").split("|")
    ok(len(last) == 3 and "f0" in last[0],
       f"last row is the centre frequency  ({last[0] if last else '?'})")
    ok(len(last) == 3 and last[1].strip() == "—",
       "it has no 'current value' -- f0 is never read from the console")
    ok(len(last) == 3 and last[2].startswith("+20.3"),
       f"sign is +offset, to be added to f0  ({last[2] if len(last) > 2 else '?'})")
    ok(len(last) == 3 and "增量" in last[2],
       "and it is labelled as a change, not a setting")
    ok("增量" in out.get("TIP", "") and "deltaB0" in out.get("TIP", ""),
       "tooltip states the sign convention for the operator to check")
    ok(out.get("ROWS_NONE", "").strip() == "5",
       "no offset -> no row")

    print("6. with the map's f0 known, the row is absolute")
    ab = out.get("ABS", "").split("|")
    ok(len(ab) == 3 and "MHz" in ab[0],
       f"row switches to MHz  ({ab[0] if ab else '?'})")
    ok(len(ab) == 3 and ab[1].strip() == "123.253430",
       f"'current' is the field map's own frequency  ({ab[1] if len(ab) > 1 else '?'})")
    # 123253430 + 20.31 Hz = 123.253450 MHz to six decimals.
    ok(len(ab) == 3 and ab[2].strip() == "123.253450",
       f"target = map f0 + offset  ({ab[2] if len(ab) > 2 else '?'})")
    ok("控制台当前显示" in out.get("ABSTIP", ""),
       "and it says out loud that the console's f0 is not the baseline")


def test_roi_weighting():
    """A target region must actually get the array's attention.

    Synthetic, so it does not depend on subject data: the coil-like maps are
    localised over a small region, and the field is worst there. Under
    whole-brain weighting a region that small cannot outvote the rest.
    """
    from core.hybrid_shim import joint_optimize

    b0, mask, basis, affine, shape, terms = _phantom()

    # A small target, and "coils" that mostly act on it.
    target = np.zeros(shape, bool)
    target[6:11, 6:11, 5:10] = True
    target &= mask
    frac = target.sum() / mask.sum()

    xx, yy, zz = np.meshgrid(*[np.arange(s) for s in shape], indexing="ij")
    maps = []
    for cx, cy in ((8, 8), (8, 10), (10, 8), (10, 10)):
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - 7) ** 2
        maps.append(30.0 * np.exp(-d2 / 18.0))
    maps = np.stack(maps, axis=-1)

    # Make the target the hard part, so whole-brain weighting has something
    # else it would rather spend on.
    b0 = b0 + 60.0 * target

    print(f"6. target region weighting  ({frac * 100:.1f}% of the mask)")

    def run(roi, lam=1.0):
        out = joint_optimize(b0, basis, mask.astype(float), np.zeros(shape),
                             maps, maps.shape[-1], 3.0, 0.0, 0.0, 0.0,
                             sh_terms=terms,
                             sh_bounds=[(-2000.0, 2000.0)] * len(terms),
                             roi_mask=roi, lambda_rest=lam)
        return out[2], out[3]

    res_wb, st_wb = run(None)
    res_roi, st_roi = run(target, 1.0)

    t_wb, t_roi = np.std(res_wb[target]), np.std(res_roi[target])
    print(f"   target std  whole-brain {t_wb:.2f} -> weighted {t_roi:.2f} Hz")
    ok(t_roi < t_wb, "weighting improves the target region")
    ok("roi_std_pre" in st_roi and "roi_std_post" in st_roi,
       "target-region statistics are reported")
    ok(st_roi.get("_roi_weighted") is True
       and st_roi.get("_roi_voxels") == int(target.sum()),
       "the result records that it was weighted, and over how many voxels")
    ok("roi_std_pre" not in st_wb,
       "a whole-brain run reports no target statistics")

    print("7. degenerate regions fall back instead of failing")
    res_e, st_e = run(np.zeros(shape, bool))
    ok(np.allclose(res_e, res_wb), "an empty region falls back to whole brain")
    ok("roi_std_pre" not in st_e, "and reports no target statistics")
    outside = np.zeros(shape, bool)
    outside[0, 0, 0] = True                      # outside the mask
    res_o, _ = run(outside)
    ok(np.allclose(res_o, res_wb), "a region disjoint from the mask falls back")


if __name__ == "__main__":
    test_solve()
    test_roi_weighting()
    test_ui()
    print("\nPASS" if not FAILS else "\nFAIL  " + "; ".join(FAILS))
    sys.exit(1 if FAILS else 0)
