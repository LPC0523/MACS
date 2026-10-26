"""Synthetic checks for the PFC residual-mode study.

    D:\\python.exe tools/research/pfc_modes/test_pfc_modes.py

Everything here runs on constructed data with a known answer -- no subject is
loaded -- so a failure means the maths is wrong rather than the cohort changed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tools.research.pfc_modes.common import (            # noqa: E402
    SH8, SH9, basis_with_a00, independent_energy, sh_matrix)
from tools.research.pfc_modes.phase1_slepian import (    # noqa: E402
    taper_weight, whitened_gevp)
from tools.research.pfc_modes.phase1b_subspace import subspace_affinity  # noqa: E402

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def _grid():
    shape = (18, 20, 16)
    affine = np.diag([2.5, 2.5, 3.0, 1.0])
    affine[:3, 3] = [-22.5, -25.0, -24.0]
    return shape, affine


def test_projection():
    print("1. projection identity and SH orthogonality")
    shape, affine = _grid()
    basis = basis_with_a00(affine, shape)
    rng = np.random.default_rng(3)
    brain = np.zeros(shape, bool)
    brain[2:16, 2:18, 2:14] = True

    A = sh_matrix(basis, SH8, brain)
    C = rng.normal(size=(int(brain.sum()), 12))
    Q, _ = np.linalg.qr(A)
    P8C = Q @ (Q.T @ C)
    C_perp = C - P8C

    ok(np.max(np.abs(C - (P8C + C_perp))) < 1e-10,
       f"C == P8 C + C_perp  (max |diff| {np.max(np.abs(C - (P8C + C_perp))):.2e})")
    orth = np.max(np.abs(A.T @ C_perp)) / max(np.max(np.abs(A.T @ C)), 1e-12)
    ok(orth < 1e-10, f"S8^T C_perp == 0 relative to S8^T C  ({orth:.2e})")

    # A field that IS a harmonic must project away entirely.
    pure = (3.0 * basis["A20"] - 1.5 * basis["B21"])[brain]
    resid = pure - Q @ (Q.T @ pure)
    ok(np.linalg.norm(resid) / np.linalg.norm(pure) < 1e-10,
       "a pure harmonic has no independent part")

    # independent_energy must be 0 for the harmonics themselves and ~100% for
    # something orthogonal to them by construction.
    ok(abs(independent_energy(A, A)) < 1e-8,
       "independent_energy(SH, SH) == 0")
    ok(independent_energy(C_perp, A) > 99.999,
       "independent_energy(C_perp, SH) == 100%")

    # A00 must be in SH9's span, and substantially outside SH8's.
    #
    # Not "entirely" outside: a constant is only orthogonal to the linear terms
    # when the region is centred on the isocentre, and the second-order terms
    # are never orthogonal to it. Measured on this grid, a constant is 91% (a
    # centred ellipsoid 92%, one displaced by three voxels 59%) independent of
    # the 8 harmonics -- so the threshold is set at 50%, which separates "A00
    # buys real authority" from "A00 is redundant" without encoding one
    # region's geometry as if it were a law.
    a00 = basis["A00"][brain][:, None]
    ok(independent_energy(a00, sh_matrix(basis, SH9, brain)) < 1e-8,
       "A00 is inside the 9-term span")
    a00_ind = independent_energy(a00, A)
    ok(a00_ind > 50.0,
       f"A00 is substantially outside the 8-term span ({a00_ind:.1f}%)")


def test_taper():
    print("2. tapered weight, and that it is never confused with the hard mask")
    shape, affine = _grid()
    pfc = np.zeros(shape, bool)
    pfc[6:12, 7:13, 5:11] = True

    for taper in (5.0, 7.5, 10.0):
        w = taper_weight(pfc, affine, taper)
        ok(np.all(w[pfc] == 1.0), f"taper {taper}: 1.0 inside the mask")
        ok(w.min() >= 0.0 and w.max() <= 1.0, f"taper {taper}: values in [0,1]")
        ok(np.any((w > 0) & ~pfc), f"taper {taper}: extends beyond the mask")
        # Everything beyond the taper distance must be exactly zero.
        from scipy.ndimage import distance_transform_edt
        spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
        d = distance_transform_edt(~pfc, sampling=spacing)
        ok(np.all(w[d > taper + 1e-9] == 0.0),
           f"taper {taper}: zero beyond {taper} mm")

    w5 = taper_weight(pfc, affine, 5.0)
    w10 = taper_weight(pfc, affine, 10.0)
    ok(w10.sum() > w5.sum() > pfc.sum(),
       "a wider taper weighs more, and both exceed the hard mask")
    ok(abs(w5[pfc].sum() - pfc.sum()) < 1e-9,
       "the hard mask's own voxels are unweighted (taper adds only outside)")


def test_gevp():
    print("3. GEVP with a singular B")
    rng = np.random.default_rng(7)
    n, m, r = 400, 12, 8
    # B deliberately rank-deficient, exactly as C_perp makes it.
    F = rng.normal(size=(n, r)) @ rng.normal(size=(r, m))
    wB = np.ones(n)
    wP = np.zeros(n)
    wP[:80] = 1.0
    A = F.T @ (wP[:, None] * F)
    B = F.T @ (wB[:, None] * F)

    ok(np.linalg.matrix_rank(B) < m, f"B is singular (rank {np.linalg.matrix_rank(B)}/{m})")
    lam, V, rank = whitened_gevp(A, B)
    ok(rank == np.linalg.matrix_rank(B), f"kept rank {rank} matches B's rank")
    ok(np.all(np.diff(lam) <= 1e-12), "eigenvalues are sorted descending")
    ok(lam.min() > -1e-9 and lam.max() < 1 + 1e-9,
       f"eigenvalues lie in [0,1]  ({lam.min():.3e}, {lam.max():.6f})")

    # Residual of the generalised problem, restricted to B's range.
    res = []
    for k in range(len(lam)):
        v = V[:, k]
        res.append(np.linalg.norm(A @ v - lam[k] * (B @ v))
                   / max(np.linalg.norm(A @ v), 1e-12))
    ok(max(res) < 1e-8, f"A v = lambda B v  (max relative residual {max(res):.2e})")

    # The eigenvalue must equal the concentration ratio the mode achieves.
    for k in (0, 1):
        f = F @ V[:, k]
        got = float(np.sum(wP * f ** 2) / np.sum(wB * f ** 2))
        ok(abs(got - lam[k]) < 1e-9,
           f"mode {k}: lambda equals the realised energy fraction "
           f"({got:.6f} vs {lam[k]:.6f})")

    # And the top mode must beat a chance-level uniform field.
    chance = wP.sum() / n
    ok(lam[0] > chance, f"top mode ({lam[0]:.3f}) beats chance ({chance:.3f})")


def test_current_bounds():
    print("4. bounds act on real channel currents, not mode coefficients")
    from tools.research.pfc_modes.phase3_validation import solve

    rng = np.random.default_rng(11)
    n, m, K = 300, 12, 3
    A_sh = rng.normal(size=(n, 8))
    C = rng.normal(size=(n, m))
    b = rng.normal(size=n) * 50.0
    w = np.ones(n)
    bounds = [(-2000.0, 2000.0)] * 8
    imax = 3.0

    # A mode matrix with deliberately large channel amplitudes: a coefficient
    # bound would pass while the amplifiers would not.
    V = rng.normal(size=(m, K)) * 8.0
    theta, a, res, obj = solve(b, A_sh, C @ V, V, bounds, w, imax)
    i_real = V @ a
    ok(np.max(np.abs(i_real)) <= imax + 1e-6,
       f"real currents within +-{imax} A  (peak {np.max(np.abs(i_real)):.4f})")
    ok(np.max(np.abs(a)) > 0, "the solve actually used the modes")

    # Full-channel case: the identity mode matrix reduces to a box.
    theta2, i2, res2, obj2 = solve(b, A_sh, C, np.eye(m), bounds, w, imax)
    ok(np.max(np.abs(i2)) <= imax + 1e-6,
       f"full-channel currents within bounds (peak {np.max(np.abs(i2)):.4f})")

    # The subset argument: a reduced set cannot beat the full one.
    ok(obj >= obj2 - 1e-6 * max(abs(obj2), 1.0),
       f"reduced objective {obj:.6f} >= full objective {obj2:.6f}")


def test_subspace_affinity():
    print("5. subspace affinity behaves")
    rng = np.random.default_rng(5)
    A = rng.normal(size=(12, 4))
    ok(abs(subspace_affinity(A, A, 3)[0] - 1.0) < 1e-10,
       "a subspace is identical to itself")
    Q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    ok(subspace_affinity(Q[:, :3], Q[:, 3:6], 3)[0] < 1e-10,
       "orthogonal subspaces score 0")
    # Rotating within the span must not change it.
    R, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    ok(abs(subspace_affinity(A[:, :3], A[:, :3] @ R, 3)[0] - 1.0) < 1e-10,
       "rotation inside the span leaves affinity at 1")


def test_grid_checks():
    print("6. grid / affine guards")
    from core.data_context import same_grid

    shape, affine = _grid()
    ok(same_grid(shape, affine, shape, affine), "identical grids match")
    shifted = affine.copy(); shifted[0, 3] += 5.0
    ok(not same_grid(shape, shifted, shape, affine),
       "a translated affine does not match")
    ok(not same_grid((10, 10, 10), affine, shape, affine),
       "a different shape does not match")


if __name__ == "__main__":
    test_projection()
    test_taper()
    test_gevp()
    test_current_bounds()
    test_subspace_affinity()
    test_grid_checks()
    print("\nPASS" if not FAILS else "\nFAIL  " + "; ".join(FAILS))
    sys.exit(1 if FAILS else 0)
