"""DSV coverage: one lattice, one rule, both counts.

    D:\\python.exe tests/test_sphere_coverage.py

The load-bearing case is the first: a sphere entirely inside the field of view
with every voxel valid must report coverage of exactly 1.0. A continuous
volume estimate cannot do that -- it lands near 1 and wobbles with the radius --
which is why the denominator counts lattice sites rather than cubic metres.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.spatial_grid import (COVERAGE_THRESHOLD,  # noqa: E402
                               sphere_support, world_coordinates)

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def aff(vox=(2.0, 2.0, 2.0), off=(-30.0, -30.0, -30.0), rot_deg=0.0):
    A = np.eye(4)
    R = np.eye(3)
    if rot_deg:
        th = np.deg2rad(rot_deg)
        R = np.array([[np.cos(th), -np.sin(th), 0.0],
                      [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    A[:3, :3] = R @ np.diag(vox)
    A[:3, 3] = off
    return A


def centre_of(A, shape):
    return world_coordinates(A, shape).mean(axis=0)


def t_exact_one():
    print("1. a fully sampled sphere is exactly 1.0")
    shape = (31, 31, 31)
    A = aff(off=(-30.0, -30.0, -30.0))
    c = centre_of(A, shape)
    valid = np.ones(shape, dtype=bool)
    for dsv_cm in (2.0, 4.0, 6.0):
        r = sphere_support(A, shape, c, dsv_cm / 100.0, valid)
        ok(r["coverage"] == 1.0,
           "DSV %.0f cm inside the FOV: coverage exactly 1.0 (%d/%d)"
           % (dsv_cm, r["valid_sphere_voxels"], r["theoretical_sphere_voxels"]))
        ok(not r["truncated"], "and not truncated")
        ok(r["valid_sphere_voxels"] == r["theoretical_sphere_voxels"],
           "numerator equals denominator")


def t_outside_fov():
    print("2. a sphere larger than the field of view is truncated")
    shape = (16, 16, 16)             # 32 mm across at 2 mm
    A = aff(off=(-16.0, -16.0, -16.0))
    c = centre_of(A, shape)
    valid = np.ones(shape, dtype=bool)
    small = sphere_support(A, shape, c, 0.02, valid)
    ok(small["coverage"] == 1.0, "a 2 cm sphere still fits (1.0)")
    big = sphere_support(A, shape, c, 0.30, valid)
    ok(big["coverage"] < 1.0,
       "a 30 cm sphere on a 3.2 cm grid is not fully covered (%.4f)"
       % big["coverage"])
    ok(big["truncated"], "and is flagged truncated")
    ok(big["theoretical_sphere_voxels"] > int(np.prod(shape)),
       "its denominator exceeds the whole array (%d > %d)"
       % (big["theoretical_sphere_voxels"], int(np.prod(shape))))
    ok(big["valid_sphere_voxels"] <= int(np.prod(shape)),
       "while the numerator cannot")
    # Monotone: growing the sphere past the FOV can only lower coverage.
    prev = 1.1
    for dsv in (0.02, 0.04, 0.08, 0.16, 0.30):
        cov = sphere_support(A, shape, c, dsv, valid)["coverage"]
        ok(cov <= prev + 1e-12, "DSV %.0f cm -> %.4f, not above the previous"
           % (100 * dsv, cov))
        prev = cov


def t_holes():
    print("3. holes in the valid mask lower coverage")
    shape = (31, 31, 31)
    A = aff()
    c = centre_of(A, shape)
    full = np.ones(shape, dtype=bool)
    base = sphere_support(A, shape, c, 0.04, full)
    ok(base["coverage"] == 1.0, "baseline is 1.0")

    holed = full.copy()
    holed[14:17, 14:17, 14:17] = False        # 27 voxels punched out of it
    r = sphere_support(A, shape, c, 0.04, holed)
    ok(r["coverage"] < 1.0, "a hole lowers coverage (%.4f)" % r["coverage"])
    ok(r["theoretical_sphere_voxels"] == base["theoretical_sphere_voxels"],
       "without changing the denominator")
    ok(base["valid_sphere_voxels"] - r["valid_sphere_voxels"] == 27,
       "exactly the punched voxels are lost (%d)"
       % (base["valid_sphere_voxels"] - r["valid_sphere_voxels"]))
    ok(not r["mask"][15, 15, 15], "and the returned mask excludes them")


def t_geometry():
    print("4. anisotropic voxels, rotation, off-centre spheres")
    shape = (41, 33, 25)
    A = aff(vox=(1.5, 2.0, 3.5), off=(-30.0, -32.0, -42.0))
    c = centre_of(A, shape)
    valid = np.ones(shape, dtype=bool)
    r = sphere_support(A, shape, c, 0.03, valid)
    ok(r["coverage"] == 1.0, "anisotropic voxels: still exactly 1.0")

    Ar = aff(vox=(2.0, 2.0, 2.0), off=(-30.0, -30.0, -30.0), rot_deg=37.0)
    cr = centre_of(Ar, (31, 31, 31))
    rr = sphere_support(Ar, (31, 31, 31), cr, 0.03,
                        np.ones((31, 31, 31), dtype=bool))
    ok(rr["coverage"] == 1.0, "rotated affine: still exactly 1.0")
    ok(rr["theoretical_sphere_voxels"] > 0, "and a non-empty sphere")

    # A centre deliberately between voxel centres.
    off_c = c + np.array([0.0009, -0.0007, 0.0011])
    ro = sphere_support(A, shape, off_c, 0.02, valid)
    ok(ro["coverage"] == 1.0, "centre off the lattice: still exactly 1.0")
    ok(ro["mask"].sum() == ro["valid_sphere_voxels"],
       "the mask and the count agree")


def t_bounds():
    print("5. coverage stays in [0, 1] and the mask is consistent")
    shape = (21, 21, 21)
    A = aff()
    valid = np.ones(shape, dtype=bool)
    pts = world_coordinates(A, shape)
    for c in (pts.mean(axis=0), pts[0], pts[-1],
              pts.mean(axis=0) + np.array([0.5, 0.0, 0.0])):
        for dsv in (0.005, 0.03, 0.5):
            r = sphere_support(A, shape, c, dsv, valid)
            ok(0.0 <= r["coverage"] <= 1.0,
               "coverage in range for centre %s, DSV %.0f cm (%.4f)"
               % (np.round(c, 3), 100 * dsv, r["coverage"]))
            ok(int(r["mask"].sum()) == r["valid_sphere_voxels"],
               "mask count matches the numerator")

    # A sphere entirely outside the array: nothing valid, but the denominator
    # is still the sphere.
    far = pts.mean(axis=0) + np.array([1.0, 0.0, 0.0])
    r = sphere_support(A, shape, far, 0.02, valid)
    ok(r["valid_sphere_voxels"] == 0 and r["theoretical_sphere_voxels"] > 0,
       "a sphere off the array has coverage 0 (%d/%d)"
       % (r["valid_sphere_voxels"], r["theoretical_sphere_voxels"]))
    ok(r["coverage"] == 0.0 and r["truncated"], "and is truncated")


def t_chunking():
    print("6. chunk size changes nothing")
    shape = (25, 25, 25)
    A = aff()
    c = centre_of(A, shape)
    valid = np.ones(shape, dtype=bool)
    valid[5:9, 5:9, 5:9] = False
    ref = sphere_support(A, shape, c, 0.06, valid, chunk=64)
    for ch in (3, 7, 16, 1024):
        r = sphere_support(A, shape, c, 0.06, valid, chunk=ch)
        ok(r["valid_sphere_voxels"] == ref["valid_sphere_voxels"]
           and r["theoretical_sphere_voxels"] == ref["theoretical_sphere_voxels"],
           "chunk=%d gives the same counts" % ch)


if __name__ == "__main__":
    for fn in (t_exact_one, t_outside_fov, t_holes, t_geometry, t_bounds,
               t_chunking):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
