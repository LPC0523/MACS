"""Voxel centres in world coordinates, in metres. One implementation.

The convention is nibabel's: an affine maps voxel index to RAS+ millimetres,
so world x is right, y is anterior, z is superior. Everything here divides by
1000 so downstream code is in metres and a gradient in Hz/m means what it says.

There is no scanner frame anywhere in this project. The affine gives the RAS
world frame and nothing more: `PatientPosition` and `PhaseEncodingDirection`
are not read anywhere, so RAS-Y is *not* known to be the phase-encoding
direction, and nothing here may present it as one.

Why this module exists
----------------------
Four places already turn an affine into world coordinates -- `data_context`
via `nib.affines.apply_affine`, `hybrid_shim.build_sh_basis` and
`sh_canonical.grid_points` via einsum, `physics` via a matrix product. They
agree, but a fifth copy written inside a new feature would eventually not.
Production is frozen, so those are left alone; `test_spatial_grid.py` asserts
this module reproduces them exactly, which turns "four copies" into "one
implementation with three verified mirrors".

The axis-vs-direction trap
--------------------------
Array axis 0 is not world x. With an oblique or rotated affine they are not
even close, and with a plain axial acquisition axis 0 is usually *minus* x.
Only `world_coordinates` may be used to build a direction field; taking
`np.gradient(..., axis=0)` and calling it Gx is wrong and this module offers
no such shortcut.
"""

from __future__ import annotations

import numpy as np

__all__ = ["world_coordinates", "weighted_centre", "direction_field",
           "unit_vector", "sphere_support", "RAS_AXES", "MM_PER_M",
           "COVERAGE_THRESHOLD"]

MM_PER_M = 1000.0

#: Unit direction vectors in the RAS world frame. Named for the anatomical
#: axis, never for a scanner gradient axis.
RAS_AXES = {
    "RAS-X": np.array([1.0, 0.0, 0.0]),      # right (+) / left (-)
    "RAS-Y": np.array([0.0, 1.0, 0.0]),      # anterior (+) / posterior (-)
    "RAS-Z": np.array([0.0, 0.0, 1.0]),      # superior (+) / inferior (-)
}


def world_coordinates(affine, shape, mask=None):
    """(N, 3) world coordinates of voxel centres, in metres.

    With `mask`, only the True voxels are returned, in C order -- the same
    order `arr[mask]` gives, so a field sampled with the mask lines up row for
    row with these coordinates.
    """
    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError("affine must be 4x4, got %s" % (affine.shape,))
    if not np.all(np.isfinite(affine)):
        raise ValueError("affine contains non-finite values")

    s = tuple(int(v) for v in tuple(shape)[:3])
    ii, jj, kk = np.meshgrid(np.arange(s[0]), np.arange(s[1]),
                             np.arange(s[2]), indexing="ij")
    ijk = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).astype(np.float64)
    w = np.einsum("ij,...j->...i", affine, ijk)[..., :3] / MM_PER_M

    if mask is None:
        return w.reshape(-1, 3)
    m = np.asarray(mask, dtype=bool)
    if m.shape != s:
        raise ValueError("mask shape %s does not match grid %s" % (m.shape, s))
    return w[m]


def weighted_centre(points, weights=None):
    """(3,) weighted centroid of `points`, in the same units.

    This is `r0`. Subtracting it is not cosmetic: a linear field measured about
    the wrong origin carries a constant offset, and a constant field shifts
    every spin's phase identically, so it encodes no position at all. Putting
    the origin at the centre of the region being encoded keeps that offset out
    of the fit instead of letting it soak up amplitude.
    """
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("points must be (N, 3), got %s" % (p.shape,))
    if p.shape[0] == 0:
        raise ValueError("no points: the ROI is empty")
    if weights is None:
        return p.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != p.shape[0]:
        raise ValueError("weights length %d does not match %d points"
                         % (w.shape[0], p.shape[0]))
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    tot = float(w.sum())
    if tot <= 0:
        raise ValueError("weights sum to zero")
    return (w[:, None] * p).sum(axis=0) / tot


def unit_vector(d):
    """`d` normalised. Raises on a zero or non-finite vector."""
    v = np.asarray(d, dtype=np.float64).ravel()
    if v.shape != (3,):
        raise ValueError("direction must have 3 components, got %s" % (v.shape,))
    if not np.all(np.isfinite(v)):
        raise ValueError("direction contains non-finite values")
    n = float(np.linalg.norm(v))
    if n <= 0:
        raise ValueError("direction has zero length")
    return v / n


def direction_field(points, direction, centre=None, weights=None):
    """(N,) of `d . (r - r0)` in metres -- the shape of a unit linear gradient.

    Multiply by a gradient strength in Hz/m to get a target field in Hz.
    """
    p = np.asarray(points, dtype=np.float64)
    d = unit_vector(direction)
    r0 = weighted_centre(p, weights) if centre is None else \
        np.asarray(centre, dtype=np.float64).ravel()
    return (p - r0) @ d


#: A sphere is only a DSV if the calibration actually covers it. Below this
#: fraction the region is the intersection of a sphere with whatever support
#: exists, which is a different and smaller thing, and a capability quoted over
#: it would be quoted over a shape nobody chose.
COVERAGE_THRESHOLD = 0.95


def _sphere_index_box(affine, centre_m, radius_m):
    """Index-space bounding box of a world sphere. May extend past the array.

    The sphere maps to an ellipsoid in index space, whose half-extent along
    index axis k is `radius * ||row k of inv(A)||`. The box is deliberately
    allowed to run outside the array: that is the whole point -- a sphere
    sticking out of the field of view has to be counted as sticking out, not
    silently trimmed to fit.
    """
    A = np.asarray(affine, dtype=np.float64)
    inv = np.linalg.inv(A)
    c_mm = np.asarray(centre_m, dtype=np.float64) * MM_PER_M
    r_mm = float(radius_m) * MM_PER_M
    centre_idx = (inv @ np.append(c_mm, 1.0))[:3]
    half = r_mm * np.linalg.norm(inv[:3, :3], axis=1)
    lo = np.floor(centre_idx - half).astype(np.int64) - 1
    hi = np.ceil(centre_idx + half).astype(np.int64) + 2
    return lo, hi


def sphere_support(affine, shape, centre, dsv_m, valid_mask=None,
                   chunk=64) -> dict:
    """A spherical region and how much of it the data actually covers.

    Both counts are taken on the *same* lattice, by the same rule: a voxel
    counts if its centre lies inside the sphere.

        theoretical  every lattice site inside the sphere, including sites
                     that fall outside the array
        valid        those sites that are also inside the array and True in
                     `valid_mask`

    Counting only sites inside the array would call a sphere larger than the
    field of view "fully covered" -- every voxel it contains is a voxel it
    contains, and the part hanging outside would never be counted as missing.
    Dividing a continuous sphere volume by a voxel volume would mix two
    different measures and could not give exactly 1.0 for a sphere that is
    genuinely fully sampled. Both counts here are discrete site counts, so a
    covered sphere reports exactly 1.0.

    The index box is walked in chunks, so a 40 cm sphere on a fine grid never
    materialises a full coordinate array.

    Nothing hard-codes a diameter for any array: the answer follows from the
    maps that are loaded, so a wider calibration enlarges the usable DSV by
    itself. `truncated` is True below `COVERAGE_THRESHOLD`; such a region is
    still legitimate to inspect, it is just not a DSV, and the caller should
    name it `sphere and valid support` rather than quote a capability over it.
    """
    A = np.asarray(affine, dtype=np.float64)
    if A.shape != (4, 4) or not np.all(np.isfinite(A)):
        raise ValueError("affine must be a finite 4x4 matrix")
    if abs(float(np.linalg.det(A[:3, :3]))) <= 0:
        raise ValueError("affine is singular; the lattice is degenerate")
    shp = tuple(int(v) for v in tuple(shape)[:3])
    c = np.asarray(centre, dtype=np.float64).ravel()
    if c.shape != (3,):
        raise ValueError("centre must have 3 components")
    r = float(dsv_m) / 2.0
    if r <= 0:
        raise ValueError("dsv_m must be positive")
    if valid_mask is not None:
        v = np.asarray(valid_mask, dtype=bool)
        if v.shape != shp:
            raise ValueError("valid_mask shape %s does not match grid %s"
                             % (v.shape, shp))
    else:
        v = None

    lo, hi = _sphere_index_box(A, c, r)
    c_mm = c * MM_PER_M
    r_mm = r * MM_PER_M
    r2 = r_mm * r_mm

    theoretical = 0
    valid = 0
    mask = np.zeros(shp, dtype=bool)

    for i0 in range(int(lo[0]), int(hi[0]), chunk):
        i1 = min(i0 + chunk, int(hi[0]))
        for j0 in range(int(lo[1]), int(hi[1]), chunk):
            j1 = min(j0 + chunk, int(hi[1]))
            for k0 in range(int(lo[2]), int(hi[2]), chunk):
                k1 = min(k0 + chunk, int(hi[2]))
                ii, jj, kk = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1),
                                         np.arange(k0, k1), indexing="ij")
                idx = np.stack([ii, jj, kk, np.ones_like(ii)],
                               axis=-1).astype(np.float64)
                w_mm = np.einsum("ab,...b->...a", A, idx)[..., :3]
                inside = ((w_mm - c_mm) ** 2).sum(axis=-1) <= r2
                if not inside.any():
                    continue
                theoretical += int(inside.sum())
                # The numerator adds the two further conditions: the site must
                # exist in the array, and be valid there.
                in_grid = ((ii >= 0) & (ii < shp[0]) & (jj >= 0)
                           & (jj < shp[1]) & (kk >= 0) & (kk < shp[2]))
                sel = inside & in_grid
                if not sel.any():
                    continue
                gi, gj, gk = ii[sel], jj[sel], kk[sel]
                if v is not None:
                    keep = v[gi, gj, gk]
                    gi, gj, gk = gi[keep], gj[keep], gk[keep]
                mask[gi, gj, gk] = True
                valid += int(gi.size)

    cov = (valid / theoretical) if theoretical else 0.0
    cov = float(min(max(cov, 0.0), 1.0))
    return {
        "mask": mask,
        "centre_m": c.tolist(),
        "dsv_m": float(dsv_m),
        "valid_sphere_voxels": int(valid),
        "theoretical_sphere_voxels": int(theoretical),
        "coverage": cov,
        "coverage_method": ("discrete lattice site count; numerator and "
                            "denominator use the same rule and the same "
                            "affine"),
        "truncated": bool(cov < COVERAGE_THRESHOLD),
        "region_name": ("DSV %.1f cm" % (100 * dsv_m) if cov >= COVERAGE_THRESHOLD
                        else "sphere %.1f cm and valid support" % (100 * dsv_m)),
    }
