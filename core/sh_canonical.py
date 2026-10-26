"""One spherical-harmonic basis, shared by the app and by the SF_v2 comparison.

`SF_v2/standalone/sh_basis.py` builds its second order from `z*z`. That is not a
solid harmonic -- its Laplacian is 2, not 0 -- and it spans a different subspace
from the one a scanner's Z2 shim actually drives. Measured over random points in
a 10 cm ball, seven of the eight principal angles between the two subspaces are
zero and the eighth is 34.9 degrees, and only 43% of the true `2z^2-x^2-y^2`
direction lies inside the SF_v2 span.

The consequence is not academic. Projecting out the SF_v2 basis leaves more than
half of the scanner's real Z2 capability sitting in what is then called the
"high-order residual", so a coil scored against it is being credited (or
blamed) for field the scanner could have produced itself. Any figure computed
that way -- including the 0-0.15% high-order improvement -- describes a
different question from the one the app asks.

This module is therefore the single definition. It matches
`hybrid_shim.build_sh_basis` term for term; the difference is only that this
one takes an (N, 3) point list rather than an affine and a shape, so the same
geometry can be evaluated on a voxel grid or on a mesh's field points.

Calibration factors are deliberately *not* applied here. They convert a console
setting into field and belong to the machine, not to the geometry; a subspace
comparison must not depend on them, and applying them would silently rescale
each axis of the basis.
"""

from __future__ import annotations

import numpy as np

__all__ = ["SH_TERMS", "sh_matrix", "sh_projector_apply", "project_out",
           "principal_angles_deg", "subspace_overlap"]

#: Solid harmonics through second order, in the app's order. Zeroth order is
#: excluded: it is the centre frequency, handled separately as a free f0.
SH_TERMS = ("A10", "A11", "B11", "A20", "A21", "B21", "A22", "B22")


def _terms(x, y, z):
    return {
        "A10": z,
        "A11": x,
        "B11": y,
        "A20": 2.0 * z * z - x * x - y * y,      # NOT z*z
        "A21": x * z,
        "B21": y * z,
        "A22": x * x - y * y,
        "B22": 2.0 * x * y,
    }


def sh_matrix(points, order: int = 2, center=None, names=None,
              orthonormalise: bool = True):
    """(N, K) basis evaluated at `points` (metres).

    `orthonormalise` returns a QR factor, which makes `Q @ Q.T` the orthogonal
    projector and the columns comparable in scale. Turn it off when the raw
    physical amplitudes are wanted.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {pts.shape}")

    c = np.asarray(center, dtype=float) if center is not None else pts.mean(axis=0)
    x, y, z = (pts[:, 0] - c[0]), (pts[:, 1] - c[1]), (pts[:, 2] - c[2])

    if names is None:
        names = [n for n in SH_TERMS
                 if (order >= 1 and n in ("A10", "A11", "B11"))
                 or (order >= 2 and n not in ("A10", "A11", "B11"))]
    tbl = _terms(x, y, z)
    raw = np.column_stack([tbl[n] for n in names])

    if not orthonormalise:
        return raw, list(names)
    q, _ = np.linalg.qr(raw)
    return q, list(names)


def grid_points(affine, shape, mask=None):
    """World coordinates (metres) of voxel centres, optionally masked.

    Uses the same convention as `hybrid_shim.build_sh_basis`: index -> affine ->
    millimetres -> metres.
    """
    s = tuple(int(v) for v in shape[:3])
    ii, jj, kk = np.meshgrid(np.arange(s[0]), np.arange(s[1]), np.arange(s[2]),
                             indexing="ij")
    ijk = np.stack([ii, jj, kk, np.ones_like(ii)], axis=-1).astype(float)
    w = np.einsum("ij,...j->...i", np.asarray(affine, dtype=float), ijk)[..., :3]
    w = w / 1000.0
    if mask is None:
        return w.reshape(-1, 3)
    m = np.asarray(mask, dtype=bool)
    return w[m]


def sh_projector_apply(Q, v):
    """Project `v` onto the span of orthonormal `Q`, without forming Q Q^T."""
    return Q @ (Q.T @ v)


def project_out(Q, v):
    """(I - Q Q^T) v -- the part of `v` no shim setting can produce."""
    return v - sh_projector_apply(Q, v)


def principal_angles_deg(Qa, Qb):
    """Principal angles (degrees) between two orthonormal column spans."""
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def subspace_overlap(Q, v):
    """Fraction of `v`'s norm that lies inside span(Q), in [0, 1]."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        return 0.0
    return float(np.linalg.norm(sh_projector_apply(Q, v)) / n)
