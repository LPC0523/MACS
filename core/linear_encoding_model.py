"""Can a multi-coil array synthesise a linear gradient? Offline, numbers only.

IDEALIZED OFFLINE ENCODING SIMULATION. This module answers one question --
whether a set of measured coil field maps can be combined into a field that is
close to linear, and what currents that would take. It does not synchronise to
a scanner, does not model an amplifier, does not touch hardware, and proves
nothing about imaging speed.

The separable representation
----------------------------
The field is linear in the currents, so it never has to be stored as a volume
per time point:

    B(r, t) = C(r, :) i(t)

`C` is (n_voxels, n_channels) in Hz/A and fixed; `i` is (n_time, n_channels) in
amperes. A million voxels by two thousand samples would be 16 GB as float64 and
is completely unnecessary: the pair is 100 MB and a frame at any chosen time is
one matrix-vector product away. Phase follows the same trick --

    q(t) = integral of i    (A.s)          phase_cycles(r, t) = C(r, :) q(t)

so the time integral is done once on the J current channels rather than once
per voxel. Nothing in this module allocates an (Nx, Ny, Nz, Nt) array, and
`test_linear_encoding_model.py` asserts it.

Linear decomposition
--------------------
Each channel's map is split, inside the ROI and about the ROI's weighted
centre, against the shared basis [1, x-x0, y-y0, z-z0]:

    C_j(r) = f0_j + gx_j (x-x0) + gy_j (y-y0) + gz_j (z-z0) + R_j(r)

    f0_j            Hz/A        constant component
    gx_j gy_j gz_j  Hz/m/A      linear component
    R_j(r)          Hz/A        nonlinear spatial component

Three components, and only the first is inert:

* the **constant component** carries no spatial position information at all.
  It advances every spin's phase by the same amount, so it is separated out and
  reported but never counted towards encoding -- otherwise a coil would look
  useful merely because it can shift the whole head's frequency.
* the **linear component** is the part expressible as a conventional linear
  k-space offset, which is why it gets its own basis and its own waveform.
* the **nonlinear spatial component** also carries spatial information. It is
  not noise and not waste. What it cannot do is be written as a single
  Cartesian k vector, so using it would require a forward model and a
  model-based reconstruction rather than an FFT. This module measures it and
  makes no claim either way about whether it helps.

Once decomposed, any current history gives the gradient waveform for free:

    g(t) = G i(t)        (3, n_time) in Hz/m,  G = [gx; gy; gz]

Field-map units
---------------
The maps are used as Hz/A. That is what `coil_registry` says in a comment and
what the magnitudes look like, but no calibration current is recorded anywhere
in the project and no file carries a unit. `unit_status` must therefore be
passed in explicitly and is recorded in every result; the caller confirms it,
this module does not guess it.
"""

from __future__ import annotations

import numpy as np

from core.linear_encoding_waveform import check_time_axis, cumulative_integral
from core.spatial_grid import direction_field, unit_vector, weighted_centre

__all__ = ["UnitStatus", "CoilEncodingModel", "EncodingSimulation",
           "build_model", "simulate_currents", "simulate_target_gradient"]

#: How the caller vouches for the field-map unit. Anything other than
#: `user_confirmed` blocks a formal run; the data can still be inspected.
class UnitStatus:
    USER_CONFIRMED = "user_confirmed"
    UNCONFIRMED = "unconfirmed"
    ALL = (USER_CONFIRMED, UNCONFIRMED)


def _check_maps(maps, mask, affine):
    """Validate and flatten the coil maps onto the ROI. Returns (C, mask)."""
    m = np.asarray(mask, dtype=bool)
    a = np.asarray(maps)
    if a.ndim != 4:
        raise ValueError("maps must be (X, Y, Z, channels), got %d dims" % a.ndim)
    if a.shape[:3] != m.shape:
        raise ValueError("maps grid %s does not match mask %s"
                         % (a.shape[:3], m.shape))
    if int(m.sum()) == 0:
        raise ValueError("the ROI is empty: no voxel is selected")
    if a.shape[3] == 0:
        raise ValueError("no channels")
    C = a[m].astype(np.float64)
    if not np.all(np.isfinite(C)):
        n = int((~np.isfinite(C)).sum())
        raise ValueError("coil maps contain %d non-finite values inside the "
                         "ROI -- refusing to fit to NaN" % n)
    aff = np.asarray(affine, dtype=np.float64)
    if aff.shape != (4, 4) or not np.all(np.isfinite(aff)):
        raise ValueError("affine must be a finite 4x4 matrix")
    return C, m


class CoilEncodingModel:
    """A coil array decomposed into constant, linear and nonlinear parts.

    Everything is expressed inside one ROI and about one origin. Both are part
    of the answer, not incidental: the same array can look linear over a small
    frontal patch and hopeless over a whole brain.
    """

    def __init__(self, C, points, weights, centre, channel_names,
                 affine, mask, unit_status, unit_label="Hz/A"):
        self.C = C                              # (n_roi, J) Hz/A
        self.points = points                    # (n_roi, 3) m
        self.weights = weights                  # (n_roi,)
        self.centre = centre                    # (3,) m -- r0
        self.channel_names = list(channel_names)
        self.affine = np.asarray(affine, dtype=np.float64)
        self.mask = mask
        self.unit_status = unit_status
        self.unit_label = unit_label

        w = self.weights
        self.w_sum = float(w.sum())

        # Shared basis, about the weighted centre so the linear columns are
        # weighted-orthogonal to the constant one.
        d = self.points - self.centre
        self.basis = np.column_stack([np.ones(d.shape[0]), d[:, 0], d[:, 1],
                                      d[:, 2]])

        Wb = self.basis * w[:, None]
        gram = self.basis.T @ Wb
        coeff, *_ = np.linalg.lstsq(gram, Wb.T @ self.C, rcond=None)
        self.f0 = coeff[0]                      # (J,)   Hz/A
        self.G = coeff[1:]                      # (3, J) Hz/m/A
        self.residual = self.C - self.basis @ coeff     # (n_roi, J) Hz/A

        # Weighted Gram of the ROI-demeaned maps: what the array can span once
        # the uniform component is taken out.
        self.C_demeaned = self.C - (w @ self.C) / self.w_sum
        self._svals = None

    # ── properties of the array ─────────────────────────────────────
    @property
    def n_channels(self) -> int:
        return self.C.shape[1]

    @property
    def n_voxels(self) -> int:
        return self.C.shape[0]

    def singular_values(self) -> np.ndarray:
        """Singular values of the weighted, ROI-demeaned coil basis.

        The count of physical channels is not the number of spatial patterns
        the array can produce. Channels whose maps look alike add amplitude,
        not new shapes, and only the spectrum says which case this is.
        """
        if self._svals is None:
            Cw = self.C_demeaned * np.sqrt(self.weights)[:, None]
            self._svals = np.linalg.svd(Cw, compute_uv=False)
        return self._svals

    def effective_rank(self, tol_ratio: float = 1e-3) -> int:
        s = self.singular_values()
        if s.size == 0 or s[0] <= 0:
            return 0
        return int(np.sum(s > tol_ratio * s[0]))

    def channel_table(self) -> list:
        """Per channel: constant, the three gradient coefficients, residual."""
        w = self.weights
        out = []
        for j in range(self.n_channels):
            r = self.residual[:, j]
            out.append({
                "channel": self.channel_names[j],
                "constant_hz_per_a": float(self.f0[j]),
                "gx_hz_per_m_per_a": float(self.G[0, j]),
                "gy_hz_per_m_per_a": float(self.G[1, j]),
                "gz_hz_per_m_per_a": float(self.G[2, j]),
                "gradient_norm_hz_per_m_per_a": float(np.linalg.norm(self.G[:, j])),
                "nonlinear_rms_hz_per_a": float(
                    np.sqrt(np.sum(w * r ** 2) / self.w_sum)),
                "linear_fraction": _linear_fraction(self.C_demeaned[:, j],
                                                    r, w),
            })
        return out

    # ── synthesis ───────────────────────────────────────────────────
    def unit_gradient_currents(self, direction, ridge: float = 0.0):
        """Minimum-norm currents for a 1 Hz/m gradient along `direction`.

        Returns (a, info). `a` is in A per (Hz/m): scale it by a gradient
        strength in Hz/m to get amperes. No bound is applied here -- feasibility
        against +-3 A is a separate question answered by the constraint module,
        because the unconstrained answer is what says whether the *shape* is
        reachable at all.

        Both the target and the coil maps are demeaned over the ROI first, so
        the fit cannot spend amplitude on a uniform offset that encodes nothing.
        """
        d = unit_vector(direction)
        t = direction_field(self.points, d, centre=self.centre)   # metres
        w = self.weights
        t = t - float(w @ t) / self.w_sum

        sw = np.sqrt(w)[:, None]
        A = self.C_demeaned * sw
        b = t * np.sqrt(w)
        if ridge > 0:
            A = np.vstack([A, np.sqrt(ridge) * np.eye(self.n_channels)])
            b = np.concatenate([b, np.zeros(self.n_channels)])
        a, *_ = np.linalg.lstsq(A, b, rcond=None)

        achieved = self.G @ a                    # (3,) Hz/m per (Hz/m) request
        return a, {
            "direction": d.tolist(),
            "ridge": float(ridge),
            "achieved_gradient_vector": achieved.tolist(),
            "gradient_gain": float(achieved @ d),
            "constant_hz_per_unit": float(self.f0 @ a),
            "current_norm_a_per_hz_per_m": float(np.linalg.norm(a)),
        }

    # ── on-demand frames, one at a time ─────────────────────────────
    def field_at(self, currents) -> np.ndarray:
        """(n_roi,) field in Hz for one current vector. One frame, never all."""
        i = np.asarray(currents, dtype=np.float64).ravel()
        if i.shape[0] != self.n_channels:
            raise ValueError("expected %d currents, got %d"
                             % (self.n_channels, i.shape[0]))
        return self.C @ i

    def linear_part_at(self, currents) -> np.ndarray:
        i = np.asarray(currents, dtype=np.float64).ravel()
        return self.basis @ np.concatenate([[self.f0 @ i], self.G @ i])

    def residual_at(self, currents) -> np.ndarray:
        i = np.asarray(currents, dtype=np.float64).ravel()
        return self.residual @ i

    def phase_cycles_at(self, q) -> np.ndarray:
        """(n_roi,) accumulated phase in cycles for one integrated-current
        vector `q` (A.s). Multiply by 2*pi for radians."""
        v = np.asarray(q, dtype=np.float64).ravel()
        if v.shape[0] != self.n_channels:
            raise ValueError("expected %d integrated currents, got %d"
                             % (self.n_channels, v.shape[0]))
        return self.C @ v

    def scatter(self, values, fill=np.nan) -> np.ndarray:
        """One ROI vector placed back into a full volume. A single 3-D array."""
        v = np.asarray(values, dtype=np.float64).ravel()
        if v.shape[0] != self.n_voxels:
            raise ValueError("expected %d values, got %d"
                             % (self.n_voxels, v.shape[0]))
        out = np.full(self.mask.shape, fill, dtype=np.float64)
        out[self.mask] = v
        return out


def _linear_fraction(total_demeaned, residual, w):
    """Share of a channel's ROI variance explained by the linear terms."""
    num = float(np.sum(w * residual ** 2))
    den = float(np.sum(w * total_demeaned ** 2))
    if den <= 0:
        return float("nan")
    return float(max(0.0, 1.0 - num / den))


def build_model(maps, mask, affine, channel_names=None, weights=None,
                unit_status=UnitStatus.UNCONFIRMED,
                unit_label="Hz/A") -> CoilEncodingModel:
    """Decompose a coil array over one ROI. `maps` is (X, Y, Z, channels) Hz/A.

    `unit_status` is not defaulted to confirmed on purpose. The maps carry no
    unit and no calibration current is recorded anywhere in this project, so
    the interpretation is the caller's assertion and travels with the result.
    """
    from core.spatial_grid import world_coordinates

    if unit_status not in UnitStatus.ALL:
        raise ValueError("unit_status must be one of %s" % (UnitStatus.ALL,))
    C, m = _check_maps(maps, mask, affine)
    pts = world_coordinates(affine, m.shape, mask=m)

    if weights is None:
        w = np.ones(C.shape[0], dtype=np.float64)
    else:
        wa = np.asarray(weights, dtype=np.float64)
        w = wa[m] if wa.shape == m.shape else wa.ravel()
        if w.shape[0] != C.shape[0]:
            raise ValueError("weights give %d values for %d ROI voxels"
                             % (w.shape[0], C.shape[0]))
        if np.any(w < 0) or not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite and non-negative")
        if w.sum() <= 0:
            raise ValueError("weights sum to zero")

    names = (list(channel_names) if channel_names is not None
             else ["ch%d" % (k + 1) for k in range(C.shape[1])])
    if len(names) != C.shape[1]:
        raise ValueError("%d channel names for %d channels"
                         % (len(names), C.shape[1]))

    return CoilEncodingModel(C, pts, w, weighted_centre(pts, w), names,
                             affine, m, unit_status, unit_label)


class EncodingSimulation:
    """The result of running one current history through one model.

    Holds the separable pieces -- currents, their integral, the gradient and
    k-space waveforms -- and makes spatial frames on demand. It never holds a
    field volume per time point.
    """

    def __init__(self, model, t, currents, static_currents, meta=None):
        self.model = model
        self.t = check_time_axis(t)
        self.i = np.asarray(currents, dtype=np.float64)
        if self.i.ndim != 2 or self.i.shape[0] != self.t.shape[0]:
            raise ValueError("currents must be (n_time, n_channels) matching t")
        if self.i.shape[1] != model.n_channels:
            raise ValueError("currents have %d channels, model has %d"
                             % (self.i.shape[1], model.n_channels))
        if not np.all(np.isfinite(self.i)):
            raise ValueError("currents contain non-finite values")
        self.i_static = np.asarray(static_currents, dtype=np.float64).ravel()
        self.meta = dict(meta or {})

        # ── dynamic vs total ─────────────────────────────────────────
        # `self.i` is the TOTAL current, static baseline included, because that
        # is what an amplifier sees. But the encoding question is about what
        # the *added* waveform does, and a static shim contributes a constant
        # gradient that would otherwise be silently folded into every k number.
        # Both are kept, named, and never mixed.
        self.i_total = self.i
        self.i_dynamic = self.i_total - self.i_static

        self.q_total = cumulative_integral(self.t, self.i_total)
        self.q_dynamic = cumulative_integral(self.t, self.i_dynamic)
        self.q = self.q_total                    # legacy alias, total

        self.g_total = self.i_total @ model.G.T          # (T, 3) Hz/m
        self.g_dynamic = self.i_dynamic @ model.G.T
        self.g = self.g_total                    # legacy alias, total

        self.f0_t = self.i_total @ model.f0              # (T,) Hz

        # Idealised k offsets, cycles/m. Computed from q rather than by
        # integrating g again, so the two can never disagree by a rounding
        # rule. Neither includes the scanner's own imaging gradients.
        self.k_dynamic = self.q_dynamic @ model.G.T
        self.k_total_local_model = self.q_total @ model.G.T
        self.k = self.k_total_local_model        # legacy alias, total

    # ── frames ──────────────────────────────────────────────────────
    def index_at(self, time_s: float) -> int:
        return int(np.argmin(np.abs(self.t - float(time_s))))

    def field_frame(self, idx: int) -> np.ndarray:
        return self.model.field_at(self.i[int(idx)])

    def residual_frame(self, idx: int) -> np.ndarray:
        return self.model.residual_at(self.i[int(idx)])

    def phase_frame_cycles(self, idx: int) -> np.ndarray:
        return self.model.phase_cycles_at(self.q_total[int(idx)])

    def phase_frame_rad(self, idx: int) -> np.ndarray:
        return 2.0 * np.pi * self.phase_frame_cycles(idx)

    def phase_frame_dynamic_rad(self, idx: int) -> np.ndarray:
        """Phase from the added waveform alone, radians."""
        return 2.0 * np.pi * self.model.phase_cycles_at(
            self.q_dynamic[int(idx)])

    def phase_frame_nonlinear_dynamic_rad(self, idx: int) -> np.ndarray:
        """Phase from the dynamic nonlinear spatial component alone.

        This is the part that cannot be written as a k offset; it is separated
        so it is never quietly counted as one.
        """
        return 2.0 * np.pi * (self.model.residual @ self.q_dynamic[int(idx)])

    def field_frame_dynamic(self, idx: int) -> np.ndarray:
        return self.model.field_at(self.i_dynamic[int(idx)])

    def residual_frame_dynamic(self, idx: int) -> np.ndarray:
        return self.model.residual_at(self.i_dynamic[int(idx)])

    # ── summaries that never build a 4-D array ──────────────────────
    def peak_index(self) -> int:
        """Where the added gradient is strongest -- the frame worth looking at.

        Uses the dynamic part: with a static baseline the total gradient can be
        largest at t=0 before anything has happened.
        """
        return int(np.argmax(np.linalg.norm(self.g_dynamic, axis=1)))

    def k_max(self) -> float:
        return float(np.linalg.norm(self.k, axis=1).max())

    def returns_to_origin(self, tol_ratio: float = 1e-3) -> dict:
        """Does the trajectory close? A waveform with a non-zero mean leaves a
        net k offset, which is a real effect and not a rounding error."""
        end = float(np.linalg.norm(self.k[-1]))
        peak = self.k_max()
        rel = end / peak if peak > 0 else 0.0
        return {"k_end_cycles_per_m": end, "k_max_cycles_per_m": peak,
                "relative": float(rel), "closed": bool(rel <= tol_ratio)}


def simulate_currents(model, t, currents, static_currents=None,
                      meta=None) -> EncodingSimulation:
    """Mode B: the caller supplies the per-channel current history directly."""
    i_static = (np.zeros(model.n_channels) if static_currents is None
                else np.asarray(static_currents, dtype=np.float64).ravel())
    if i_static.shape[0] != model.n_channels:
        raise ValueError("static currents have %d entries for %d channels"
                         % (i_static.shape[0], model.n_channels))
    total = np.asarray(currents, dtype=np.float64) + i_static
    m = {"mode": "direct_channel_currents"}
    m.update(meta or {})
    return EncodingSimulation(model, t, total, i_static, m)


def simulate_target_gradient(model, t, g_of_t, direction,
                             static_currents=None, ridge: float = 0.0,
                             meta=None):
    """Mode A: a scalar gradient waveform along one RAS direction.

    `g_of_t` is in Hz/m. Returns (simulation, unit_info); `unit_info` carries
    the per-(Hz/m) currents so the caller can see what one unit of gradient
    costs before any scaling is applied.
    """
    a, info = model.unit_gradient_currents(direction, ridge=ridge)
    g = np.asarray(g_of_t, dtype=np.float64).ravel()
    t = check_time_axis(t)
    if g.shape[0] != t.shape[0]:
        raise ValueError("gradient waveform has %d samples for %d times"
                         % (g.shape[0], t.shape[0]))
    encoding = g[:, None] * a[None, :]           # (T, J) amperes
    m = {"mode": "target_linear_gradient",
         "direction": info["direction"],
         "unit_currents_a_per_hz_per_m": a.tolist(),
         "requested_peak_gradient_hz_per_m": float(np.abs(g).max())}
    m.update(meta or {})
    sim = simulate_currents(model, t, encoding, static_currents, m)
    return sim, info
