"""The separable model: decomposition, synthesis, phase and k-space.

    D:\\python.exe tests/test_linear_encoding_model.py

Synthetic coils throughout, so every expected answer is known in closed form.
The point of these is that the physics identities hold exactly -- if a coil
*is* a perfect x gradient, the model must recover it to machine precision, and
if a waveform integrates to a known phase, the simulator must agree with the
analytic value rather than merely look plausible.
"""

from __future__ import annotations

import sys
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.linear_encoding_metrics import (  # noqa: E402
    encoding_metrics, gram_matrix)
from core.linear_encoding_model import (  # noqa: E402
    UnitStatus, build_model, simulate_currents, simulate_target_gradient)
from core.linear_encoding_waveform import WaveformSpec, generate  # noqa: E402
from core.spatial_grid import world_coordinates  # noqa: E402

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def close(a, b, tol, msg):
    d = abs(float(a) - float(b))
    ok(d <= tol, "%s  (|%.6g - %.6g| = %.3g <= %.3g)" % (msg, a, b, d, tol))


SHAPE = (12, 13, 14)
AFF = np.array([[2.0, 0, 0, -12.0],
                [0, 2.5, 0, -16.0],
                [0, 0, 3.0, -21.0],
                [0, 0, 0, 1.0]])


def synthetic(kinds, shape=SHAPE, affine=AFF):
    """Build coil maps from closures of world coordinates, in Hz/A."""
    pts = world_coordinates(affine, shape)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    maps = np.stack([f(x, y, z).reshape(shape) for f in kinds], axis=-1)
    return maps.astype(np.float64)


MASK = np.ones(SHAPE, dtype=bool)


# ── 1. a coil that is an exact X gradient is recovered exactly ──────
def t_exact_x():
    print("1. exact X-gradient coil is recovered")
    G0 = 200.0                       # Hz/m per amp
    maps = synthetic([lambda x, y, z: G0 * x,
                      lambda x, y, z: 50.0 * z,
                      lambda x, y, z: 30.0 * np.ones_like(x)])
    m = build_model(maps, MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)

    close(m.G[0, 0], G0, 1e-9, "channel 1 gx recovered")
    close(m.G[1, 0], 0.0, 1e-9, "channel 1 gy is zero")
    close(m.G[2, 1], 50.0, 1e-9, "channel 2 gz recovered")
    close(np.abs(m.residual).max(), 0.0, 1e-9, "purely linear coils leave no residual")
    close(m.f0[2], 30.0, 1e-9, "the uniform coil's constant is recovered")
    close(np.linalg.norm(m.G[:, 2]), 0.0, 1e-9,
          "the uniform coil contributes no gradient")

    a, info = m.unit_gradient_currents([1, 0, 0])
    close(info["gradient_gain"], 1.0, 1e-9, "unit X gradient is exactly reachable")
    close(a[0], 1.0 / G0, 1e-12, "unit current = 1/G0 on the X channel")
    close(abs(a[1]), 0.0, 1e-9, "no current on the Z channel")

    # The constant coil must not be used at all: it cannot help a gradient and
    # the demeaning is what stops the fit from spending amplitude on it.
    close(abs(a[2]), 0.0, 1e-9, "no current on the uniform coil")


# ── 2. two non-orthogonal coils combine with opposite signs ─────────
def t_non_orthogonal():
    print("2. two non-orthogonal coils reach a target by cancellation")
    # Neither coil is an x gradient; both are mixtures. Their difference is.
    maps = synthetic([lambda x, y, z: 100.0 * x + 80.0 * y,
                      lambda x, y, z: 100.0 * x + 80.0 * y - 60.0 * x])
    m = build_model(maps, MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)
    a, info = m.unit_gradient_currents([1, 0, 0])
    close(info["gradient_gain"], 1.0, 1e-6, "X gradient reached")
    ok(a[0] * a[1] < 0, "the two channels take opposite signs (%.4g, %.4g)"
       % (a[0], a[1]))
    g = m.G @ a
    close(g[1], 0.0, 1e-6, "the y component cancels")

    mm = encoding_metrics(m, a, direction=[1, 0, 0],
                          requested_gradient_hz_per_m=1.0)
    close(mm["nrmse"], 0.0, 1e-6, "the synthesised field matches the target")
    close(mm["direction_error_deg"], 0.0, 1e-6, "no direction error")


# ── 3. collinear coils are low rank ────────────────────────────────
def t_rank():
    print("3. collinear coil maps are detected as low rank")
    base = lambda x, y, z: 120.0 * x + 40.0 * z          # noqa: E731
    maps = synthetic([base, lambda x, y, z: 2.0 * base(x, y, z),
                      lambda x, y, z: -0.5 * base(x, y, z),
                      lambda x, y, z: 3.0 * base(x, y, z)])
    m = build_model(maps, MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)
    g = gram_matrix(m)
    ok(g["n_channels"] == 4, "four physical channels")
    ok(g["effective_rank_1e-3"] == 1,
       "but only one spatial pattern (effective rank %d)"
       % g["effective_rank_1e-3"])
    ok(g["singular_values_normalised"][1] < 1e-9,
       "the second singular value is numerically zero")

    # And a genuinely 3-D array is full rank, so the check discriminates.
    m3 = build_model(synthetic([lambda x, y, z: 100 * x,
                                lambda x, y, z: 90 * y,
                                lambda x, y, z: 80 * z]),
                     MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)
    ok(gram_matrix(m3)["effective_rank_1e-3"] == 3,
       "three independent coils give effective rank 3")


# ── 4. validation ──────────────────────────────────────────────────
def t_validation():
    print("4. inputs are validated, not patched up")
    maps = synthetic([lambda x, y, z: 100 * x, lambda x, y, z: 90 * y])
    for bad, why, frag in (
            (dict(mask=np.zeros(SHAPE, dtype=bool)), "empty ROI", "empty"),
            (dict(mask=np.ones((3, 3, 3), dtype=bool)), "mask shape", "does not match"),
            (dict(affine=np.eye(3)), "affine shape", "4x4"),
    ):
        kw = dict(maps=maps, mask=MASK, affine=AFF,
                  unit_status=UnitStatus.USER_CONFIRMED)
        kw.update(bad)
        try:
            build_model(**kw)
            ok(False, "%s should be rejected" % why)
        except ValueError as e:
            ok(frag in str(e), "%s rejected: %s" % (why, str(e)[:52]))

    nanmaps = maps.copy()
    nanmaps[0, 0, 0, 0] = np.nan
    try:
        build_model(nanmaps, MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)
        ok(False, "NaN should be rejected")
    except ValueError as e:
        ok("non-finite" in str(e), "NaN in the maps rejected")

    try:
        build_model(maps, MASK, AFF, unit_status="assumed")
        ok(False, "an invented unit_status should be rejected")
    except ValueError:
        ok(True, "unit_status must be one of the declared values")

    m = build_model(maps, MASK, AFF)
    ok(m.unit_status == UnitStatus.UNCONFIRMED,
       "unit_status defaults to unconfirmed, never to confirmed")


# ── 5. superposition, phase and k-space against analytic values ────
def t_physics():
    print("5. B = C i, phase = 2*pi*int B dt, k = int g dt")
    G0 = 250.0
    maps = synthetic([lambda x, y, z: G0 * x, lambda x, y, z: 90.0 * y])
    m = build_model(maps, MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)

    i1 = np.array([0.7, -0.3])
    i2 = np.array([0.1, 0.9])
    f1, f2 = m.field_at(i1), m.field_at(i2)
    ok(np.allclose(m.field_at(i1 + i2), f1 + f2, atol=1e-9),
       "the field is linear in the currents")
    ok(np.allclose(m.field_at(3.0 * i1), 3.0 * f1, atol=1e-9),
       "and homogeneous")

    A, f = 1.5, 200.0               # amps, Hz
    t, w = generate(WaveformSpec("sine", amplitude=1.0, frequency=f,
                                 n_cycles=3, sample_rate_hz=400_000.0))
    cur = np.stack([A * w, np.zeros_like(w)], axis=1)
    sim = simulate_currents(m, t, cur)

    # Gradient waveform: only the x channel is driven, so g = G0 * i(t).
    ok(np.allclose(sim.g[:, 0], G0 * A * w, atol=1e-9), "gx(t) = G0 i(t)")
    ok(np.allclose(sim.g[:, 1], 0.0, atol=1e-9), "gy stays zero")

    # k = int g dt has the analytic form G0*A*(1-cos wt)/w, in cycles/m.
    om = 2 * np.pi * f
    k_ana = G0 * A * (1 - np.cos(om * t)) / om
    close(np.abs(sim.k[:, 0] - k_ana).max(), 0.0, 1e-6,
          "kx(t) matches the analytic integral, in cycles/m")

    # Phase at one voxel: 2*pi * C(r) . q(t), with C the *total* field. The
    # absolute world x is what belongs here, not x - x0: a uniform field does
    # advance every spin's phase, it just advances them all equally, so it
    # carries no position information. Demeaning is for judging encoding
    # quality; phase is physical and keeps the constant.
    idx = int(np.argmax(m.points[:, 0]))
    x_abs = m.points[idx, 0]
    j = sim.index_at(t[-1] / 2)
    phi = sim.phase_frame_rad(j)[idx]
    q_ana = A * (1 - np.cos(om * t[j])) / om
    close(phi, 2 * np.pi * G0 * x_abs * q_ana, 1e-6,
          "phase at one voxel matches 2*pi*G0*x*int i dt (absolute x)")
    # And the demeaned encoding view differs from it by exactly the constant.
    const_phase = 2 * np.pi * float(m.f0 @ sim.q[j])
    close(phi - const_phase,
          2 * np.pi * G0 * (x_abs - m.centre[0]) * q_ana, 1e-6,
          "removing the constant leaves the position-dependent phase")

    # A whole number of cycles must bring k back to the origin.
    r = sim.returns_to_origin()
    ok(r["closed"], "three whole cycles return k to the origin (%.3g of peak)"
       % r["relative"])

    # A cosine has a non-zero integral over whole cycles only through its
    # endpoints; a sine with an offset genuinely does not close.
    _, w2 = generate(WaveformSpec("sine", amplitude=1.0, frequency=f,
                                  n_cycles=3, sample_rate_hz=400_000.0,
                                  dc_offset=0.4))
    sim2 = simulate_currents(m, t, np.stack([w2, np.zeros_like(w2)], axis=1))
    ok(not sim2.returns_to_origin()["closed"],
       "a waveform with a DC offset does not return to the origin")


# ── 6. no 4-D array is ever allocated ──────────────────────────────
def t_memory():
    print("6. memory stays separable")
    shape = (40, 40, 40)            # 64k voxels
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-40, -40, -40]
    pts = world_coordinates(aff, shape)
    maps = np.stack([(120.0 * pts[:, 0]).reshape(shape),
                     (90.0 * pts[:, 1]).reshape(shape),
                     (70.0 * pts[:, 2]).reshape(shape)], axis=-1)
    mask = np.ones(shape, dtype=bool)
    m = build_model(maps, mask, aff, unit_status=UnitStatus.USER_CONFIRMED)

    n_t = 4000
    t = np.linspace(0, 0.02, n_t)
    cur = np.stack([np.sin(2 * np.pi * 300 * t), np.zeros(n_t),
                    np.zeros(n_t)], axis=1)

    four_d_bytes = m.n_voxels * n_t * 8
    tracemalloc.start()
    sim = simulate_currents(m, t, cur)
    _ = sim.field_frame(sim.peak_index())
    _ = sim.phase_frame_rad(sim.peak_index())
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("       a full 4-D field would be %.1f GB; the simulation peaked at "
          "%.1f MB" % (four_d_bytes / 2 ** 30, peak / 2 ** 20))
    ok(peak < four_d_bytes / 50,
       "peak allocation is under 2%% of a 4-D field (%.1f MB vs %.1f GB)"
       % (peak / 2 ** 20, four_d_bytes / 2 ** 30))
    ok(sim.q.shape == (n_t, 3) and sim.k.shape == (n_t, 3),
       "only per-channel and per-axis time series are stored")


# ── 7. target-gradient mode ────────────────────────────────────────
def t_target_mode():
    print("7. target linear gradient mode")
    G0 = 300.0
    maps = synthetic([lambda x, y, z: G0 * x,
                      lambda x, y, z: 40.0 * y + 15.0 * x * z])
    m = build_model(maps, MASK, AFF, unit_status=UnitStatus.USER_CONFIRMED)
    beta = 500.0                     # Hz/m peak
    t, w = generate(WaveformSpec("sine", amplitude=beta, frequency=150.0,
                                 n_cycles=2, sample_rate_hz=300_000.0))
    sim, info = simulate_target_gradient(m, t, w, [1, 0, 0])

    peak = sim.peak_index()
    close(np.abs(sim.g[:, 0]).max(), beta, 1e-6,
          "the achieved peak gradient equals the request")
    close(np.abs(sim.i[peak, 0]), beta / G0, 1e-6,
          "peak current = requested gradient / (Hz/m/A)")
    ok(np.abs(sim.i[:, 1]).max() < 1e-9,
       "the nonlinear channel is not driven for a pure X request")

    mm = encoding_metrics(m, sim.i[peak], direction=[1, 0, 0],
                          requested_gradient_hz_per_m=beta)
    close(mm["gradient_gain"], 1.0, 1e-6, "gain 1.0")
    close(mm["nonlinear_rms_hz"], 0.0, 1e-6, "no nonlinear residual")
    # One conversion, named for what it converts. 42577 Hz/mT.
    from core.linear_encoding_metrics import GAMMA_HZ_PER_MT, to_mt_per_m
    close(GAMMA_HZ_PER_MT, 42577.0, 1e-9, "gamma = 42577 Hz/mT")
    close(mm["achieved_gradient_magnitude_mt_per_m"], beta / 42577.0, 1e-12,
          "achieved mT/m = achieved Hz/m / 42577")
    close(mm["requested_gradient_mt_per_m"], beta / 42577.0, 1e-12,
          "requested mT/m = requested Hz/m / 42577")
    close(float(to_mt_per_m(42577.0)), 1.0, 1e-12, "42577 Hz/m is exactly 1 mT/m")
    # requested and achieved are separate keys precisely so a table cannot put
    # one in the Hz/m column and the other in the mT/m column.
    ok("gradient_gain" in mm and abs(mm["gradient_gain"] - 1.0) < 1e-6,
       "gain is reported alongside, so the two can never be confused")


if __name__ == "__main__":
    for fn in (t_exact_x, t_non_orthogonal, t_rank, t_validation, t_physics,
               t_memory, t_target_mode):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
