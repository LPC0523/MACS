"""Waveforms, integration, and the coordinate grid they will be used with.

    D:\\python.exe tests/test_linear_encoding_waveform.py

The integration tests matter more than they look. Every k-space number in this
feature is an integral of a waveform, so a rule that is off by one panel puts
the trajectory somewhere the currents never went.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.linear_encoding_waveform import (  # noqa: E402
    WaveformSpec, check_nyquist, check_time_axis, cumulative_integral,
    derivative, generate, integral, load_csv_waveform, waveform_stats)
from core.spatial_grid import (  # noqa: E402
    direction_field, unit_vector, weighted_centre, world_coordinates)

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def close(a, b, tol, msg):
    d = abs(float(a) - float(b))
    ok(d <= tol, "%s  (|%.6g - %.6g| = %.3g <= %.3g)" % (msg, a, b, d, tol))


# ── 1. sine: amplitude, frequency, phase ────────────────────────────
def t_sine():
    print("1. sine sampling, amplitude, frequency, phase")
    A, f = 2.5, 250.0
    t, y = generate(WaveformSpec("sine", amplitude=A, frequency=f,
                                 n_cycles=4, sample_rate_hz=200_000.0))
    close(y.max(), A, 1e-3, "peak equals amplitude")
    close(np.sqrt(np.mean(y ** 2)), A / np.sqrt(2), 1e-3, "RMS = A/sqrt(2)")
    close(t[-1], 4 / f, 1e-12, "four cycles at 250 Hz lasts 16 ms")
    ok(abs(y[0]) < 1e-9, "starts at zero with zero phase")

    # A quarter-period phase shift turns the sine into a cosine.
    _, yc = generate(WaveformSpec("sine", amplitude=A, frequency=f, n_cycles=4,
                                  sample_rate_hz=200_000.0,
                                  phase_rad=np.pi / 2))
    _, yr = generate(WaveformSpec("cosine", amplitude=A, frequency=f,
                                  n_cycles=4, sample_rate_hz=200_000.0))
    ok(np.allclose(yc, yr, atol=1e-9), "sine with +pi/2 phase == cosine")

    # Analytic check of the running integral: int A sin(wt) = A(1-cos wt)/w
    w = 2 * np.pi * f
    q = cumulative_integral(t, y)
    ana = A * (1 - np.cos(w * t)) / w
    close(np.abs(q - ana).max(), 0.0, 1e-6, "running integral matches analytic")


# ── 2. triangle and trapezoid integrate correctly ───────────────────
def t_shapes():
    print("2. triangle and trapezoid integration")
    for kind in ("triangle", "trapezoid", "square", "cosine"):
        t, y = generate(WaveformSpec(kind, amplitude=1.0, frequency=100.0,
                                     n_cycles=6, sample_rate_hz=400_000.0))
        dt = float(np.median(np.diff(t)))
        # A square wave costs one sample of area here and cannot do better.
        # The axis includes both endpoints, both land on a zero crossing, and
        # the trapezoid rule gives each half weight -- so the pair contributes
        # one full sample that the six periods do not cancel. That is the
        # discontinuity `check_nyquist` already refuses to vouch for, not a
        # defect in the rule: no sample rate removes it, it only shrinks with
        # dt. Continuous shapes are held to machine precision.
        tol = 1.1 * dt if kind == "square" else 1e-6
        close(integral(t, y), 0.0, tol,
              "%s over whole cycles integrates to zero (tol %.1e)" % (kind, tol))
        close(cumulative_integral(t, y)[-1], integral(t, y), 1e-12,
              "%s: cumulative endpoint == total integral" % kind)

    # A triangle of unit amplitude has RMS 1/sqrt(3); a square has 1.
    _, tri = generate(WaveformSpec("triangle", frequency=100.0, n_cycles=8,
                                   sample_rate_hz=400_000.0))
    close(np.sqrt(np.mean(tri ** 2)), 1 / np.sqrt(3), 2e-3, "triangle RMS")
    _, sq = generate(WaveformSpec("square", frequency=100.0, n_cycles=8,
                                  sample_rate_hz=400_000.0))
    close(np.sqrt(np.mean(sq ** 2)), 1.0, 1e-9, "square RMS")
    ok(waveform_stats(*generate(WaveformSpec("square", frequency=100.0,
                                             n_cycles=2,
                                             sample_rate_hz=100_000.0)),
                      kind="square")["discontinuous"],
       "square is flagged discontinuous")


# ── 3. time axis and Nyquist ────────────────────────────────────────
def t_axis():
    print("3. time axis and Nyquist reporting")
    try:
        check_time_axis([0.0, 1e-3, 1e-3, 2e-3])
        ok(False, "repeated time should be rejected")
    except ValueError as e:
        ok("strictly increasing" in str(e), "repeated time rejected: %s"
           % str(e)[:60])
    try:
        check_time_axis([0.0, 2e-3, 1e-3])
        ok(False, "decreasing time should be rejected")
    except ValueError:
        ok(True, "decreasing time rejected")

    n = check_nyquist(700.0, 400.0)
    ok(not n["nyquist_ok"], "1.75 samples/cycle is below Nyquist: %s"
       % n["note"][:46])
    n = check_nyquist(1000.0, 400.0)
    ok(n["nyquist_ok"] and not n["integration_ok"],
       "2.5 samples/cycle is above Nyquist but its integral is not trusted")
    n = check_nyquist(100_000.0, 400.0)
    ok(n["nyquist_ok"] and n["integration_ok"], "250 samples/cycle is fine")
    # A square wave is not band-limited and its edges are never resolved --
    # but that is a claim about the edges, not about the integral, which
    # converges with dt like any other. Saying "the integral is untrustworthy"
    # was the wrong conclusion drawn from the right observation.
    n = check_nyquist(100_000.0, 400.0, kind="square")
    ok(n["waveform_bandlimited"] is False, "square is not band-limited")
    ok(n["edge_resolution"] is False, "and its edges are unresolved")
    ok(n["integration_accuracy"] == "adequate",
       "yet at 250 samples/cycle the integral is still adequate")
    ok(n["hardware_realisability"] == "unknown",
       "whether an amplifier could follow it is unknown, not asserted")
    ok("not band-limited" in n["note"], "and the note says why")


# ── 4. ramp removes the step at t=0 ─────────────────────────────────
def t_ramp():
    print("4. raised-cosine ramp")
    spec = WaveformSpec("cosine", amplitude=1.0, frequency=500.0, n_cycles=10,
                        sample_rate_hz=500_000.0, ramp_s=2e-3)
    t, y = generate(spec)
    ok(abs(y[0]) < 1e-9 and abs(y[-1]) < 1e-9,
       "ramped cosine starts and ends at zero")
    _, y0 = generate(WaveformSpec("cosine", amplitude=1.0, frequency=500.0,
                                  n_cycles=10, sample_rate_hz=500_000.0))
    ok(abs(y0[0]) > 0.9, "without a ramp the cosine starts at full amplitude")
    ok(np.abs(derivative(t, y)).max() < np.abs(
        derivative(*generate(WaveformSpec("square", frequency=500.0,
                                          n_cycles=10,
                                          sample_rate_hz=500_000.0)))).max(),
       "ramped waveform has a far smaller slew than a square")


# ── 5. CSV import ───────────────────────────────────────────────────
def t_csv():
    print("5. CSV current tables")
    import tempfile

    d = Path(tempfile.mkdtemp())
    good = d / "good.csv"
    good.write_text("time_s,ch1,ch2\n0,0.0,0.1\n0.001,0.5,-0.2\n"
                    "0.002,1.0,-0.4\n", encoding="utf-8")
    t, v, names = load_csv_waveform(good, expect_channels=2)
    ok(v.shape == (3, 2) and names == ["ch1", "ch2"], "shape and names read")
    close(v[2, 0], 1.0, 1e-12, "value read")

    try:
        load_csv_waveform(good, expect_channels=12)
        ok(False, "channel-count mismatch should be rejected")
    except ValueError as e:
        ok("refusing to guess" in str(e), "channel-count mismatch rejected")

    bad = d / "bad.csv"
    bad.write_text("time_s,ch1\n0,0.0\n0.002,1.0\n0.001,0.5\n", encoding="utf-8")
    try:
        load_csv_waveform(bad)
        ok(False, "non-increasing CSV time should be rejected")
    except ValueError as e:
        ok("strictly increasing" in str(e), "non-increasing CSV time rejected")


# ── 6. spatial grid comes from the affine, not the array axes ───────
def t_grid():
    print("6. world coordinates from the affine")
    from core.hybrid_shim import build_sh_basis

    shape = (7, 8, 9)
    # A rotated, translated, anisotropic affine -- axis 0 is nowhere near +x.
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    aff = np.eye(4)
    aff[:3, :3] = R @ np.diag([2.0, 3.0, 4.0])
    aff[:3, 3] = [10.0, -20.0, 30.0]

    pts = world_coordinates(aff, shape)
    ok(pts.shape == (np.prod(shape), 3), "one row per voxel")
    close(pts[0] @ [1, 0, 0] * 1000, 10.0, 1e-9, "origin voxel x = affine[0,3]")

    # Stepping one voxel along axis 0 moves along the rotated column, and its
    # x component is cos(30)*2 mm -- not 2 mm, which is what treating axis 0 as
    # x would give.
    step = (pts[shape[1] * shape[2]] - pts[0]) * 1000
    close(step[0], 2.0 * np.cos(th), 1e-9, "axis 0 step x = 2*cos(30) mm")
    close(step[1], 2.0 * np.sin(th), 1e-9, "axis 0 step y = 2*sin(30) mm")
    ok(abs(step[0] - 2.0) > 0.2, "axis 0 is NOT world x under rotation")

    # Agreement with the production implementation, so this is one algorithm
    # with a verified mirror rather than a fifth copy.
    basis = build_sh_basis(aff, shape)
    from core.hybrid_shim import CALIBRATION, GAMMA
    ok(np.allclose(basis["A11"].ravel(),
                   CALIBRATION["A11"] * GAMMA * pts[:, 0], atol=1e-12),
       "matches hybrid_shim.build_sh_basis x coordinates exactly")
    ok(np.allclose(basis["A10"].ravel(),
                   CALIBRATION["A10"] * GAMMA * pts[:, 2], atol=1e-12),
       "matches build_sh_basis z coordinates exactly")

    # Translation must not move the direction field, because it is measured
    # about the ROI centre.
    w = np.random.default_rng(0).random(pts.shape[0])
    f1 = direction_field(pts, [0, 1, 0], weights=w)
    aff2 = aff.copy()
    aff2[:3, 3] += [123.0, -45.0, 6.0]
    f2 = direction_field(world_coordinates(aff2, shape), [0, 1, 0], weights=w)
    ok(np.allclose(f1, f2, atol=1e-12),
       "direction field is translation invariant about the weighted centre")
    close(float(w @ f1), 0.0, 1e-9, "direction field has zero weighted mean")

    try:
        unit_vector([0, 0, 0])
        ok(False, "zero direction should be rejected")
    except ValueError:
        ok(True, "zero direction rejected")
    try:
        weighted_centre(np.zeros((0, 3)))
        ok(False, "empty ROI should be rejected")
    except ValueError as e:
        ok("empty" in str(e), "empty ROI rejected")


if __name__ == "__main__":
    for fn in (t_sine, t_shapes, t_axis, t_ramp, t_csv, t_grid):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
