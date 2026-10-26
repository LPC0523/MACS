"""Current bounds, the two strategies, and an electrical model with no data.

    D:\\python.exe tests/test_linear_encoding_constraints.py

The clipping tests are the important ones. Clipping would keep every number
inside +-3 A while turning the waveform into a different one, and since the
k-space trajectory is that waveform's integral, the plots would show a
trajectory the currents no longer produce. These assert the shape survives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.linear_encoding_metrics import (  # noqa: E402
    ConstraintViolation, apply_strategy, check_current_bounds,
    electrical_estimate)
from core.linear_encoding_waveform import WaveformSpec, generate  # noqa: E402

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def close(a, b, tol, msg):
    d = abs(float(a) - float(b))
    ok(d <= tol, "%s  (|%.6g - %.6g| = %.3g <= %.3g)" % (msg, a, b, d, tol))


def two_channel(peak=(2.0, 1.0), freq=200.0, cycles=3, rate=100_000.0):
    t, w = generate(WaveformSpec("sine", amplitude=1.0, frequency=freq,
                                 n_cycles=cycles, sample_rate_hz=rate))
    dyn = np.stack([peak[0] * w, peak[1] * np.cos(2 * np.pi * freq * t)],
                   axis=1)
    return t, dyn


# ── 1. bounds with and without a baseline ──────────────────────────
def t_bounds():
    print("1. +-3 A checked per channel and per sample")
    t, dyn = two_channel(peak=(2.0, 1.0))
    r = check_current_bounds(t, dyn, limit_a=3.0)
    ok(r["ok"] and r["violation_count"] == 0, "2.0 A and 1.0 A peaks pass")
    ok(r["clipping_applied"] is False, "the report states no clipping")

    # Peaks become 4.0 and 2.0 A: only ch1 is out. Naming ch2 as well would be
    # over-reporting, which sends someone to look at a channel that is fine.
    r = check_current_bounds(t, dyn * 2.0, limit_a=3.0)
    ok(not r["ok"], "a 4.0 A peak fails")
    ok(r["violating_channels"] == ["ch1"],
       "only the channel that actually exceeds is named (%s)"
       % r["violating_channels"])
    ok(r["per_channel"][1]["saturated"] is False, "ch2 at 2.0 A is not flagged")
    ok(r["first_violation_time_s"] is not None, "first violation time reported")

    # And when both do exceed, both are named.
    r = check_current_bounds(t, dyn * 4.0, limit_a=3.0)
    ok(r["violating_channels"] == ["ch1", "ch2"],
       "both channels named when both exceed")

    # A baseline makes the usable window asymmetric.
    base = np.array([2.5, -2.0])
    r = check_current_bounds(t, dyn + base, limit_a=3.0, i_static=base)
    c1, c2 = r["per_channel"]
    close(c1["dynamic_window_upper_a"], 0.5, 1e-12, "ch1 has 0.5 A of room up")
    close(c1["dynamic_window_lower_a"], -5.5, 1e-12, "ch1 has 5.5 A down")
    close(c2["dynamic_window_upper_a"], 5.0, 1e-12, "ch2 has 5.0 A up")
    close(c2["dynamic_window_lower_a"], -1.0, 1e-12, "ch2 has 1.0 A down")
    ok(not r["ok"], "the 2.0 A swing does not fit in ch1's 0.5 A of headroom")


# ── 2. reject produces nothing ─────────────────────────────────────
def t_reject():
    print("2. reject refuses instead of returning a modified waveform")
    t, dyn = two_channel(peak=(5.0, 1.0))
    try:
        apply_strategy(t, dyn, limit_a=3.0, strategy="reject")
        ok(False, "an out-of-bounds request should be rejected")
    except ConstraintViolation as e:
        ok("reject" in str(e) and "未产生任何输出" in str(e),
           "rejected with no output: %s" % str(e)[:58])

    out = apply_strategy(t, two_channel(peak=(2.0, 1.0))[1], limit_a=3.0,
                         strategy="reject")
    ok(out["scale"] == 1.0 and not out["scaled"],
       "a feasible request passes through untouched")
    ok(np.allclose(out["dynamic_currents"], two_channel(peak=(2.0, 1.0))[1]),
       "and is returned bit for bit")


# ── 3. uniform scale preserves shape, ratio and phase ──────────────
def t_uniform_scale():
    print("3. uniform_scale keeps the waveform's shape and channel ratios")
    t, dyn = two_channel(peak=(6.0, 3.0))
    out = apply_strategy(t, dyn, limit_a=3.0, strategy="uniform_scale")
    s = out["scale"]
    close(s, 0.5, 1e-9, "scale = 3.0 / 6.0")
    close(np.abs(out["dynamic_currents"]).max(), 3.0, 1e-9,
          "the peak lands exactly on the bound")
    ok(out["bounds"]["ok"], "the scaled waveform satisfies the bound")
    ok(out["clipping_applied"] is False, "no clipping was used")

    # Shape: the scaled waveform is exactly s times the original everywhere,
    # which is what clipping would destroy.
    ok(np.allclose(out["dynamic_currents"], s * dyn, atol=1e-12),
       "every sample is exactly s times the request")
    r_before = dyn[:, 0] / np.where(np.abs(dyn[:, 1]) > 1e-9, dyn[:, 1], np.nan)
    r_after = (out["dynamic_currents"][:, 0] /
               np.where(np.abs(out["dynamic_currents"][:, 1]) > 1e-9,
                        out["dynamic_currents"][:, 1], np.nan))
    ok(np.allclose(r_before, r_after, equal_nan=True, atol=1e-9),
       "the channel-to-channel ratio is unchanged at every sample")
    # Phase: the zero crossings do not move.
    z0 = np.where(np.diff(np.sign(dyn[:, 0])))[0]
    z1 = np.where(np.diff(np.sign(out["dynamic_currents"][:, 0])))[0]
    ok(np.array_equal(z0, z1), "zero crossings are unmoved (phase preserved)")
    close(out["requested_peak_dynamic_a"], 6.0, 1e-9, "requested peak reported")
    close(out["feasible_peak_dynamic_a"], 3.0, 1e-9, "feasible peak reported")


# ── 4. scaling against an asymmetric window ────────────────────────
def t_asymmetric():
    print("4. uniform_scale respects an asymmetric window from the baseline")
    t, dyn = two_channel(peak=(2.0, 1.0))
    base = np.array([2.5, -2.0])
    out = apply_strategy(t, dyn, i_static=base, limit_a=3.0,
                         strategy="uniform_scale")
    total = out["currents"]
    ok(np.all(np.abs(total) <= 3.0 + 1e-9),
       "every sample of every channel is inside +-3 A")
    # ch1 may swing +0.5 on a 2.0 request -> 0.25; ch2 may swing -1.0 on 1.0
    # -> 1.0. The binding channel is ch1.
    close(out["scale"], 0.25, 1e-9, "scale set by the tightest channel (ch1)")
    close(total[:, 0].max(), 3.0, 1e-6, "ch1 reaches exactly +3 A")
    ok(np.allclose(out["currents"] - out["dynamic_currents"], base, atol=1e-12),
       "the static baseline is untouched by the scaling")

    # A baseline that is itself out of bounds cannot be rescued.
    try:
        apply_strategy(t, dyn, i_static=np.array([3.5, 0.0]), limit_a=3.0,
                       strategy="uniform_scale")
        ok(False, "an out-of-bounds baseline should be refused")
    except ConstraintViolation as e:
        ok("baseline" in str(e), "out-of-bounds baseline refused: %s"
           % str(e)[:52])


# ── 5. no silent clipping anywhere ─────────────────────────────────
def t_no_clipping():
    print("5. nothing clips")
    t, dyn = two_channel(peak=(9.0, 4.5))
    out = apply_strategy(t, dyn, limit_a=3.0, strategy="uniform_scale")
    # If any sample had been clipped the waveform would flatten at the bound:
    # the count of samples sitting exactly on +-3 A would be large. A scaled
    # sine touches its peak twice per cycle and no more.
    at_bound = int(np.sum(np.abs(np.abs(out["dynamic_currents"]) - 3.0) < 1e-9))
    ok(at_bound <= 8, "only the true peaks touch the bound (%d samples)"
       % at_bound)
    # And the scaled signal is still a pure sine: its RMS/peak ratio is 1/sqrt2.
    col = out["dynamic_currents"][:, 0]
    close(np.sqrt(np.mean(col ** 2)) / np.abs(col).max(), 1 / np.sqrt(2), 2e-3,
          "the scaled channel is still sinusoidal (RMS/peak = 1/sqrt2)")
    ok(all(not r.get("clipping_applied", False)
           for r in (out, out["bounds"])), "every report says clipping: False")


# ── 6. the electrical model refuses to invent numbers ──────────────
def t_electrical():
    print("6. electrical model is unknown until R and L are supplied")
    t, dyn = two_channel(peak=(2.0, 1.0))
    e = electrical_estimate(t, dyn)
    ok(e["available"] is False, "not available without R or L")
    ok(e["peak_voltage_v"] is None and e["resistive_power_w"] is None,
       "voltage and power are None, not 0")
    ok(e["gradient_coupling_model_status"] == "not_characterized",
       "gradient coupling is 'not_characterized', not assumed absent")
    ok(e["amplifier_bandwidth_status"] == "unknown", "bandwidth unknown")

    # Diagonal R and L against the analytic voltage of a sine.
    f, A, R, L = 200.0, 2.0, 0.5, 1e-4
    t2, w = generate(WaveformSpec("sine", amplitude=A, frequency=f, n_cycles=4,
                                  sample_rate_hz=1_000_000.0))
    cur = np.stack([w, np.zeros_like(w)], axis=1)
    e = electrical_estimate(t2, cur, resistance_ohm=[R, R],
                            inductance_h=[L, L])
    ok(e["available"], "available once R and L are given")
    # v = R A sin + L A w cos  ->  peak = A sqrt(R^2 + (wL)^2)
    om = 2 * np.pi * f
    close(e["per_channel"][0]["peak_voltage_v"],
          A * np.sqrt(R ** 2 + (om * L) ** 2), 1e-3,
          "peak voltage matches A*sqrt(R^2+(wL)^2)")
    close(e["per_channel"][0]["resistive_power_w"], R * A ** 2 / 2, 1e-3,
          "resistive power matches R*A^2/2")

    # A mutual inductance matrix must actually be used: driving channel 1 alone
    # has to induce a voltage on channel 2.
    M = np.array([[L, 0.4 * L], [0.4 * L, L]])
    e2 = electrical_estimate(t2, cur, resistance_ohm=[R, R],
                             mutual_inductance_h=M)
    close(e2["per_channel"][1]["peak_voltage_v"],
          0.4 * L * A * om, 1e-3,
          "channel 2 sees M*di1/dt even though it carries no current")
    ok(e2["per_channel"][1]["peak_voltage_v"] > 0,
       "mutual inductance is not silently dropped")

    ev = electrical_estimate(t2, cur, resistance_ohm=[R, R],
                             inductance_h=[L, L], voltage_limit_v=[0.5, 100.0])
    ok(ev["voltage_violations"]["count"] > 0
       and ev["voltage_violations"]["channels"] == ["ch1"],
       "a voltage limit is checked and the offending channel named")


if __name__ == "__main__":
    for fn in (t_bounds, t_reject, t_uniform_scale, t_asymmetric,
               t_no_clipping, t_electrical):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
