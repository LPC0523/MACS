"""Time waveforms for the offline encoding simulator. Numbers only, no hardware.

IDEALIZED OFFLINE ENCODING SIMULATION. Nothing here is synchronised to a
scanner, no waveform produced here can be sent anywhere, and a shape that
integrates nicely on paper may be unreachable by a real amplifier.

Units, fixed everywhere in this module
--------------------------------------
    t          seconds
    amplitude  whatever the caller is generating -- Hz/m for a gradient
               waveform, A for a channel current waveform. The generator does
               not know which, so it never labels the axis for you.
    frequency  Hz
    phase      radians
    integral   amplitude * seconds

Integration is trapezoidal throughout, including the cumulative form. Not
because it is the most accurate rule, but because the cumulative integral has
to agree with the total: `cumulative_integral(...)[-1]` and `integral(...)`
must be the same number, or a k-space trajectory will not end where the zeroth
moment says it does. Simpson's rule on a non-uniform grid does not give that
for free; the trapezoid does.
"""

from __future__ import annotations

import numpy as np

__all__ = ["WAVEFORMS", "WaveformSpec", "generate", "time_axis",
           "load_csv_waveform", "waveform_stats", "integral",
           "cumulative_integral", "check_time_axis", "check_nyquist",
           "DISCONTINUOUS"]

#: Shapes whose derivative is unbounded. Fine as mathematics, and the caller
#: is told so rather than being stopped: a square wave is a legitimate thing to
#: put through a linearity check even though no amplifier will follow it.
DISCONTINUOUS = ("square",)

WAVEFORMS = ("sine", "cosine", "triangle", "trapezoid", "square",
             "piecewise_linear", "custom_csv")

#: Below this the ramp is treated as absent rather than as a very steep edge.
_MIN_RAMP_S = 1e-12


class WaveformSpec:
    """What to generate. Validated on construction, not at use.

    `ramp_s` applies a raised-cosine envelope over the first and last ramp, so
    the waveform starts and ends at zero. Without it a sine that begins at
    non-zero amplitude has an implicit step at t=0 whose derivative is
    infinite, which then shows up as a spurious spike in di/dt and in any
    voltage estimate built on it.
    """

    def __init__(self, kind="sine", amplitude=1.0, frequency=100.0,
                 phase_rad=0.0, duration_s=0.01, sample_rate_hz=1e5,
                 dc_offset=0.0, ramp_s=0.0, n_cycles=None, duty=0.5,
                 flat_fraction=0.5, points=None, csv_time=None,
                 csv_values=None):
        if kind not in WAVEFORMS:
            raise ValueError("unknown waveform %r; expected one of %s"
                             % (kind, ", ".join(WAVEFORMS)))
        self.kind = kind
        self.amplitude = float(amplitude)
        self.frequency = float(frequency)
        self.phase_rad = float(phase_rad)
        self.sample_rate_hz = float(sample_rate_hz)
        self.dc_offset = float(dc_offset)
        self.ramp_s = float(ramp_s)
        self.duty = float(duty)
        self.flat_fraction = float(flat_fraction)
        self.points = points            # piecewise_linear: [(t, value), ...]
        self.csv_time = csv_time
        self.csv_values = csv_values

        # Duration may be given directly or as a whole number of cycles. Cycles
        # win when both are present, because "four cycles" is a statement about
        # the waveform and "10 ms" is a statement about the clock.
        if n_cycles is not None:
            if self.frequency <= 0:
                raise ValueError("n_cycles needs a positive frequency")
            self.duration_s = float(n_cycles) / self.frequency
        else:
            self.duration_s = float(duration_s)
        self.n_cycles = n_cycles

        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.duration_s <= 0 and kind != "custom_csv":
            raise ValueError("duration_s must be positive")
        if self.ramp_s < 0:
            raise ValueError("ramp_s must be >= 0")
        if 2 * self.ramp_s > self.duration_s and self.ramp_s > _MIN_RAMP_S:
            raise ValueError(
                "ramp_s %.6g s twice over exceeds duration %.6g s"
                % (self.ramp_s, self.duration_s))
        if kind == "piecewise_linear" and not points:
            raise ValueError("piecewise_linear needs `points`")
        if kind == "custom_csv" and (csv_time is None or csv_values is None):
            raise ValueError("custom_csv needs csv_time and csv_values")

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "amplitude": self.amplitude,
             "frequency_hz": self.frequency, "phase_rad": self.phase_rad,
             "duration_s": self.duration_s,
             "sample_rate_hz": self.sample_rate_hz,
             "dc_offset": self.dc_offset, "ramp_s": self.ramp_s,
             "n_cycles": self.n_cycles, "duty": self.duty,
             "flat_fraction": self.flat_fraction,
             "discontinuous": self.kind in DISCONTINUOUS}
        if self.kind == "piecewise_linear":
            d["points"] = [[float(a), float(b)] for a, b in self.points]
        return d


def time_axis(duration_s: float, sample_rate_hz: float) -> np.ndarray:
    """Samples from 0 to `duration_s` inclusive, strictly increasing.

    The endpoint is included so a whole number of cycles really closes: with
    an exclusive end the trapezoidal integral of a full sine misses the last
    panel and the trajectory does not return to the origin.
    """
    if sample_rate_hz <= 0 or duration_s <= 0:
        raise ValueError("duration and sample rate must be positive")
    n = int(round(duration_s * sample_rate_hz)) + 1
    if n < 2:
        raise ValueError("fewer than two samples: raise the sample rate")
    return np.linspace(0.0, duration_s, n)


def check_time_axis(t) -> np.ndarray:
    """`t` as float64, or an error saying exactly what is wrong with it."""
    a = np.asarray(t, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError("time axis needs at least two samples")
    if not np.all(np.isfinite(a)):
        raise ValueError("time axis contains non-finite values")
    d = np.diff(a)
    if np.any(d <= 0):
        bad = int(np.argmax(d <= 0))
        raise ValueError(
            "time axis must be strictly increasing; t[%d]=%.9g is not greater "
            "than t[%d]=%.9g" % (bad + 1, a[bad + 1], bad, a[bad]))
    return a


def check_nyquist(sample_rate_hz: float, frequency_hz: float,
                  kind: str = "sine") -> dict:
    """Is this sample rate enough to represent this waveform?

    Reported, never enforced, and split into questions that have different
    answers. Ten samples per cycle is the practical line for a trapezoidal
    integral to track a sine to a fraction of a percent. A square wave is not
    band-limited, so its edges are never resolved at any finite rate -- but
    that is a statement about the edges, not about the integral: the integral
    of an ideal square wave converges perfectly well as dt shrinks. What cannot
    be settled here is what a real amplifier would do with those edges.
    """
    out = {"sample_rate_hz": float(sample_rate_hz),
           "frequency_hz": float(frequency_hz),
           "samples_per_cycle": (float(sample_rate_hz / frequency_hz)
                                 if frequency_hz > 0 else float("inf")),
           "nyquist_ok": True,
           "waveform_bandlimited": kind not in DISCONTINUOUS,
           "edge_resolution": kind not in DISCONTINUOUS,
           "integration_accuracy": "unknown",
           "hardware_realisability": "unknown",
           "integration_ok": True, "note": ""}
    if frequency_hz <= 0:
        return out
    spc = out["samples_per_cycle"]
    if spc <= 2.0:
        out["nyquist_ok"] = False
        out["integration_ok"] = False
        out["integration_accuracy"] = "unusable"
        out["note"] = ("%.2f samples per cycle is at or below Nyquist -- this "
                       "waveform is not represented at all" % spc)
    elif spc < 10.0:
        out["integration_ok"] = False
        out["integration_accuracy"] = "coarse"
        out["note"] = ("%.2f samples per cycle: representable, but the "
                       "trapezoidal integral behind every phase and k number "
                       "is coarse at this rate" % spc)
    else:
        out["integration_accuracy"] = "adequate"
    if kind in DISCONTINUOUS:
        out["note"] = ((out["note"] + "; ") if out["note"] else "") + (
            "a %s wave is not band-limited: its edges are unresolved at any "
            "rate, and what a real amplifier would do with them is not "
            "something this simulation can determine. The ideal waveform's "
            "integral still converges with dt." % kind)
    return out


def _raised_cosine_envelope(t, duration_s, ramp_s):
    """1 in the middle, smoothly 0 at both ends. Zero derivative at the joins."""
    if ramp_s <= _MIN_RAMP_S:
        return np.ones_like(t)
    e = np.ones_like(t)
    up = t < ramp_s
    e[up] = 0.5 * (1.0 - np.cos(np.pi * t[up] / ramp_s))
    dn = t > (duration_s - ramp_s)
    e[dn] = 0.5 * (1.0 - np.cos(np.pi * (duration_s - t[dn]) / ramp_s))
    return np.clip(e, 0.0, 1.0)


def _shape(kind, t, spec):
    """Unit-amplitude shape in [-1, 1], before amplitude, offset and envelope."""
    f, ph = spec.frequency, spec.phase_rad
    if kind == "sine":
        return np.sin(2 * np.pi * f * t + ph)
    if kind == "cosine":
        return np.cos(2 * np.pi * f * t + ph)
    if kind == "square":
        # +1 on the closed upper half, so a sample landing exactly on a zero
        # crossing is not left at 0 and does not bias the integral.
        return np.where(np.sin(2 * np.pi * f * t + ph) >= 0, 1.0, -1.0)
    if kind == "triangle":
        # Phase measured the same way as the sine: both start at 0 and rise.
        u = (f * t + ph / (2 * np.pi)) % 1.0
        return _tri(u)
    if kind == "trapezoid":
        u = (f * t + ph / (2 * np.pi)) % 1.0
        return _trap(u, spec.flat_fraction)
    if kind == "piecewise_linear":
        pts = sorted((float(a), float(b)) for a, b in spec.points)
        tp = np.array([p[0] for p in pts])
        vp = np.array([p[1] for p in pts])
        if np.any(np.diff(tp) <= 0):
            raise ValueError("piecewise_linear points must have increasing time")
        return np.interp(t, tp, vp, left=vp[0], right=vp[-1])
    raise ValueError("no shape for %r" % kind)


def _tri(u):
    """Triangle rising through zero at u=0, peak at 0.25, zero at 0.5."""
    return np.where(u < 0.25, 4 * u,
                    np.where(u < 0.75, 2 - 4 * u, 4 * u - 4))


def _trap(u, flat_fraction):
    """Trapezoid: ramp, flat top, ramp, flat bottom. `flat_fraction` of a half
    period is spent flat."""
    ff = float(np.clip(flat_fraction, 0.0, 1.0))
    ramp = (1.0 - ff) / 2.0            # fraction of a half-period spent ramping
    out = np.empty_like(u)
    h = u * 2.0                        # position within the half period
    first = h < 1.0
    hh = np.where(first, h, h - 1.0)
    sgn = np.where(first, 1.0, -1.0)
    if ramp <= 0:
        val = np.ones_like(hh)
    else:
        val = np.where(hh < ramp, hh / ramp,
                       np.where(hh < 1.0 - ramp, 1.0, (1.0 - hh) / ramp))
    out[:] = sgn * val
    return out


def generate(spec: WaveformSpec):
    """(t, y) for one waveform. `t` in seconds, `y` in the caller's unit."""
    if spec.kind == "custom_csv":
        t = check_time_axis(spec.csv_time)
        y = np.asarray(spec.csv_values, dtype=np.float64).ravel()
        if y.shape[0] != t.shape[0]:
            raise ValueError("csv_values has %d entries for %d time samples"
                             % (y.shape[0], t.shape[0]))
        if not np.all(np.isfinite(y)):
            raise ValueError("csv_values contains non-finite values")
        return t, spec.amplitude * y + spec.dc_offset

    t = time_axis(spec.duration_s, spec.sample_rate_hz)
    y = _shape(spec.kind, t, spec)
    y = spec.amplitude * y * _raised_cosine_envelope(t, spec.duration_s,
                                                     spec.ramp_s)
    return t, y + spec.dc_offset


def integral(t, y) -> float:
    """Trapezoidal integral over the whole axis, in amplitude*seconds."""
    t = check_time_axis(t)
    return float(np.trapezoid(np.asarray(y, dtype=np.float64), t))


def cumulative_integral(t, y) -> np.ndarray:
    """Running trapezoidal integral, starting at 0, same length as `t`.

    Works on a 1-D waveform or on an (T, N) block, integrating down the time
    axis -- the channel form is what the phase and k-space code needs, and
    doing it here keeps one integration rule in the project.
    """
    t = check_time_axis(t)
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        if a.shape[0] != t.shape[0]:
            raise ValueError("y has %d samples for %d times"
                             % (a.shape[0], t.shape[0]))
        out = np.zeros_like(a)
        out[1:] = np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t))
        return out
    if a.ndim == 2:
        if a.shape[0] != t.shape[0]:
            raise ValueError("y has %d rows for %d times"
                             % (a.shape[0], t.shape[0]))
        out = np.zeros_like(a)
        out[1:] = np.cumsum(0.5 * (a[1:] + a[:-1]) * np.diff(t)[:, None],
                            axis=0)
        return out
    raise ValueError("y must be 1-D or 2-D, got %d dimensions" % a.ndim)


def derivative(t, y) -> np.ndarray:
    """d/dt by central differences. Same shape as `y`."""
    t = check_time_axis(t)
    a = np.asarray(y, dtype=np.float64)
    if a.ndim == 1:
        return np.gradient(a, t)
    return np.stack([np.gradient(a[:, j], t) for j in range(a.shape[1])],
                    axis=1)


def waveform_stats(t, y, kind: str = "") -> dict:
    """Peak, RMS, integral and slew of one waveform, with its units unstated.

    The unit is the caller's: Hz/m for a gradient, A for a current. Naming it
    here would be a guess, and a wrong unit label is worse than none.
    """
    t = check_time_axis(t)
    a = np.asarray(y, dtype=np.float64).ravel()
    if a.shape[0] != t.shape[0]:
        raise ValueError("waveform has %d samples for %d times"
                         % (a.shape[0], t.shape[0]))
    if not np.all(np.isfinite(a)):
        raise ValueError("waveform contains non-finite values")
    d = derivative(t, a)
    return {
        "n_samples": int(a.size),
        "duration_s": float(t[-1] - t[0]),
        "dt_s": float(np.median(np.diff(t))),
        "max": float(a.max()), "min": float(a.min()),
        "peak_abs": float(np.abs(a).max()),
        "mean": float(a.mean()),
        "rms": float(np.sqrt(np.mean(a ** 2))),
        "integral": float(np.trapezoid(a, t)),
        "final_cumulative_integral": float(cumulative_integral(t, a)[-1]),
        "max_abs_derivative": float(np.abs(d).max()),
        "discontinuous": bool(kind in DISCONTINUOUS),
    }


def load_csv_waveform(path, expect_channels=None):
    """Read `time_s, ch1, ch2, ...` from a CSV. Returns (t, values, names).

    `values` is (T, N) in the file's own unit -- amps for a current table.
    `expect_channels` is checked when given: a table with the wrong number of
    columns is a different coil, and silently padding or truncating it would
    put currents on the wrong channels.
    """
    import csv as _csv
    from pathlib import Path

    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(_csv.reader(fh))
    rows = [r for r in rows if r and any(str(c).strip() for c in r)]
    if len(rows) < 3:
        raise ValueError("%s has fewer than two data rows" % p.name)

    head = [str(c).strip() for c in rows[0]]
    try:
        float(head[0])
        names = ["ch%d" % (k + 1) for k in range(len(head) - 1)]
        body = rows
    except ValueError:
        if head[0].lower() not in ("time_s", "t", "time"):
            raise ValueError(
                "%s: first column must be time_s, found %r" % (p.name, head[0]))
        names = head[1:]
        body = rows[1:]

    arr = np.array([[float(c) for c in r] for r in body], dtype=np.float64)
    t = check_time_axis(arr[:, 0])
    vals = arr[:, 1:]
    if not np.all(np.isfinite(vals)):
        raise ValueError("%s contains non-finite values" % p.name)
    if expect_channels is not None and vals.shape[1] != int(expect_channels):
        raise ValueError(
            "%s has %d channel columns but the coil has %d -- refusing to "
            "guess which channel is which"
            % (p.name, vals.shape[1], int(expect_channels)))
    return t, vals, names
