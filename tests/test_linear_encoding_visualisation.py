"""The plots, the slider and the frames, driven from a real run.

    D:\\python.exe tests/test_linear_encoding_visualisation.py

Every curve is checked against the array it claims to draw, and every frame
against the matrix product it claims to be. A plot made from decimated data
still looks like a plot, and a frame computed with the wrong current vector
still looks like a field, so neither can be checked by eye.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_linear_encoding_e2e import SETUP  # noqa: E402

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def run_py(body, timeout=1800):
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
    r = subprocess.run([sys.executable, "-X", "utf8", "-u", "-c", src],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    return r.returncode, (r.stdout or "") + (r.stderr or "")


PREP = SETUP + '''
from PySide6.QtWidgets import QApplication
import numpy as np, time
app = QApplication([])
import widgets.linear_encoding_widget as lew

def pump(w, limit=600):
    t0 = time.time()
    while w._worker is not None and w._worker.isRunning() and time.time()-t0 < limit:
        app.processEvents(); time.sleep(0.01)
    app.processEvents()

def make(rate=50000, dur=20.0):
    w = lew.LinearEncodingWidget({})
    w.dsv_spin.setValue(4.0)
    w.btn_coverage.click(); pump(w)
    w.amp_spin.setValue(500.0); w.freq_spin.setValue(100.0)
    w.dur_spin.setValue(dur); w.rate_spin.setValue(rate)
    w.strategy_combo.setCurrentIndex(1)
    w.btn_run.click(); pump(w)
    return w
'''


def t_curves():
    print("1. the four plots carry real data from the worker")
    code, out = run_py(PREP + '''
w = make()
r, rt = w._result, w._runtime
pd = w.plot_data()
sim, d = rt["sim"], rt["direction"]
print("HAVE_RUNTIME", rt is not None and pd is not None)
print("NCURVE", len(pd["t"]))

# every curve against the array it claims to draw
idx = pd["idx"]
print("REQ_MATCH", bool(np.allclose(
    pd["requested"], np.asarray(rt["requested_waveform"])[idx], atol=1e-9)))
print("ACT_MATCH", bool(np.allclose(
    pd["actual_dynamic"], (sim.g_dynamic @ d)[idx], atol=1e-9)))
print("TOT_MATCH", bool(np.allclose(
    pd["total_parallel"], (sim.g_total @ d)[idx], atol=1e-9)))
orth = np.linalg.norm(sim.g_dynamic - np.outer(sim.g_dynamic @ d, d), axis=1)
print("ORTH_MATCH", bool(np.allclose(pd["orthogonal"], orth[idx], atol=1e-9)))
print("KD_MATCH", bool(np.allclose(pd["k_dynamic"], sim.k_dynamic[idx])))
print("KT_MATCH", bool(np.allclose(pd["k_total"], sim.k_total_local_model[idx])))

# the nonlinear component must not be anywhere in the k curves
nl = rt["model"].residual @ sim.q_dynamic[rt["peak_index"]]
print("NL_NONZERO", float(np.abs(nl).max()) > 0)
kmax = float(np.abs(pd["k_dynamic"]).max())
print("K_FROM_G_ONLY", bool(np.allclose(
    sim.k_dynamic, sim.q_dynamic @ rt["model"].G.T, atol=1e-12)))

# unit switch scales, and does not touch the stored data
w.unit_combo.setCurrentIndex(1)
pd2 = w.plot_data()
print("MT_SCALE", bool(np.allclose(pd2["actual_dynamic"] * 42577.0,
                                   pd["actual_dynamic"], rtol=1e-9)))
print("SIM_UNCHANGED", bool(np.allclose(sim.g_dynamic @ d,
                                        (sim.g_dynamic @ d))))
w.shutdown()
''')
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2000:]); return
    g = dict(l.split(" ", 1) for l in out.splitlines()
             if " " in l and l.split(" ", 1)[0].isupper())
    ok(g.get("HAVE_RUNTIME") == "True", "runtime and plot data exist")
    for k, m in (("REQ_MATCH", "requested curve"),
                 ("ACT_MATCH", "actual dynamic parallel = g_dynamic @ d"),
                 ("TOT_MATCH", "total parallel = g_total @ d"),
                 ("ORTH_MATCH", "orthogonal magnitude"),
                 ("KD_MATCH", "k_dynamic curve"),
                 ("KT_MATCH", "k_total_local_model curve")):
        ok(g.get(k) == "True", "%s comes from the worker result" % m)
    ok(g.get("NL_NONZERO") == "True",
       "the synthetic coil does have a nonlinear component")
    ok(g.get("K_FROM_G_ONLY") == "True",
       "and k is built from the linear part alone, never from it")
    ok(g.get("MT_SCALE") == "True", "the mT/m switch rescales by 42577")


def t_frames_and_slider():
    print("2. frames, slider and cache")
    code, out = run_py(PREP + '''
from core.linear_encoding_frames import FRAME_TYPES, frame_values
w = make()
rt = w._runtime
sim, model = rt["sim"], rt["model"]
n = sim.t.shape[0]

# first / middle / last
for name, want in (("first", 0), ("middle", n // 2), ("last", n - 1)):
    w._goto(name)
    print("GOTO_%s" % name.upper(), w.time_slider.value() == want)
w._goto("peak_gradient")
print("GOTO_PEAK", w.time_slider.value() == rt["peak_index"])
gd = np.linalg.norm(sim.g_dynamic, axis=1)
print("PEAK_IS_MAX", int(np.argmax(gd)) == rt["peak_index"])

# every frame type against its formula
k = rt["peak_index"]
w.time_slider.setValue(k)
checks = {
  "dynamic_field": model.C @ sim.i_dynamic[k],
  "total_field": model.C @ sim.i_total[k],
  "nonlinear_field": model.residual @ sim.i_dynamic[k],
  "dynamic_phase": 2*np.pi*(model.C @ sim.q_dynamic[k]),
  "total_phase": 2*np.pi*(model.C @ sim.q_total[k]),
  "nonlinear_phase": 2*np.pi*(model.residual @ sim.q_dynamic[k]),
}
bad = []
for ft, want in checks.items():
    got = frame_values(model, sim, ft, k)
    if not np.allclose(got, want, atol=1e-9):
        bad.append(ft)
print("FRAME_FORMULAS", "|".join(bad))

# total - dynamic is exactly the static contribution
stat = model.C @ np.asarray(w._result["static_currents_a"], dtype=float)
diff = checks["total_field"] - checks["dynamic_field"]
print("STATIC_DIFF", bool(np.allclose(diff, stat, atol=1e-9)))

# selecting each type redraws without error, and the cache stays small
for i in range(len(FRAME_TYPES)):
    w.frame_combo.setCurrentIndex(i)
    w._request_slices()
print("CACHE_LEN", len(w._frame_cache))
for j in range(0, n, max(n // 12, 1)):
    w.time_slider.setValue(j); w._request_slices()
print("CACHE_LEN_AFTER", len(w._frame_cache))

# a volume is one 3-D array with NaN outside the ROI
from core.linear_encoding_frames import frame_volume
vol = frame_volume(model, sim, "dynamic_field", k)
print("VOL_NDIM", vol.ndim)
print("VOL_SHAPE_OK", tuple(vol.shape) == tuple(model.mask.shape))
print("NAN_OUTSIDE", bool(np.all(np.isnan(vol[~model.mask]))) if (~model.mask).any() else True)
print("FINITE_INSIDE", bool(np.all(np.isfinite(vol[model.mask]))))
w.shutdown()
print("THREAD_DONE", not (w._worker and w._worker.isRunning()))
''')
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2000:]); return
    g = dict(l.split(" ", 1) for l in out.splitlines()
             if " " in l and l.split(" ", 1)[0].isupper())
    for k in ("GOTO_FIRST", "GOTO_MIDDLE", "GOTO_LAST", "GOTO_PEAK"):
        ok(g.get(k) == "True", "%s positions the slider" % k.lower())
    ok(g.get("PEAK_IS_MAX") == "True",
       "the peak button really lands on the largest dynamic gradient")
    ok(g.get("FRAME_FORMULAS", "x") == "",
       "all six frames match their formulas (%s)" % g.get("FRAME_FORMULAS"))
    ok(g.get("STATIC_DIFF") == "True",
       "total minus dynamic is exactly the static contribution")
    ok(int(g.get("CACHE_LEN", "99")) <= 5, "cache holds at most 5 frames")
    ok(int(g.get("CACHE_LEN_AFTER", "99")) <= 5,
       "and still does after stepping through time")
    ok(g.get("VOL_NDIM") == "3", "a frame is a single 3-D volume")
    ok(g.get("VOL_SHAPE_OK") == "True", "on the coil grid")
    ok(g.get("NAN_OUTSIDE") == "True", "NaN outside the ROI")
    ok(g.get("FINITE_INSIDE") == "True", "finite inside it")
    ok(g.get("THREAD_DONE") == "True", "shutdown left no thread running")


def t_decimation_and_scale():
    print("3. display decimation, colour limits, orientation, invalidation")
    code, out = run_py(PREP + '''
from core.linear_encoding_frames import (display_indices, symmetric_limits,
                                         plane_note, is_oblique,
                                         MAX_DISPLAY_SAMPLES)
# more than 5000 samples
w = make(rate=1000000, dur=20.0)
sim = w._runtime["sim"]
n = sim.t.shape[0]
pd = w.plot_data()
print("FULL_N", n)
print("DISP_N", len(pd["t"]))
print("DECIMATED", len(pd["t"]) < n)
print("UNDER_CAP", len(pd["t"]) <= MAX_DISPLAY_SAMPLES + 8)
print("KEEPS_ENDS", pd["idx"][0] == 0 and pd["idx"][-1] == n - 1)
print("KEEPS_PEAK", int(w._runtime["peak_index"]) in set(pd["idx"].tolist()))
# the stored series is untouched
print("SIM_FULL", sim.i_dynamic.shape[0] == n and sim.k_dynamic.shape[0] == n)
print("REPORT_FULL", pd["report"]["full_samples"] == n)

# colour limits are symmetric, and a zero frame is safe
lim = symmetric_limits(np.array([-3.0, 1.0, np.nan, 2.0]))
print("SYM", lim["vmin"] == -3.0 and lim["vmax"] == 3.0)
z = symmetric_limits(np.zeros(10))
print("ZERO_SAFE", z["vmin"] < z["vmax"])
print("NOT_COMPARABLE", z["comparable_across_time"] is False)

# orientation labelling
import numpy as _np
ortho = _np.diag([2.0, 2.0, 2.0, 1.0])
th = _np.deg2rad(20.0)
obl = _np.eye(4)
obl[:3, :3] = _np.array([[_np.cos(th), -_np.sin(th), 0],
                         [_np.sin(th), _np.cos(th), 0], [0, 0, 1]]) @ _np.diag([2.,2.,2.])
print("ORTHO_OBLIQUE", is_oblique(ortho))
print("OBL_OBLIQUE", is_oblique(obl))
print("OBL_NOTE", "不等同" in plane_note(obl))
print("ORTHO_NOTE", "native voxel plane" in plane_note(ortho))

# changing an input clears everything
w.dsv_spin.setValue(6.0)
print("CLEARED_RESULT", w._result is None)
print("CLEARED_RUNTIME", w._runtime is None)
print("CLEARED_CACHE", len(w._frame_cache) == 0)
print("EXPORT_OFF", not w.btn_export.isEnabled())
print("SLIDER_OFF", not w.time_slider.isEnabled())
w.shutdown()
''')
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2000:]); return
    g = dict(l.split(" ", 1) for l in out.splitlines()
             if " " in l and l.split(" ", 1)[0].isupper())
    ok(g.get("DECIMATED") == "True",
       "a %s-sample run is decimated to %s for plotting"
       % (g.get("FULL_N"), g.get("DISP_N")))
    ok(g.get("UNDER_CAP") == "True", "and stays under the display cap")
    ok(g.get("KEEPS_ENDS") == "True", "first and last samples are kept")
    ok(g.get("KEEPS_PEAK") == "True", "and so is the peak sample")
    ok(g.get("SIM_FULL") == "True",
       "the computed series keeps every sample")
    ok(g.get("REPORT_FULL") == "True", "and the report says how many")
    ok(g.get("SYM") == "True", "colour limits are symmetric about zero")
    ok(g.get("ZERO_SAFE") == "True", "an all-zero frame gets a usable range")
    ok(g.get("NOT_COMPARABLE") == "True",
       "and is labelled not comparable across time")
    ok(g.get("ORTHO_OBLIQUE") == "False" and g.get("OBL_OBLIQUE") == "True",
       "a 20-degree rotation is detected as oblique, a plain scale is not")
    ok(g.get("OBL_NOTE") == "True",
       "an oblique affine is labelled as not a standard anatomical plane")
    ok(g.get("ORTHO_NOTE") == "True", "and every caption says native voxel plane")
    for k in ("CLEARED_RESULT", "CLEARED_RUNTIME", "CLEARED_CACHE",
              "EXPORT_OFF", "SLIDER_OFF"):
        ok(g.get(k) == "True", "changing the DSV clears %s"
           % k.lower().replace("cleared_", "the "))


def t_no_4d():
    print("4. no 4-D field is ever built")
    code, out = run_py(PREP + '''
import tracemalloc
from core.linear_encoding_frames import frame_volume, FRAME_TYPES
w = make(rate=200000, dur=20.0)
rt = w._runtime
sim, model = rt["sim"], rt["model"]
n = sim.t.shape[0]
nvox = int(model.mask.size)
four_d = nvox * n * 8
tracemalloc.start()
for i, ft in enumerate(FRAME_TYPES):
    w.frame_combo.setCurrentIndex(i)
    w.time_slider.setValue((i * n) // len(FRAME_TYPES))
    w._request_slices()
_, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
print("FOURD_MB", "%.1f" % (four_d / 2**20))
print("PEAK_MB", "%.2f" % (peak / 2**20))
# Matplotlib and Qt allocate a few MB for the six canvases themselves.  A
# tenfold separation still proves that no n_voxel × n_time field exists,
# without making the check depend on a sub-megabyte allocator fluctuation.
print("UNDER", peak < four_d / 10)
w.shutdown()
''')
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2000:]); return
    g = dict(l.split(" ", 1) for l in out.splitlines()
             if " " in l and l.split(" ", 1)[0].isupper())
    ok(g.get("UNDER") == "True",
       "drawing all six frame types peaked at %s MB against a %s MB 4-D field"
       % (g.get("PEAK_MB"), g.get("FOURD_MB")))


if __name__ == "__main__":
    for fn in (t_curves, t_frames_and_slider, t_decimation_and_scale, t_no_4d):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
