"""End to end: the three buttons, driven for real against synthetic coils.

    D:\\python.exe tests/test_linear_encoding_e2e.py

The point of this file is that nothing is faked. A synthetic coil is written to
disk as a real coil configuration, the page is pointed at it, and the buttons
are clicked -- `btn_coverage.click()`, `btn_run.click()`, `on_export(dir)`. The
numbers on screen come back through the worker from `linear_encoding_run`, and
the exported files are read back and compared against them.

`show_result(fake)` proves the table renders. It does not prove the button
works, which is what this checks.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FAILS = []


def ok(cond, msg):
    print("  [%s] %s" % ("OK" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def run_py(body, timeout=1200):
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


#: Builds a synthetic coil on disk and registers it, so the page can select it
#: exactly as it selects a real one.
SETUP = '''
import json, shutil, tempfile
import numpy as np, nibabel as nib

TMP = tempfile.mkdtemp(prefix="le_e2e_")
COIL = os.path.join(TMP, "coil_data", "SYNTH 3ch")
os.makedirs(COIL, exist_ok=True)
SH = (16, 16, 16)
aff = np.diag([4.0, 4.0, 4.0, 1.0]); aff[:3, 3] = [-30.0, -30.0, -30.0]
ii, jj, kk = np.meshgrid(*[np.arange(s) for s in SH], indexing="ij")
w = np.stack([ii, jj, kk, np.ones_like(ii)], -1).astype(float)
world = np.einsum("ij,...j->...i", aff, w)[..., :3] / 1000.0
x, y, z = world[..., 0], world[..., 1], world[..., 2]
fields = [400.0 * x + 30.0 * (x ** 2 - y ** 2),
          360.0 * y, 320.0 * z]
for j, f in enumerate(fields):
    d = os.path.join(COIL, "ch%d_nii" % (j + 1))
    os.makedirs(d, exist_ok=True)
    nib.save(nib.Nifti1Image(f.astype(np.float32), aff),
             os.path.join(d, "ch%d_nii_deltaFieldmap.nii.gz" % (j + 1)))

import core.coil_registry as CR
from pathlib import Path as _P
_orig = CR.list_coils
class _C:
    name = "SYNTH 3ch"; channel_count = 3; current_limit_a = 3.0
    fieldmap_dir = COIL; description = "synthetic"; is_default = False
    imported = False; fieldmap_version = ""
    # A synthetic stand-in for a shipped coil, so it carries the unit
    # confirmation in its own configuration the way config/coils/*.json do.
    field_unit = "Hz/A"; field_unit_status = "user_confirmed"
    field_unit_confirmed_date = "2026-08-11"
    field_unit_note = "synthetic test coil"; source_json = "synthetic"
    def channel_dirs(self):
        return [_P(COIL) / ("ch%d_nii" % (k + 1)) for k in range(3)]
_synth = _C()
CR.list_coils = lambda: [_synth]
CR.get_coil = lambda n=None: _synth
'''


def t_full_loop():
    print("1. coverage -> run -> export, all three buttons clicked")
    code, out = run_py(SETUP + '''
from PySide6.QtWidgets import QApplication
app = QApplication([])
import widgets.linear_encoding_widget as lew
w = lew.LinearEncodingWidget({})

def pump(worker, limit=600):
    import time
    t0 = time.time()
    while worker is not None and worker.isRunning() and time.time() - t0 < limit:
        app.processEvents(); time.sleep(0.01)
    app.processEvents()

# region: a DSV the synthetic support fully covers
w.region_combo.setCurrentIndex(0)
w.dsv_spin.setValue(4.0)
w.centre_combo.setCurrentIndex(0)
print("REGION_BEFORE", w._region is None)
w.btn_coverage.click()
pump(w._worker)
print("REGION_AFTER", w._region is not None)
print("COVERAGE", "%.4f" % w._region["coverage"], w._region["truncated"])
print("COVERAGE_LABEL", w.coverage_label.text()[:8])

# unit is confirmed by default now
print("UNIT", w.unit_confirm.isChecked())

# target-gradient run
w.mode_combo.setCurrentIndex(0)
w.wave_combo.setCurrentText("sine")
w.amp_spin.setValue(500.0)
w.freq_spin.setValue(100.0)
w.dur_spin.setValue(20.0)
w.rate_spin.setValue(50000)
w.eps_spin.setValue(0.30)
w.theta_spin.setValue(5.0)
w.strategy_combo.setCurrentIndex(1)          # uniform scale
print("HEAD_BEFORE", w.headline.text()[:6])
w.btn_run.click()
pump(w._worker)
r = w._result
print("RESULT", r is not None)
print("HEAD_AFTER", w.headline.text()[:12])
print("REQ", "%.1f" % r["requested_peak_gradient_hz_per_m"])
print("CAP", "%.1f" % r["capacity_gradient_hz_per_m"])
print("ACT", "%.1f" % r["actual_peak_gradient_hz_per_m"])
print("SCALE", "%.6f" % r["uniform_scale"])
print("EPSUSED", r["epsilon_nonlinear"])
print("THETAUSED", r["theta_max_deg"])
print("KDYN", "%.6f" % r["k_dynamic_peak_cycles_per_m"])
print("KTOT", "%.6f" % r["k_total_local_peak_cycles_per_m"])
sim = r["_sim"]
import numpy as np
print("DYN_TOT_DIFFER", bool(np.allclose(sim.i_dynamic, sim.i_total)))
print("PEAK_CUR", "%.4f" % float(np.abs(sim.i_total).max()))
print("EXPORT_ENABLED", w.btn_export.isEnabled())

# export
import tempfile
d = tempfile.mkdtemp(prefix="le_out_")
written = w.on_export(d)
from core.linear_encoding_export import EXPORT_FILES
missing = [f for f in EXPORT_FILES if not os.path.isfile(os.path.join(d, f))]
print("EXPORT_MISSING", "|".join(missing))
import json, csv as _csv
met = json.load(open(os.path.join(d, "metrics.json"), encoding="utf-8"))
print("JSON_ACT", "%.1f" % met["actual_peak_gradient_hz_per_m"])
print("JSON_EPS", met["epsilon_nonlinear"])
rows = list(_csv.reader(
    [l for l in open(os.path.join(d, "currents.csv"), encoding="utf-8")
     if not l.startswith("#")]))
print("CSVROWS", len(rows) - 1)
print("CSVCOLS", len(rows[0]))
krows = list(_csv.reader(
    [l for l in open(os.path.join(d, "k_offset.csv"), encoding="utf-8")
     if not l.startswith("#")]))
kmax = max(abs(float(r[1])) for r in krows[1:])
print("CSV_KX_MAX", "%.6f" % kmax)
rep = open(os.path.join(d, "report.md"), encoding="utf-8").read()
print("REPORT_IDEAL", "IDEALIZED" in rep)
print("REPORT_CAP", "模型可达上界" in rep)
w.shutdown()
print("WORKER_DONE", not (w._worker and w._worker.isRunning()))
''')
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2500:])
        return
    g = dict(l.split(" ", 1) for l in out.splitlines() if " " in l
             and l.split(" ", 1)[0].isupper())
    ok(g.get("REGION_BEFORE") == "True" and g.get("REGION_AFTER") == "True",
       "coverage button built the region")
    ok(g.get("COVERAGE", "0").startswith("1.0"), "synthetic DSV fully covered")
    ok("完整" in g.get("COVERAGE_LABEL", ""), "and is labelled a full DSV")
    ok(g.get("UNIT") == "True", "Hz/A is confirmed by default")
    ok(g.get("HEAD_BEFORE") == "尚未运行。"[:6], "headline started empty")
    ok(g.get("RESULT") == "True", "run button produced a real result")
    ok("模型梯度" in g.get("HEAD_AFTER", ""),
       "headline now comes from the backend (%s)" % g.get("HEAD_AFTER"))
    req, cap, act = (float(g["REQ"]), float(g["CAP"]), float(g["ACT"]))
    ok(req == 500.0, "requested peak is what was typed (%.1f)" % req)
    ok(cap > 0, "capacity is a real SOCP result (%.1f Hz/m)" % cap)
    ok(act > 0, "actual peak is nonzero (%.1f Hz/m)" % act)
    ok(abs(act - min(req, cap)) / max(act, 1.0) < 0.05
       or float(g["SCALE"]) < 1.0,
       "actual tracks the smaller of requested and capacity, or was scaled")
    ok(g.get("EPSUSED") == "0.3" and g.get("THETAUSED") == "5.0",
       "epsilon and theta reached the solver (eps=%s, theta=%s)"
       % (g.get("EPSUSED"), g.get("THETAUSED")))
    ok(g.get("DYN_TOT_DIFFER") == "True",
       "with no static baseline dynamic == total, as it should be")
    ok(float(g["PEAK_CUR"]) <= 3.0 + 1e-6,
       "every current stays inside +-3 A (%.4f)" % float(g["PEAK_CUR"]))
    ok(float(g["KDYN"]) > 0, "k_dynamic is nonzero (%.6f)" % float(g["KDYN"]))
    ok(g.get("EXPORT_ENABLED") == "True", "export became available")
    ok(g.get("EXPORT_MISSING", "x") == "", "all seven files written")
    ok(abs(float(g["JSON_ACT"]) - act) < 1e-6,
       "metrics.json agrees with the on-screen result")
    ok(g.get("JSON_EPS") == "0.3", "and carries epsilon")
    # Was "or True". The header has time plus dynamic and total columns
    # for every channel, so 1 + 2*3 = 7 for the synthetic array.
    ok(int(g.get("CSVCOLS", "0")) == 7,
       "currents.csv keeps dynamic and total apart (%s columns)"
       % g.get("CSVCOLS"))
    ok(abs(float(g["CSV_KX_MAX"]) - float(g["KDYN"])) / max(float(g["KDYN"]), 1e-9)
       < 0.2, "k_offset.csv matches the reported k peak")
    ok(g.get("REPORT_IDEAL") == "True", "report carries the IDEALIZED notice")
    ok(g.get("REPORT_CAP") == "True", "and the capacity wording")
    ok(g.get("WORKER_DONE") == "True", "shutdown left no running worker")


def t_guards():
    print("2. the guards actually block")
    code, out = run_py(SETUP + '''
from PySide6.QtWidgets import QApplication
app = QApplication([])
import widgets.linear_encoding_widget as lew
import numpy as np, tempfile, time

def pump(w, limit=600):
    t0 = time.time()
    while w._worker is not None and w._worker.isRunning() and time.time()-t0 < limit:
        app.processEvents(); time.sleep(0.01)
    app.processEvents()

def fresh(dsv=4.0):
    w = lew.LinearEncodingWidget({})
    w.dsv_spin.setValue(dsv); w.btn_coverage.click(); pump(w)
    return w

# (a) unconfirmed unit blocks the run
w = fresh()
w.unit_confirm.setChecked(False)
w.btn_run.click(); pump(w)
print("UNCONF_RESULT", w._result is None)
print("UNCONF_RUN_ENABLED", w.btn_run.isEnabled())

# (b) zero custom direction is refused
w = fresh()
w.dir_combo.setCurrentIndex(3)
w.dir_x.setValue(0.0); w.dir_y.setValue(0.0); w.dir_z.setValue(0.0)
w.btn_run.click(); pump(w)
print("ZERODIR_RESULT", w._result is None)

# (c) changing an input invalidates the region
w = fresh()
print("REGION_OK", w._region is not None)
w.dsv_spin.setValue(6.0)
print("REGION_INVALIDATED", w._region is None)
w.btn_run.click(); pump(w)
print("NO_REGION_RESULT", w._result is None)

# (d) truncated DSV blocks the capacity mode
w = lew.LinearEncodingWidget({})
w.dsv_spin.setValue(30.0)        # far larger than the synthetic support
w.btn_coverage.click(); pump(w)
print("TRUNC", w._region["truncated"], "%.3f" % w._region["coverage"])
w.btn_run.click(); pump(w)
print("TRUNC_RESULT", w._result is None)
print("TRUNC_MSG", ("覆盖率" in w.headline.text()) or ("失败" in w.headline.text()))

# (e) reject vs uniform scale on an impossible request
w = fresh()
w.amp_spin.setValue(5.0e5)       # far beyond capacity
w.strategy_combo.setCurrentIndex(0)          # reject
w.btn_run.click(); pump(w)
print("REJECT_RESULT", w._result is None)
print("REJECT_EXPORT", w.btn_export.isEnabled())
w2 = fresh()
w2.amp_spin.setValue(5.0e5)
w2.strategy_combo.setCurrentIndex(1)         # uniform scale
w2.btn_run.click(); pump(w2)
if w2._result:
    r = w2._result
    print("SCALE_OK", r["uniform_scale"] < 1.0)
    print("SCALE_PEAK", "%.4f" % float(np.abs(r["_sim"].i_total).max()))
    dyn = r["_sim"].i_dynamic
    ratio = dyn[:, 0] / np.where(np.abs(dyn[:, 1]) > 1e-12, dyn[:, 1], np.nan)
    print("SCALE_SHAPE", bool(np.nanstd(ratio) < 1e-6))
else:
    print("SCALE_OK False")

# (f) direct CSV, right and wrong channel counts
w = fresh()
d = tempfile.mkdtemp()
good = os.path.join(d, "g.csv")
t = np.linspace(0, 0.01, 201)
with open(good, "w", encoding="utf-8") as f:
    f.write("time_s,ch1,ch2,ch3\\n")
    for k in range(t.size):
        f.write("%.9g,%.6g,%.6g,%.6g\\n"
                % (t[k], 1.5*np.sin(2*np.pi*100*t[k]), 0.0, 0.0))
bad = os.path.join(d, "b.csv")
with open(bad, "w", encoding="utf-8") as f:
    f.write("time_s,ch1,ch2\\n0,0,0\\n0.001,1,0\\n")
w.mode_combo.setCurrentIndex(1)
print("DIRECT_EPS_DISABLED", not w.eps_spin.isEnabled())
print("DIRECT_AMP_DISABLED", not w.amp_spin.isEnabled())
w.csv_path.setText(good)
w.btn_run.click(); pump(w)
print("CSV_OK", w._result is not None)
if w._result:
    print("CSV_MODE", w._result["mode"])
    print("CSV_CAP_NONE", w._result["capacity"] is None)
w3 = fresh(); w3.mode_combo.setCurrentIndex(1); w3.csv_path.setText(bad)
w3.btn_run.click(); pump(w3)
print("CSV_BAD_RESULT", w3._result is None)
print("CSV_BAD_MSG", "通道" in w3.headline.text() or "失败" in w3.headline.text())
print("BUTTONS_BACK", w3.btn_run.isEnabled() and w3.btn_coverage.isEnabled())
print("SERIAL_IMPORTED", "serial" in sys.modules)
''')
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2500:])
        return
    g = dict(l.split(" ", 1) for l in out.splitlines() if " " in l
             and l.split(" ", 1)[0].isupper())
    ok(g.get("UNCONF_RESULT") == "True" and g.get("UNCONF_RUN_ENABLED") == "False",
       "unconfirmed unit blocks the run, in the UI and in the backend")
    ok(g.get("ZERODIR_RESULT") == "True", "a zero direction vector is refused")
    ok(g.get("REGION_INVALIDATED") == "True",
       "changing the DSV invalidates the cached region")
    ok(g.get("NO_REGION_RESULT") == "True",
       "and running without a region produces nothing")
    ok(g.get("TRUNC", "").startswith("True"),
       "a 30 cm sphere on a 6 cm support is truncated (%s)" % g.get("TRUNC"))
    ok(g.get("TRUNC_RESULT") == "True",
       "and the capacity mode refuses to run on it")
    ok(g.get("REJECT_RESULT") == "True", "reject produces no result")
    ok(g.get("REJECT_EXPORT") == "False", "and leaves export disabled")
    ok(g.get("SCALE_OK") == "True", "uniform scale ran and scaled below 1")
    ok(float(g.get("SCALE_PEAK", "9")) <= 3.0 + 1e-6,
       "scaled currents stay inside +-3 A (%s)" % g.get("SCALE_PEAK"))
    ok(g.get("SCALE_SHAPE") == "True",
       "and the channel-to-channel ratio is unchanged")
    ok(g.get("DIRECT_EPS_DISABLED") == "True"
       and g.get("DIRECT_AMP_DISABLED") == "True",
       "direct mode disables epsilon, theta and the Hz/m amplitude")
    ok(g.get("CSV_OK") == "True", "a well-formed CSV runs")
    ok(g.get("CSV_MODE") == "direct_currents", "in direct mode")
    ok(g.get("CSV_CAP_NONE") == "True", "with no capacity claim attached")
    ok(g.get("CSV_BAD_RESULT") == "True", "a wrong channel count is rejected")
    ok(g.get("BUTTONS_BACK") == "True", "and the buttons come back after failure")
    ok(g.get("SERIAL_IMPORTED") == "False", "pyserial never imported")


if __name__ == "__main__":
    for fn in (t_full_loop, t_guards):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
