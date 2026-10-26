"""Production smoke test: does the shipping app still start, run and behave?

    D:\\python.exe tests/smoke_production.py

Run before packaging. Each check is something that has actually broken in this
project at least once, which is why they are named individually rather than
being a generic import sweep.

Two things about how the Qt checks are run:

* They go in a **subprocess**, and that subprocess ends with `os._exit(0)`.
  PySide6 crashes during interpreter teardown on this machine -- after the
  window has been built, every page selected, and the window closed, all of
  which succeed. Ending the process before Python starts tearing down Qt
  objects avoids an exit code that says "failed" about work that finished.
  If a Qt check fails for a real reason it prints and is caught before that.
* They use the **native** platform, not `offscreen`. The offscreen plugin
  crashes on this app's pages; the native one does not, and the native one is
  what ships.

Nothing here opens a serial port or writes to hardware.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

RESULTS = []

APP_MODULES = [
    "main", "main_window", "core.version", "core.paths",
    "core.B0_cal_algorithm", "core.coil_registry", "core.data_context",
    "core.dicom_to_nifti", "core.fsl_replace", "core.hybrid_shim",
    "core.pipeline", "core.physics", "core.shim_optimize", "core.shim_state",
    "core.adaptive_joint", "core.shim_metrics", "core.dynamic_shim",
    "core.amp_bridge", "core.run_report", "core.roi_registry", "core.errors",
    "core.cpp_bet_bridge", "core.ants_io", "core.theme",
    "widgets.b0_widget", "widgets.error_card", "widgets.optimize_widget",
    "widgets.overview_widget", "widgets.results_widget",
    "widgets.selfcheck_widget", "widgets.shim_widget",
    "widgets.system_shim_widget", "widgets.teensy_widget",
    "workers.optimization_worker", "workers.pipeline_worker",
    "workers.serial_worker", "workers.shim_pipeline_worker",
    "workers.system_shim_worker",
]

RESEARCH_ONLY = ["core.cvar_shim", "core.pfc_guarded_joint",
                 "core.controllability_audit", "core.subject_split"]


def check(name):
    def deco(fn):
        def run():
            try:
                detail = fn()
                RESULTS.append((name, True, detail or ""))
                print("  [OK]   %-40s %s" % (name, detail or ""))
            except Exception as e:
                RESULTS.append((name, False, "%s: %s" % (type(e).__name__, e)))
                print("  [FAIL] %-40s %s: %s" % (name, type(e).__name__, e))
                traceback.print_exc()
        return run
    return deco


def run_py(body, timeout=900, native=True):
    """Run `body` in a subprocess that exits before Qt teardown."""
    src = textwrap.dedent("""
        import os, sys
        sys.path.insert(0, %r)
        try:
        %s
        except BaseException:
            import traceback; traceback.print_exc()
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(17)
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    """) % (str(ROOT), textwrap.indent(textwrap.dedent(body), "    "))
    r = subprocess.run([sys.executable, "-X", "utf8", "-u", "-c", src],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
        raise AssertionError("exit %d: %s"
                             % (r.returncode, " | ".join(out[-3:])[:220]))
    return (r.stdout or "").strip()


@check("1. app modules import")
def t_imports():
    ok, missing = [], []
    for m in APP_MODULES:
        try:
            importlib.import_module(m)
            ok.append(m)
        except ModuleNotFoundError as e:
            if m.split(".")[-1] in str(e):
                missing.append(m)
            else:
                raise
    if missing:
        raise AssertionError("absent: %s" % ", ".join(missing))
    return "%d modules" % len(ok)


@check("2. MACS --selftest")
def t_selftest():
    r = subprocess.run([sys.executable, "-X", "utf8", "main.py", "--selftest"],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=900, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "SELFTEST PASSED" not in out:
        bad = [l for l in out.splitlines() if "[FAIL]" in l]
        raise AssertionError("exit %d; %s" % (r.returncode, " | ".join(bad)[:200]))
    n_ok = sum(1 for l in out.splitlines() if "[OK]" in l)
    ver = next((l for l in out.splitlines() if l.startswith("MACS v")), "")
    return "%d checks, %s" % (n_ok, ver.split("—")[0].strip())


@check("3. window builds, every page selectable, closes")
def t_window():
    out = run_py("""
        from PySide6.QtWidgets import QApplication
        import main_window as mw
        app = QApplication([])
        win = mw.MainWindow()
        n = win.stack.count()
        for i in range(n):
            win.stack.setCurrentIndex(i)
            app.processEvents()
        win.close()
        app.processEvents()
        print("PAGES %d" % n)
    """)
    n = [l for l in out.splitlines() if l.startswith("PAGES")]
    if not n:
        raise AssertionError("no page count reported: %s" % out[-160:])
    return "%s pages, each selected, window closed" % n[0].split()[1]


@check("4. research modules unreachable from app")
def t_isolation():
    import re
    pat = re.compile(r"\b(cvar_shim|pfc_guarded_joint|controllability_audit"
                     r"|subject_split)\b")
    hits = []
    for sub in (".", "core", "widgets", "workers"):
        for p in (ROOT / sub).glob("*.py"):
            if p.stem in ("cvar_shim", "pfc_guarded_joint",
                          "controllability_audit", "subject_split"):
                continue
            if pat.search(p.read_text(encoding="utf-8", errors="ignore")):
                hits.append(str(p.relative_to(ROOT)))
    if hits:
        raise AssertionError("app code references: %s" % ", ".join(hits))
    leaked = [m for m in RESEARCH_ONLY if m in sys.modules]
    if leaked:
        raise AssertionError("imported transitively: %s" % ", ".join(leaked))
    return "no references, none imported transitively"


@check("5. shim envelope loads from config")
def t_bounds():
    from core.hybrid_shim import SH12
    from core.shim_state import SystemShimState, load_bounds

    _, _, src = load_bounds()
    if not str(src).endswith("shim_delta_bounds.json"):
        raise AssertionError("fell back to built-in defaults: %s" % src)
    st = SystemShimState.default()
    if len(st.delta_bounds(SH12)) != len(SH12):
        raise AssertionError("window size mismatch")
    st.baseline["B11"] = -1757.7
    lo, hi = st.delta_bounds(SH12)[SH12.index("B11")]
    if abs(lo) >= abs(hi):
        raise AssertionError("B11 window not asymmetric: (%.1f, %.1f)" % (lo, hi))
    return "%s; B11@-1757.7 -> (%.1f, %.1f)" % (Path(src).name, lo, hi)


@check("6. C++ BET end to end")
def t_bet():
    p = ROOT / "tools" / "bet" / "verify_bet_e2e.py"
    if not p.is_file():
        raise FileNotFoundError("tools/bet/verify_bet_e2e.py missing")
    r = subprocess.run([sys.executable, "-X", "utf8", str(p)], cwd=str(ROOT),
                       capture_output=True, text=True, timeout=2400,
                       encoding="utf-8", errors="replace")
    out = ((r.stdout or "") + (r.stderr or "")).strip().splitlines()
    if r.returncode != 0:
        raise AssertionError("exit %d: %s" % (r.returncode,
                                              " | ".join(out[-3:])[:200]))
    return (out[-1] if out else "passed")[:64]


@check("7. B0 -> optimise pipeline end to end")
def t_pipeline():
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "test_pipeline.py")],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=2400, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "all checks passed" not in out:
        raise AssertionError("exit %d: %s" % (
            r.returncode, " | ".join(out.strip().splitlines()[-3:])[:200]))
    line = [l for l in out.splitlines() if "ROI RMSE" in l]
    return line[0].strip()[:56] if line else "passed"


@check("8. non-ROI region definition")
def t_rest():
    r = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "test_rest_region.py")],
                       cwd=str(ROOT), capture_output=True, text=True,
                       timeout=1200, encoding="utf-8", errors="replace")
    out = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0 or "all passed" not in out:
        raise AssertionError("exit %d: %s" % (
            r.returncode, " | ".join(out.strip().splitlines()[-3:])[:200]))
    return "rest = brain & ~roi, FOV-invariant"


@check("9. version stamp readable")
def t_version():
    from core.version import banner, version_string

    v = version_string()
    if not v or not str(v).strip():
        raise AssertionError("version_string() is empty")
    banner()
    return str(v)


def main():
    print("MACS production smoke test")
    print("  root   : %s" % ROOT)
    print("  python : %s" % sys.version.split()[0])
    print()
    for fn in (t_imports, t_selftest, t_window, t_isolation, t_bounds,
               t_bet, t_pipeline, t_rest, t_version):
        fn()
    print()
    bad = [n for n, ok, _ in RESULTS if not ok]
    print("%d/%d checks passed" % (len(RESULTS) - len(bad), len(RESULTS)))
    if bad:
        print("FAILED: %s" % ", ".join(bad))
        return 1
    print("ready to package")
    return 0


if __name__ == "__main__":
    sys.exit(main())
