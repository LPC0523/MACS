"""The research page: it must not reach hardware, and must not hide tolerances.

    D:\\python.exe tests/test_linear_encoding_ui.py

Runs in a subprocess that ends with os._exit(0) and on the native Qt platform,
for the reasons in tests/smoke_production.py: PySide6 crashes during
interpreter teardown here, after the window has done its work.
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


def run_py(body, timeout=600):
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


def t_builds():
    print("1. the page builds and joins the navigation")
    code, out = run_py("""
        from PySide6.QtWidgets import QApplication
        import main_window as mw
        app = QApplication([])
        win = mw.MainWindow()
        print("NAV", len(mw._NAV))
        print("PAGES", win.stack.count())
        print("ENCODING_TITLE", mw._NAV[-2][0])
        print("LAST_TITLE", mw._NAV[-1][0])
        print("HAS", hasattr(win, "page_encoding"))
        for i in range(win.stack.count()):
            win.stack.setCurrentIndex(i)
            app.processEvents()
        win.close(); app.processEvents()
        print("WALKED")
    """)
    ok(code == 0, "subprocess exited cleanly")
    nav = pages = None
    for line in out.splitlines():
        if line.startswith("NAV"):
            nav = int(line.split()[1])
        if line.startswith("PAGES"):
            pages = int(line.split()[1])
    ok(nav == pages, "nav rows == stack pages (%s == %s)" % (nav, pages))
    ok("ENCODING_TITLE 线性编码仿真" in out, "the encoding page is named correctly")
    ok("LAST_TITLE 系统自检" in out, "self-check is the final navigation page")
    ok("HAS True" in out, "MainWindow exposes page_encoding")
    ok("WALKED" in out, "every page still selectable with the new one present")


def t_no_hardware():
    print("2. the page cannot reach hardware")
    src = (ROOT / "widgets" / "linear_encoding_widget.py").read_text(
        encoding="utf-8")
    # Code, not prose. The module docstring says "no serial access", and a
    # naive substring search flags its own disclaimer -- so this looks for the
    # constructs that would actually reach hardware.
    code_lines = chr(10).join(
        l for l in src.splitlines()
        if not l.lstrip().startswith("#"))
    for bad in ("import serial", "from serial", "QSerialPort", "pyserial",
                "amp_bridge", "pending_currents", "CMD_LOAD_TABLE",
                "SerialReaderWorker", "teensy_widget", "write(", ".open("):
        ok(bad not in code_lines, "no %r in the widget" % bad)
    # And no button text that offers to act on hardware.
    for bad in ("上传", "发送", "执行编码"):
        ok(bad not in src, "no %r anywhere, including labels" % bad)

    code, out = run_py("""
        import sys
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import widgets.linear_encoding_widget as lew
        w = lew.LinearEncodingWidget({})
        # No button may offer to upload or execute.
        from PySide6.QtWidgets import QPushButton
        labels = [b.text() for b in w.findChildren(QPushButton)]
        print("BUTTONS", "|".join(labels))
        bad = [t for t in labels
               if any(k in t for k in ("上传", "发送", "执行编码", "Upload", "Send"))]
        print("BAD", "|".join(bad))
        print("SERIAL_IMPORTED", "serial" in sys.modules)
    """)
    ok(code == 0, "widget constructs standalone")
    # This assertion used to end in "or True", which made it pass no
    # matter what the page contained.
    ok(any(l.startswith("BUTTONS") for l in out.splitlines()),
       "button labels were captured")
    for line in out.splitlines():
        if line.startswith("BAD"):
            ok(line.strip() == "BAD" or line.strip() == "BAD |",
               "no upload/execute button (%s)" % line[4:] or "none")
        if line.startswith("SERIAL_IMPORTED"):
            ok(line.split()[1] == "False",
               "pyserial is not imported by building the page")


def t_tolerances_visible():
    print("3. both tolerances are on the page, not hidden")
    code, out = run_py("""
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import widgets.linear_encoding_widget as lew
        w = lew.LinearEncodingWidget({})
        print("THETA", w.theta_spin.value())
        print("EPS", w.eps_spin.value())
        print("THETA_HIDDEN", w.theta_spin.isHidden())
        print("EPS_HIDDEN", w.eps_spin.isHidden())
        print("LIMIT", w.limit_spin.value(), w.limit_spin.isEnabled())
        print("STRATEGY", w.strategy_combo.currentText()[:6])
        print("BANNER", lew.BANNER)
        # The tick comes from the coil config, not from a hard-coded default.
        info = w.coil_unit_info()
        print("UNIT_STATUS", info["status"])
        print("UNIT_IMPORTED", info["imported"])
        print("UNIT_SRC_SHOWN", len(w.unit_source.text()) > 10)
        print("UNIT_DEFAULT", w.unit_confirm.isChecked())
        print("RUN_DEFAULT", w.btn_run.isEnabled())
        w.unit_confirm.setChecked(False)
        print("RUN_UNCONFIRMED", w.btn_run.isEnabled())
        # The backend refuses too, not just the greyed button.
        from core.linear_encoding_run import run_simulation, SimulationError
        try:
            run_simulation(w.collect_params(), {"_loaded": object()})
            print("BACKEND_GATE", False)
        except SimulationError as e:
            print("BACKEND_GATE", "单位" in str(e))
        w.unit_confirm.setChecked(True)
        print("RUN_RECONFIRMED", w.btn_run.isEnabled())
    """)
    ok(code == 0, "subprocess exited cleanly")
    ok("THETA 5.0" in out, "theta defaults to 5 degrees")
    ok("EPS 0.3" in out, "epsilon is shown with a value")
    ok("THETA_HIDDEN False" in out and "EPS_HIDDEN False" in out,
       "neither tolerance is hidden")
    ok("LIMIT 3.0 False" in out, "current limit is 3.0 A and not editable")
    ok("STRATEGY reject" in out, "reject is the default strategy")
    ok("BANNER 离线模型 · 不连接硬件" in out, "the compact offline badge is present")
    ok("UNIT_SRC_SHOWN True" in out, "the unit source is shown to the user")
    tick = "UNIT_DEFAULT True" in out
    conf = "UNIT_STATUS user_confirmed" in out
    ok(tick == conf,
       "the box is ticked exactly when the coil config confirms the unit "
       "(ticked=%s, config=%s)" % (tick, conf))
    ok(("RUN_DEFAULT True" in out) == conf,
       "and the run button follows the same source, not a default")
    ok("RUN_UNCONFIRMED False" in out, "unchecking it blocks the button")
    ok("BACKEND_GATE True" in out,
       "and the backend refuses too, not just the greyed button")
    ok("RUN_RECONFIRMED True" in out, "re-checking restores it")


def t_coverage_verdict():
    print("4. coverage below 95% is shown as TRUNCATED, in red")
    code, out = run_py("""
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import widgets.linear_encoding_widget as lew
        w = lew.LinearEncodingWidget({})
        w.show_coverage({"coverage": 0.462, "valid_sphere_voxels": 247272,
                         "theoretical_sphere_voxels": 535432, "truncated": True})
        print("TRUNC_TEXT", w.coverage_label.text()[:20])
        print("TRUNC_STYLE", w.coverage_label.styleSheet())
        w.show_coverage({"coverage": 0.999, "valid_sphere_voxels": 125471,
                         "theoretical_sphere_voxels": 125616, "truncated": False})
        print("FULL_TEXT", w.coverage_label.text()[:12])
        print("FULL_STYLE", w.coverage_label.styleSheet())
    """)
    ok(code == 0, "subprocess exited cleanly")
    ok("TRUNC_TEXT TRUNCATED" in out, "a truncated sphere says TRUNCATED")
    ok("#e53e3e" in out.split("FULL_STYLE")[0], "and is red")
    ok("FULL_TEXT 完整 DSV" in out, "a covered sphere is called a full DSV")


def t_headline_carries_tolerances():
    print("5. no gradient number is shown without epsilon and theta")
    code, out = run_py("""
        from PySide6.QtWidgets import QApplication
        app = QApplication([])
        import widgets.linear_encoding_widget as lew
        w = lew.LinearEncodingWidget({})
        # Shaped like a real run result, because show_result now reads the
        # keys the backend actually produces.
        w.show_result({
            "actual_peak_gradient_hz_per_m": 14390.0,
            "actual_peak_gradient_mt_per_m": 0.338,
            "requested_peak_gradient_hz_per_m": 15000.0,
            "capacity_gradient_hz_per_m": 14390.0,
            "capacity": {"feasible": True, "solver": "CLARABEL",
                         "status": "optimal"},
            "coverage": 0.999, "uniform_scale": 1.0,
            "nonlinear_spatial_ratio": 0.30,
            "nonlinear_spatial_rms_hz": 120.0,
            "channels_at_current_bound": ["ch3"],
            "gradient_parallel_actual_hz_per_m": 14390.0,
            "gradient_parallel_actual_mt_per_m": 0.338,
            "gradient_magnitude_actual_hz_per_m": 14401.0,
            "direction_error_deg": 5.0, "orthogonal_leakage_ratio": 0.087,
            "nonlinear_residual_ratio": 0.30,
            "nonlinear_residual_rms_hz": 120.0,
            "target_fit_nrmse": 0.31, "epsilon_nonlinear": 0.30,
            "theta_max_deg": 5.0, "region_name": "DSV 12.0 cm",
            "current_peak_a": 3.0, "saturated_channels": ["ch3"],
            "solver": "CLARABEL", "status": "optimal",
            "box_violation_a": 0.0,
            "nonlinear_constraint_relative_violation": 0.0,
            "direction_cone_relative_violation": 0.0,
            "per_channel": [{"channel": "ch1", "static_a": 0.0,
                             "dynamic_peak_abs_a": 1.2, "peak_abs_a": 1.2,
                             "headroom_up_a": 1.8, "headroom_down_a": 1.8,
                             "saturated": False}],
        })
        # Keep the compact two-line product headline on one subprocess line so
        # this assertion checks both rows without forcing explanatory prose
        # back into the UI.
        print("HEAD", w.headline.text().replace("\\n", " | "))
        print("OUTPUT_TABS", w.output_tabs.count())
        print("PLOT_TABS", w.plot_tabs.count())
        print("HELP", w.btn_help.text())
        rows = {w.metric_table.item(r, 0).text(): w.metric_table.item(r, 1).text()
                for r in range(w.metric_table.rowCount())}
        print("UNKNOWN", rows.get("电压 / 功率 / 动态可实现性"))
        print("COUPLING", rows.get("梯度耦合"))
        print("CLIP", rows.get("clipping"))
        print("EXPORT", w.btn_export.isEnabled())
    """)
    ok(code == 0, "subprocess exited cleanly")
    if code != 0:
        print(out[-2500:])
        return
    head = next((l for l in out.splitlines() if l.startswith("HEAD")), "")
    ok("0.338" in head, "the headline shows mT/m")
    ok("ε=0.3" in head, "the headline carries epsilon")
    ok("θ=5.0°" in head, "the headline carries theta")
    ok("DSV 12.0 cm" in head, "and the region")
    ok("上界" in head, "and shows the capacity")
    ok("请求" in head, "and shows the requested amplitude alongside")
    ok("缩放" in head, "and shows the applied current scaling")
    ok("OUTPUT_TABS 2" in out, "results and visualisation are separated")
    ok("PLOT_TABS 4" in out, "the four plots use tabs instead of a long stack")
    ok("HELP 使用说明" in out, "detailed guidance moved behind a help action")
    ok("UNKNOWN UNKNOWN" in out, "voltage and power are UNKNOWN, not 0")
    ok("COUPLING not_characterized" in out, "coupling is not_characterized")
    ok("CLIP 从未使用" in out, "clipping is reported as never used")


if __name__ == "__main__":
    for fn in (t_builds, t_no_hardware, t_tolerances_visible,
               t_coverage_verdict, t_headline_carries_tolerances):
        fn()
    print()
    if FAILS:
        print("%d FAILED" % len(FAILS))
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all passed")
