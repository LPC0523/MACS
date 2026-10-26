"""MACS — Multi-channel Active Coil Shimming. Qt application entry point."""
import sys
import os

# Ensure core module is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _force_utf8_console() -> None:
    """Stop console writes from killing the pipeline on non-ASCII log text.

    The packaged build runs with console=True, and a Chinese Windows console
    is GBK. Printing a check mark then raises UnicodeEncodeError inside the
    worker thread, which aborted the whole B0 run with
    "'gbk' codec can't encode character '\\u2713'". Reconfiguring to UTF-8
    with errors='replace' fixes every such message at once, rather than
    hunting down individual glyphs.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass  # detached or already wrapped; nothing to do


_force_utf8_console()

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon
from core.version import banner, version_string
from main_window import MainWindow


def selftest() -> int:
    """`MACS.exe --selftest` — verify a build has everything it needs.

    Checks atlas resolution and the optional ANTsPy backend without opening a
    window, so a packaged build can be validated from the command line.
    """
    from core.paths import app_base, find_atlases

    print(banner())
    print(f"frozen   : {getattr(sys, 'frozen', False)}")
    print(f"app_base : {app_base()}")

    failures = []

    mni, pfc = find_atlases()
    for label, p in (("MNI atlas", mni), ("PFC mask", pfc)):
        if p and os.path.exists(p):
            print(f"[OK]   {label}: {p}")
        else:
            print(f"[FAIL] {label}: not found")
            failures.append(label)

    try:
        import ants
        print(f"[OK]   ANTsPy {ants.__version__}")
    except Exception as e:
        print(f"[FAIL] ANTsPy unavailable: {type(e).__name__}: {e}")
        failures.append("ANTsPy")

    for mod in ("numpy", "scipy", "nibabel", "serial", "matplotlib"):
        try:
            __import__(mod)
            print(f"[OK]   {mod}")
        except Exception as e:
            print(f"[FAIL] {mod}: {e}")
            failures.append(mod)

    # C++ brain extraction: the accurate BET backend. Without it the pipeline
    # falls back to HD-BET CLI, or refuses to run entirely.
    try:
        from core.cpp_bet_bridge import diagnose, why_unavailable
        err = why_unavailable()
        if err:
            print(f"[FAIL] cpp_bet: {err}")
            failures.append("cpp_bet")
        else:
            print("[OK]   cpp_bet (C++ U-Net brain extraction)")
        for line in diagnose().splitlines():
            print(f"       {line}")
    except Exception as e:
        print(f"[FAIL] cpp_bet bridge: {type(e).__name__}: {e}")
        failures.append("cpp_bet")

    # Linear encoding: the page is in the navigation, so its solver has to be
    # in the build. Importing cvxpy is the weaker claim -- the solvers are
    # compiled extensions reached through plugin modules, and a bundle can
    # carry the front end with no back end. So this solves a real cone
    # program: three perfectly linear channels whose +-3 A box gives exactly
    # 1200 Hz/m. No coil map, no serial port, no network.
    try:
        import cvxpy
        solvers = cvxpy.installed_solvers()
        usable = [s for s in ("CLARABEL", "SCS") if s in solvers]
        if usable:
            print(f"[OK]   cvxpy {cvxpy.__version__}, solvers: {', '.join(usable)}")
        else:
            print(f"[FAIL] no CLARABEL or SCS; have {solvers}")
            failures.append("SOCP solver")
    except Exception as e:
        print(f"[FAIL] cvxpy unavailable: {type(e).__name__}: {e}")
        failures.append("cvxpy")
        usable = []

    try:
        import numpy as _np
        from core.linear_encoding_model import UnitStatus, build_model
        from core.linear_encoding_pareto import max_gradient_socp
        from core.linear_encoding_waveform import (WaveformSpec,
                                                   cumulative_integral,
                                                   generate)
        from core.spatial_grid import world_coordinates
        from workers.linear_encoding_worker import LinearEncodingWorker  # noqa

        _sh = (9, 9, 9)
        _a = _np.diag([4.0, 4.0, 4.0, 1.0]); _a[:3, 3] = [-16.0, -16.0, -16.0]
        _p = world_coordinates(_a, _sh)
        _maps = _np.stack([(400 * _p[:, 0]).reshape(_sh),
                           (300 * _p[:, 1]).reshape(_sh),
                           (250 * _p[:, 2]).reshape(_sh)], -1)
        _m = build_model(_maps, _np.ones(_sh, bool), _a,
                         unit_status=UnitStatus.USER_CONFIRMED)
        _r = max_gradient_socp(_m, [1, 0, 0], 0.30, 3.0, theta_max_deg=5.0)
        _g = _r.get("gradient_parallel_actual_hz_per_m", 0.0)
        if _r.get("feasible") and abs(_g - 1200.0) < 1.0:
            print(f"[OK]   linear encoding SOCP: {_g:.1f} Hz/m "
                  f"(expected 1200.0, solver {_r.get('solver')})")
        else:
            print(f"[FAIL] linear encoding SOCP gave {_g:.1f} Hz/m, "
                  f"status {_r.get('status')}")
            failures.append("linear encoding SOCP")

        _t, _w = generate(WaveformSpec("sine", amplitude=1.0, frequency=100.0,
                                       n_cycles=2, sample_rate_hz=20000.0))
        _q = cumulative_integral(_t, _w)
        if abs(float(_q[-1])) < 1e-6:
            print("[OK]   waveform and phase/k integration")
        else:
            print(f"[FAIL] two whole sine cycles integrate to {_q[-1]:.3e}")
            failures.append("waveform integration")
    except Exception as e:
        print(f"[FAIL] linear encoding backend: {type(e).__name__}: {e}")
        failures.append("linear encoding backend")

    print("SELFTEST FAILED: " + ", ".join(failures) if failures else "SELFTEST PASSED")
    return 1 if failures else 0


def _hide_own_console() -> None:
    """Hide the console window, but only when it belongs to us.

    The build stays console=True on purpose: `MACS.exe --selftest` has to be
    able to print, and `scripts/check_package.py` reads that output over a
    pipe. A windowed build would give the GUI a clean launch and leave both
    with nowhere to write.

    So the console is created and then hidden. The ownership test is what makes
    that safe. Launched from Explorer or a shortcut, Windows creates a console
    for this process and hiding it removes the black window nobody asked for.
    Launched from an existing cmd or PowerShell, the process *inherits* that
    console -- hiding it would hide the user's own terminal mid-session, so the
    window's owning process ID is compared against ours first and left alone
    when it differs.

    Not called for --selftest: that is a command-line entry point and its whole
    purpose is the text it prints.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        u32 = ctypes.windll.user32
        hwnd = k32.GetConsoleWindow()
        if not hwnd:
            return                      # no console attached at all
        owner = wintypes.DWORD()
        u32.GetWindowThreadProcessId(wintypes.HWND(hwnd), ctypes.byref(owner))
        if owner.value != k32.GetCurrentProcessId():
            return                      # somebody else's terminal
        u32.ShowWindow(wintypes.HWND(hwnd), 0)      # SW_HIDE
    except Exception:
        # A missing console API is not a reason to refuse to start.
        pass


def main():
    _hide_own_console()
    print(banner())

    app = QApplication(sys.argv)
    app.setApplicationName("MACS")
    app.setApplicationVersion(version_string(short=True).lstrip("v"))
    app.setOrganizationName("MRI System Lab")

    # Load icon if available
    icon_path = os.path.join(os.path.dirname(__file__), "resources", "shimlab_icon.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    main()
