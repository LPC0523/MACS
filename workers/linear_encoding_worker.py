"""Background worker for the linear-encoding research page.

Loading a million-voxel field map, decomposing it and running an SOCP takes
seconds; doing that on the GUI thread freezes the window, and a frozen window
is indistinguishable from a crash. So it happens here.

The worker touches no Qt widget, no serial port and no production shim state.
It calls `core.linear_encoding_run`, which is plain functions, and hands back
plain dicts. Everything it knows how to do can therefore also be done by a test
without a window.
"""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from core.linear_encoding_run import (RegionError, SimulationError,
                                      prepare_region, run_simulation)


class LinearEncodingWorker(QThread):
    """One job: either build a region, or run a simulation on one."""

    progress = Signal(int, str)
    coverage_finished = Signal(dict)
    finished_ok = Signal(dict)
    failed = Signal(str, str)

    JOB_COVERAGE = "coverage"
    JOB_SIMULATE = "simulate"

    def __init__(self, job: str, params: dict, region=None, parent=None):
        super().__init__(parent)
        self.job = job
        self.params = dict(params or {})
        self.region = region

    def _report(self, pct, msg):
        # Interruption is checked between stages rather than inside the solver:
        # CVXPY cannot be interrupted mid-solve, and pretending otherwise would
        # leave a half-torn-down thread behind.
        if self.isInterruptionRequested():
            raise _Cancelled()
        self.progress.emit(int(pct), str(msg))

    def run(self):
        try:
            if self.job == self.JOB_COVERAGE:
                region = prepare_region(self.params, self._report)
                self.coverage_finished.emit(region)
            elif self.job == self.JOB_SIMULATE:
                if self.region is None:
                    raise SimulationError("请先计算区域与覆盖率。")
                res = run_simulation(self.params, self.region, self._report)
                self.finished_ok.emit(res)
            else:
                raise SimulationError("未知的任务类型：%r" % self.job)
        except _Cancelled:
            self.failed.emit("已取消。", "interrupted")
        except (RegionError, SimulationError) as e:
            self.failed.emit(str(e), getattr(e, "detail", ""))
        except Exception as e:  # noqa: BLE001 -- the UI must never be left hanging
            import traceback
            self.failed.emit("%s：%s" % (type(e).__name__, e),
                             traceback.format_exc())


class _Cancelled(Exception):
    """Raised inside the worker when the user asks it to stop."""
