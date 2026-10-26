"""System self-check: what is installed, and whether the coils are intact.

The failures this catches are the ones that are otherwise discovered halfway
through a scan session -- a missing ONNX model, an atlas that did not get
packaged, a coil whose field maps were replaced with a different revision. All
of them are cheap to test and expensive to hit in front of a subject.

Coil checks in particular are not just "does it load": a coil that loads and
produces plausible-looking currents from stale field maps is the failure mode
with no downstream symptom at all.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)

from core.theme import palette
from core.version import banner, build_info


_MARK = {"ok": "✓", "warning": "⚠", "error": "✗"}


class _CheckWorker(QThread):
    """Runs the checks off the UI thread -- hashing field maps takes seconds."""

    row = Signal(str, str, str, str)     # group, name, level, detail
    done = Signal()

    def __init__(self, deep: bool, parent=None):
        super().__init__(parent)
        self.deep = deep

    def run(self):
        # Checked between sections so closing the window during a run stops
        # promptly. Without it the only ways out are waiting for the whole
        # sweep or terminating the thread, and terminating a QThread that is
        # executing Python kills it holding the GIL and hangs the process.
        for section in (self._environment, self._models, self._coils):
            if self.isInterruptionRequested():
                return
            section()
        if not self.isInterruptionRequested():
            self.done.emit()

    # ── sections ─────────────────────────────────────────────────────
    def _environment(self):
        import sys

        info = build_info()
        self.row.emit("环境", "版本", "ok",
                      f"{info['version']} · build {info['build_date']} · {info['git_sha']}")
        self.row.emit("环境", "运行方式", "ok",
                      "打包版" if getattr(sys, "frozen", False) else "开发环境")

        for mod, label in (("numpy", "numpy"), ("scipy", "scipy"),
                           ("nibabel", "nibabel"), ("ants", "ANTsPy（前额叶配准）"),
                           ("serial", "pyserial（Teensy）"),
                           ("pydicom", "pydicom（DICOM 读取）")):
            try:
                m = __import__(mod)
                ver = getattr(m, "__version__", "")
                self.row.emit("环境", label, "ok", ver)
            except Exception as e:
                level = "error" if mod in ("numpy", "scipy", "nibabel", "pydicom") else "warning"
                self.row.emit("环境", label, level, f"{type(e).__name__}: {e}")

    def _models(self):
        try:
            from core.paths import find_atlases

            mni, pfc = find_atlases()
            import os
            for label, p in (("MNI 图谱", mni), ("PFC mask", pfc)):
                if p and os.path.exists(p):
                    self.row.emit("模型与图谱", label, "ok", str(p))
                else:
                    self.row.emit("模型与图谱", label, "error", "未找到")
        except Exception as e:
            self.row.emit("模型与图谱", "图谱", "error", f"{type(e).__name__}: {e}")

        try:
            from core.cpp_bet_bridge import why_unavailable, diagnose

            err = why_unavailable()
            if err:
                self.row.emit("模型与图谱", "脑提取引擎 (cpp_bet)", "error", err)
            else:
                first = diagnose().splitlines()
                self.row.emit("模型与图谱", "脑提取引擎 (cpp_bet)", "ok",
                              first[0] if first else "可用")
        except Exception as e:
            self.row.emit("模型与图谱", "脑提取引擎 (cpp_bet)", "error",
                          f"{type(e).__name__}: {e}")

    def _coils(self):
        try:
            from core.coil_registry import validate_all
        except Exception as e:
            self.row.emit("线圈", "线圈配置", "error", f"{type(e).__name__}: {e}")
            return

        try:
            reports = validate_all(deep=self.deep)
        except Exception as e:
            self.row.emit("线圈", "线圈配置", "error", f"{type(e).__name__}: {e}")
            return

        if not reports:
            self.row.emit("线圈", "线圈配置", "error",
                          "没有找到任何线圈配置（config/coils/*.json）")
            return

        for rep in reports:
            detail = (f"{rep.channels_checked} 通道"
                      if rep.ok else
                      "；".join(i.message for i in rep.issues))
            self.row.emit("线圈", rep.coil_name, rep.level, detail)


class SelfCheckWidget(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = palette(False)
        self.worker = None
        self._groups = {}
        self._build_ui()
        self.run_checks(deep=False)     # cheap pass on first show

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("系统自检")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        sub = QLabel("开机检查依赖、模型、图谱与线圈标定的完整性")
        sub.setObjectName("pageSubtitle")
        root.addWidget(sub)

        row = QHBoxLayout()
        self.btn_quick = QPushButton("重新检查")
        self.btn_quick.clicked.connect(lambda: self.run_checks(deep=False))
        row.addWidget(self.btn_quick)

        self.btn_deep = QPushButton("深度检查（含场图校验）")
        self.btn_deep.setObjectName("primaryBtn")
        self.btn_deep.setToolTip(
            "逐通道读取场图、比对尺寸并计算内容哈希；线圈多时需要几秒。")
        self.btn_deep.clicked.connect(lambda: self.run_checks(deep=True))
        row.addWidget(self.btn_deep)

        self.summary = QLabel("")
        row.addWidget(self.summary)
        row.addStretch()
        root.addLayout(row)

        box = QGroupBox("检查结果")
        v = QVBoxLayout(box)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(["项目", "状态", "详情"])
        self.tree.setAlternatingRowColors(True)
        self.tree.setColumnWidth(0, 240)
        self.tree.setColumnWidth(1, 60)
        v.addWidget(self.tree)
        root.addWidget(box, stretch=1)

        self.banner = QTextEdit()
        self.banner.setReadOnly(True)
        self.banner.setMaximumHeight(52)
        self.banner.setPlainText(banner())
        root.addWidget(self.banner)

    # ── running ──────────────────────────────────────────────────────
    def run_checks(self, deep: bool = False):
        if self.worker is not None and self.worker.isRunning():
            return
        self.tree.clear()
        self._groups.clear()
        self._counts = {"ok": 0, "warning": 0, "error": 0}
        self.summary.setText("检查中 …")
        for b in (self.btn_quick, self.btn_deep):
            b.setEnabled(False)

        self.shutdown()          # never leave a previous sweep running
        self.worker = _CheckWorker(deep, parent=self)
        self.worker.row.connect(self._add_row)
        self.worker.done.connect(self._finish)
        self.worker.start()

    @Slot(str, str, str, str)
    def _add_row(self, group: str, name: str, level: str, detail: str):
        parent = self._groups.get(group)
        if parent is None:
            parent = QTreeWidgetItem(self.tree, [group, "", ""])
            parent.setExpanded(True)
            font = parent.font(0)
            font.setBold(True)
            parent.setFont(0, font)
            self._groups[group] = parent

        item = QTreeWidgetItem(parent, [name, _MARK.get(level, "?"), detail])
        item.setTextAlignment(1, Qt.AlignCenter)
        colour = {"ok": self.p["success"], "warning": self.p["warning"],
                  "error": self.p["danger"]}.get(level)
        if colour:
            from PySide6.QtGui import QColor
            item.setForeground(1, QColor(colour))
            if level != "ok":
                item.setForeground(2, QColor(colour))
        item.setToolTip(2, detail)
        self._counts[level] = self._counts.get(level, 0) + 1

    @Slot()
    def _finish(self):
        for b in (self.btn_quick, self.btn_deep):
            b.setEnabled(True)
        c = self._counts
        if c.get("error"):
            self.summary.setText(f"✗ {c['error']} 项不可用 · ⚠ {c.get('warning', 0)} 项警告")
            self.summary.setObjectName("statusErr")
        elif c.get("warning"):
            self.summary.setText(f"⚠ {c['warning']} 项警告，其余正常")
            self.summary.setObjectName("statusWarn")
        else:
            self.summary.setText(f"✓ 全部 {c.get('ok', 0)} 项正常")
            self.summary.setObjectName("statusOk")
        self.summary.style().unpolish(self.summary)
        self.summary.style().polish(self.summary)

    def shutdown(self):
        """Stop a running check sweep. Called by `MainWindow.closeEvent`."""
        w = getattr(self, "worker", None)
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.quit()
            w.wait(5000)
