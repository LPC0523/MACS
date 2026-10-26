"""System Shim page — retune the scanner's own shim, optionally with local coils.

Distinct from the one-click page by what it asks of the user. One-click decides
everything itself; this page needs to be told the console's present setting,
because the answer it gives is a *change* to that setting and there is no way
to read it from the data.

Two modes, one solver: unticking "联合线圈优化" passes zero coil channels
to `joint_optimize`, so the objective and the bounds handling are the same code
either way.

The page loads nothing itself. B0 map, brain mask and resampled coil field maps
all come from `app_state`, put there by the one-click run -- a second copy of
any of them could disagree with the first, and then the two pages answer the
same question differently with nothing on screen to explain why.

The objective is whole-brain RMSE. There is no temporal-lobe protection term.
"""
import os
import json
import numpy as np
import nibabel as nib

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QLineEdit, QTextEdit,
    QDoubleSpinBox, QFileDialog, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QSlider, QFrame,
    QCheckBox, QToolButton, QMenu,
)
from PySide6.QtCore import Qt, Slot, Signal
from PySide6.QtGui import QColor

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.hybrid_shim import (
    SH1, SH2, SH12, ORDER, UNIT,
    build_sh_basis, format_currents_64ch,
)
from core.paths import user_data_dir
from core.shim_state import (SystemShimState, TERMS, LABEL,
                             STATE_FILENAME, load_bounds,
                             write_default_bounds_file)
from core.theme import palette, style_axes
from workers.system_shim_worker import SystemShimWorker

#: Index of the one-click page in main_window._NAV.
_NAV_ONECLICK = 0

CLIM = 150.0  # Hz, display window for the before/after viewer


class SystemShimWidget(QWidget):
    #: Ask the main window to bring another page forward, by nav index.
    request_page = Signal(int)

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.worker = None

        # Everything here is adopted from app_state, never loaded on this page.
        self.b0 = None
        self.affine = None
        self.brain_mask = None
        self.basis = None
        self.coil = None
        self.coil_maps = None
        self.coil_names = []
        self.result = None

        self.shim_state = self._load_shim_state()
        #: Kept as a plain dict for the existing spin-box code.
        self.shim_values = self.shim_state.baseline
        self.p = palette(False)
        self._build_ui()
        # The headroom row depends on the baselines, so it cannot be filled
        # while the widgets are still being created.
        self._refresh_state_note()
        # Likewise the target-region note: showEvent refreshes it once there is
        # data, but until then the checkbox must not offer a region that does
        # not exist.
        self._refresh_roi_note()
        self._on_mode_changed(True)

    # ══════════════════════════════════════════════════════════════════
    #  Theme
    # ══════════════════════════════════════════════════════════════════
    def _status(self, label, kind):
        """Tag a status label so the global stylesheet colours it."""
        label.setObjectName({"ok": "statusOk", "err": "statusErr",
                             "warn": "statusWarn"}.get(kind, "hint"))
        label.style().unpolish(label)
        label.style().polish(label)

    def apply_theme(self, dark: bool, p: dict):
        self.p = p
        for fig in (self.fig_curr, self.fig_ba):
            fig.set_facecolor(p["fig_bg"])
        for key, lbl in self.cards.items():
            cur = lbl.styleSheet()
            if "color:" in cur and lbl.text() != "—":
                continue  # keeps the green/orange delta colour
            lbl.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {p['text']};")
        if self.result is not None:
            if len(self.result.get("pfc_names") or []) > 0:
                self._draw_currents()
            self._draw_viewer()
        else:
            self.canvas_curr.draw()
            self.canvas_ba.draw()

    # ══════════════════════════════════════════════════════════════════
    #  Persistent scanner shim state
    # ══════════════════════════════════════════════════════════════════
    @property
    def _state_file(self):
        """Last values used, re-filled on the next launch.

        Within one session the scanner's shim does not change between subjects,
        so re-typing eight numbers per run is pure friction.
        """
        return os.path.join(user_data_dir(), STATE_FILENAME)

    def _load_shim_state(self) -> SystemShimState:
        """Last saved state, or an all-zero default on first run.

        A missing or unreadable file is not worth reporting: zeros are the
        correct assumption for "we do not know the scanner's setting", which is
        exactly the situation on a fresh install.
        """
        try:
            return SystemShimState.load(self._state_file)
        except Exception:
            return SystemShimState.default()

    def _collect_state(self) -> SystemShimState:
        """Read the eight baseline boxes and their allowances back out of the UI."""
        st = self.shim_state
        for n, box in self.sh_boxes.items():
            st.baseline[n] = box.value()
        for n, box in self.bound_boxes.items():
            st.bounds[n] = box.value()
        return st

    def _apply_state_to_ui(self, st: SystemShimState):
        self.shim_state = st
        self.shim_values = st.baseline
        for n, box in self.sh_boxes.items():
            box.blockSignals(True)
            box.setValue(float(st.baseline.get(n, 0.0)))
            box.blockSignals(False)
        for n, box in self.bound_boxes.items():
            box.blockSignals(True)
            box.setValue(float(st.bounds.get(n, 0.0)))
            box.blockSignals(False)
        self._refresh_state_note()
        self._on_mode_changed(self.cb_joint.isChecked())

    def _refresh_state_note(self):
        st = self.shim_state
        frozen = [LABEL[n] for n, _ in TERMS if st.is_frozen(n)]
        self.state_note.setText(
            "固定不变：" + "、".join(frozen) if frozen else "")
        if st.notes:
            for n in st.notes:
                self.log.append(f"[Shim 状态] {n}")
            st.notes.clear()

    def _save_shim_state(self):
        st = self._collect_state()
        st.source = "manual" if st.source == "manual" else st.source
        try:
            path = st.save(self._state_file)
            self.shim_save_status.setText(f"✓ 已保存 → {path}")
            self._status(self.shim_save_status, "ok")
        except Exception as e:
            self.shim_save_status.setText(f"✗ {e}")
            self._status(self.shim_save_status, "err")

    def _write_bounds_file(self):
        try:
            path = write_default_bounds_file()
        except Exception as e:
            QMessageBox.warning(self, "写入失败", f"{type(e).__name__}: {e}")
            return
        QMessageBox.information(
            self, "已生成",
            f"上限配置已写到：\n{path}\n\n"
            "改完重启生效。单位与 Step 2 一致（一阶 uT/m，二阶 uT/m²），"
            "实测 delta 量级参考 A10≈44、A20≈195、B21≈−856、A22≈288。")

    def _import_shim_state(self, from_scanner: bool):
        """Load a shim state JSON.

        「从扫描仪读取」 and 「从文件导入」 differ only in where they start
        looking and what they record as the source -- both read the same
        format, because there is no live link to the scanner: what it exports
        is a file, and pretending otherwise would just hide where the numbers
        came from.
        """
        title = "选择扫描仪导出的匀场状态" if from_scanner else "选择匀场状态 JSON"
        path, _ = QFileDialog.getOpenFileName(self, title, "", "JSON (*.json);;所有文件 (*)")
        if not path:
            return
        try:
            st = SystemShimState.load(path)
        except ValueError as e:
            QMessageBox.warning(self, "读取失败", str(e))
            return
        st.source = "scanner" if from_scanner else "file"
        self._apply_state_to_ui(st)
        self.log.append(f"[Shim 状态] 已导入 {path}")
        self.log.append(f"[Shim 状态] {st.summary()}")

    # ══════════════════════════════════════════════════════════════════
    #  UI
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("系统匀场")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        # [2] One grey line instead of a "data source" panel: the fields are
        # facts to glance at, not something to read and act on.
        self.src_label = QLabel("")
        self.src_label.setObjectName("hint")
        self.src_label.setWordWrap(True)
        layout.addWidget(self.src_label)

        # [6] Shown alone, centred, when there is nothing to work with.
        self.empty_label = QLabel("请先在「一键匀场」加载数据并完成计算")
        self.empty_label.setObjectName("hint")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)
        layout.addWidget(self._build_mode())
        layout.addWidget(self._build_step2())
        layout.addWidget(self._build_step5())

        self.btn_run = QPushButton("联合优化")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setMinimumHeight(42)
        self.btn_run.clicked.connect(self._run)
        layout.addWidget(self.btn_run)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(160)
        layout.addWidget(self.log)

        self.results_box = self._build_results()
        self.results_box.setVisible(False)
        layout.addWidget(self.results_box)

        layout.addStretch()

    # ── Step 1: B0 ────────────────────────────────────────────────────
    def _pull_from_state(self):
        """Adopt whatever the one-click page last produced."""
        s = self.app_state
        self.b0 = s.get("b0_data")
        self.affine = s.get("b0_affine")
        self.brain_mask = s.get("brain_mask")
        self.coil = s.get("coil")
        self.coil_maps = s.get("coil_maps")
        self.coil_names = list(s.get("coil_channel_names") or [])

        have_b0 = self.b0 is not None and self.affine is not None
        have_coil = self.coil_maps is not None and len(self.coil_names) > 0

        # One grey line when there is data, one centred hint when there is not.
        # Which subject and which coil are worth a glance; how the numbers got
        # here is not something the operator acts on.
        self.empty_label.setVisible(not have_b0)
        self.src_label.setVisible(have_b0)
        if not have_b0:
            self.basis = None
        else:
            if self.brain_mask is None:
                self.brain_mask = self.b0 != 0
            self.basis = build_sh_basis(self.affine, self.b0.shape[:3])
            bits = [f"数据：{s.get('subject') or '—'}",
                    "B0 " + "×".join(str(v) for v in self.b0.shape[:3])]
            if have_coil and self.coil:
                bits.append(self.coil.name)
            self.src_label.setText(" · ".join(bits))

        # Joint mode needs coil maps; without them only the SH-only path runs.
        self.cb_joint.setEnabled(have_coil)
        if not have_coil:
            self.cb_joint.setChecked(False)
            self.cb_joint.setToolTip("请先在「一键匀场」页面加载数据并完成计算")
        else:
            self.cb_joint.setToolTip("")
        self.coil_label.setText(
            f"{self._coil_name()} · ±{self._coil_imax():g} A" if have_coil else "")
        if hasattr(self, "curr_section"):
            self.curr_section.setText(
                f"{self.coil.name} 电流 — 传到开发板" if (have_coil and self.coil)
                else "线圈电流 — 传到开发板")
        self.btn_run.setEnabled(have_b0)
        self._refresh_roi_note()
        self._on_mode_changed(self.cb_joint.isChecked())

    def _refresh_roi_note(self):
        """Say up front which region the objective will weight.

        The checkbox promises a target region; whether one exists depends on
        what the one-click run registered. Saying so before the solve is what
        stops the promise from being silently broken.
        """
        import numpy as _np

        if not hasattr(self, "roi_note"):
            return
        roi = self.app_state.get("roi_mask")
        if roi is None or self.brain_mask is None:
            self.chk_roi.setEnabled(False)
            self.lam_rest.setEnabled(False)
            self.roi_note.setText(
                "一键匀场未留下目标区 ROI —— 本页将按全脑等权求解。"
                "若要优先保证前额叶，请在一键匀场页选择前额叶 ROI 后重跑。")
            return

        roi = _np.asarray(roi, dtype=bool)
        brain = _np.asarray(self.brain_mask, dtype=bool)
        if roi.shape != brain.shape or not (roi & brain).any():
            self.chk_roi.setEnabled(False)
            self.lam_rest.setEnabled(False)
            self.roi_note.setText("目标区 ROI 与当前 B0 网格不匹配 —— 将按全脑等权求解。")
            return

        self.chk_roi.setEnabled(True)
        self.lam_rest.setEnabled(self.chk_roi.isChecked())
        n = int((roi & brain).sum())
        frac = 100.0 * n / max(int(brain.sum()), 1)
        name = self.app_state.get("roi_label") or "目标区"
        self.roi_note.setText(
            f"目标区：{name}，{n:,} 体素，占全脑 {frac:.1f}%。"
            "不加权时线圈的自由度会被其余脑区占去。")

    def showEvent(self, event):
        super().showEvent(event)
        self._pull_from_state()

    def _build_mode(self):
        box = QGroupBox("优化模式")
        v = QVBoxLayout(box)
        r = QHBoxLayout()
        self.cb_joint = QCheckBox("联合线圈优化")
        self.cb_joint.setChecked(True)
        self.cb_joint.toggled.connect(self._on_mode_changed)
        r.addWidget(self.cb_joint)
        # The coil is whatever the one-click page used; picking a different one
        # here would mean optimising against field maps the B0 was not paired
        # with. Read-only on purpose.
        self.coil_label = QLabel("—")
        self.coil_label.setObjectName("hint")
        r.addWidget(self.coil_label)
        r.addStretch()
        v.addLayout(r)
        return box

    def _coil_name(self) -> str:
        n = getattr(self.coil, "name", "") if self.coil else ""
        return n or "局部线圈"

    def _joint_note(self) -> str:
        return (f"系统 8 项球谐 + 「{self._coil_name()}」电流联合求解，"
                f"线圈电流全量程（±{self._coil_imax():g} A）。")

    def _coil_imax(self) -> float:
        """Current limit from the selected coil's own config.

        Not a spin box on this page: the limit is a property of the hardware,
        and letting it be retyped here would let it disagree with the value the
        one-click page enforced for the same coil.
        """
        try:
            v = float(getattr(self.coil, "current_limit_a", 0.0) or 0.0)
        except (TypeError, ValueError):
            v = 0.0
        return v if v > 0 else 3.0

    def _on_mode_changed(self, joint: bool):
        """Show or hide everything that only means something with a coil."""
        self.btn_run.setText("联合优化" if joint else "计算系统匀场")
        if hasattr(self, "curr_box"):
            self.curr_box.setVisible(joint)

    def _build_step2(self):
        box = QGroupBox("系统匀场状态")
        v = QVBoxLayout(box)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(4)

        self.sh_boxes = {}
        self.bound_boxes = {}

        hdr = QLabel("")
        grid.addWidget(hdr, 0, 0)
        for col, (name, label) in enumerate(TERMS, start=1):
            cap = QLabel(f"{label}<br><span style='font-size:10px'>{name} · "
                         f"{UNIT[ORDER[name]]}</span>")
            cap.setAlignment(Qt.AlignCenter)
            cap.setObjectName("hint")
            grid.addWidget(cap, 0, col)

        lbl_val = QLabel("当前值")
        lbl_val.setObjectName("sectionTitle")
        grid.addWidget(lbl_val, 1, 0)
        for col, (name, _) in enumerate(TERMS, start=1):
            grid.addWidget(self._make_sh_box(name), 1, col)

        lbl_bnd = QLabel("允许变化 ±")
        lbl_bnd.setObjectName("sectionTitle")
        grid.addWidget(lbl_bnd, 2, 0)
        for col, (name, _) in enumerate(TERMS, start=1):
            grid.addWidget(self._make_bound_box(name), 2, col)

        grid.setColumnStretch(len(TERMS) + 1, 1)
        v.addLayout(grid)
        v.addSpacing(6)

        # One 「导入」 menu instead of a radio pair plus two buttons: the radios
        # only ever reported which button had last been pressed.
        r = QHBoxLayout()
        btn_import = QToolButton()
        btn_import.setText("导入")
        btn_import.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(btn_import)
        menu.addAction("从扫描仪读取", lambda: self._import_shim_state(True))
        menu.addAction("从文件导入", lambda: self._import_shim_state(False))
        btn_import.setMenu(menu)
        self._import_menu = menu          # keep it alive
        r.addWidget(btn_import)

        btn_save = QPushButton("保存为默认")
        btn_save.clicked.connect(self._save_shim_state)
        r.addWidget(btn_save)

        btn_zero = QPushButton("全部归零")
        btn_zero.clicked.connect(self._zero_shim_state)
        r.addWidget(btn_zero)

        btn_bounds = QPushButton("上限配置")
        btn_bounds.setObjectName("ghostBtn")
        btn_bounds.setToolTip(
            "在 config/ 下生成 shim_delta_bounds.json，改完重启生效")
        btn_bounds.clicked.connect(self._write_bounds_file)
        r.addWidget(btn_bounds)

        self.shim_save_status = QLabel("")
        r.addWidget(self.shim_save_status)
        r.addStretch()
        v.addLayout(r)

        self.state_note = QLabel("")
        self.state_note.setObjectName("hint")
        self.state_note.setWordWrap(True)
        v.addWidget(self.state_note)

        return box

    def _zero_shim_state(self):
        st = SystemShimState.default()
        st.source = "manual"
        self._apply_state_to_ui(st)

    def _make_bound_box(self, name):
        box = QDoubleSpinBox()
        box.setRange(0.0, 2000.0)
        box.setDecimals(1)
        box.setSingleStep(10.0)
        box.setValue(float(self.shim_state.bounds.get(name, 0.0)))
        box.setFixedWidth(88)
        box.setToolTip(f"{LABEL[name]} 允许偏离当前值的幅度；0 = 锁定")
        self.bound_boxes[name] = box
        return box

    def _dim_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _make_sh_box(self, name):
        box = QDoubleSpinBox()
        box.setRange(-2000.0, 2000.0)
        box.setDecimals(1)
        box.setSingleStep(1.0)
        box.setValue(float(self.shim_values.get(name, 0.0)))
        box.setFixedWidth(90)
        self.sh_boxes[name] = box
        return box

    def _build_step5(self):
        box = QGroupBox("优化参数")
        outer = QVBoxLayout(box)
        outer.setSpacing(8)
        row = QHBoxLayout()
        row.setSpacing(36)

        # No temporal-lobe protection term: the objective is whole-brain RMSE,
        # with no region held back.
        #
        # λ_SH defaults to 0, which makes the harmonics-only solve exactly the
        # plain least-squares problem min ||A w + b0||^2 over the mask. A ridge
        # here has no conditioning job to do -- eight well-separated harmonics
        # over 200k voxels -- it only biases the coefficients toward zero, and
        # the coefficients are the deliverable.
        self.lam_sh = self._make_lam(0.0, 1.0, 0.0, 0.01, decimals=3)
        self.lam_pfc = self._make_lam(0.0, 2.0, 0.5, 0.1)
        self.lam_rest = self._make_lam(0.01, 2.0, 1.0, 0.1)

        for lbl, box_w, hint in [
            ("λ_SH", self.lam_sh, "SH 正则化 (0 = 纯最小二乘)"),
            ("λ_线圈", self.lam_pfc, "线圈电流正则化 (重，优先用 SH)"),
            ("λ_其余", self.lam_rest, "目标区以外的权重 (1.0 = 两区等权)"),
        ]:
            cell = QVBoxLayout()
            cell.setSpacing(3)
            top = QHBoxLayout()
            top.setSpacing(8)
            name = QLabel(lbl)
            name.setStyleSheet("font-weight: 600;")
            top.addWidget(name)
            top.addWidget(box_w)
            top.addStretch()
            cell.addLayout(top)
            h = QLabel(hint)
            h.setObjectName("hint")
            cell.addWidget(h)
            row.addLayout(cell)
        row.addStretch()
        outer.addLayout(row)

        # On by default. A surface array is built for one region, and counting
        # every voxel equally hands its degrees of freedom to the other 94% of
        # the brain -- measured on S28, adding the 12ch array under whole-brain
        # weighting made the PFC worse than harmonics alone (40.5 -> 43.8 Hz)
        # while the brain improved. Equal weight per region costs the brain
        # ~1.3 Hz and returns the PFC to 37.2.
        self.chk_roi = QCheckBox("优先保证目标区（前额叶）")
        self.chk_roi.setChecked(True)
        self.chk_roi.setToolTip(
            "按区域等权，而不是按体素等权。前额叶只占全脑约 6% 的体素，"
            "不加权时线圈的自由度会被其余脑区占去，前额叶反而比只用 SH 更差。\n"
            "目标区来自一键匀场配准好的 ROI；没有可用 ROI 时自动退回全脑。")
        self.chk_roi.toggled.connect(
            lambda on: self.lam_rest.setEnabled(on))
        outer.addWidget(self.chk_roi)

        self.roi_note = QLabel("")
        self.roi_note.setObjectName("hint")
        self.roi_note.setWordWrap(True)
        outer.addWidget(self.roi_note)
        return box

    def _make_lam(self, lo, hi, val, step, decimals=2):
        b = QDoubleSpinBox()
        b.setRange(lo, hi)
        b.setValue(val)
        b.setSingleStep(step)
        b.setDecimals(decimals)
        b.setFixedWidth(90)
        return b

    # ── Results ───────────────────────────────────────────────────────
    def _build_results(self):
        box = QGroupBox("结果")
        v = QVBoxLayout(box)

        # metric cards
        self.cards = {}
        row = QHBoxLayout()
        for key in ["全脑 RMSE", "中心频率残余"]:
            row.addWidget(self._make_card(key))
        v.addLayout(row)

        # SH table
        v.addWidget(self._section("系统匀场 — 输回扫描仪的值"))
        self.sh_table = QTableWidget(len(SH12), 3)
        self.sh_table.setHorizontalHeaderLabels(["项", "当前值", "输入扫描仪的值"])
        self.sh_table.verticalHeader().setVisible(False)
        self.sh_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sh_table.setMaximumHeight(240)
        v.addWidget(self.sh_table)

        self.limit_note = QLabel("")
        self.limit_note.setWordWrap(True)
        v.addWidget(self.limit_note)

        self.cf_note = QLabel("")
        self.cf_note.setWordWrap(True)
        self.cf_note.setVisible(False)
        v.addWidget(self.cf_note)

        # PFC currents -- hidden entirely in SH-only mode.
        self.curr_box = QWidget()
        cv = QVBoxLayout(self.curr_box)
        cv.setContentsMargins(0, 0, 0, 0)
        self.curr_section = self._section("线圈电流 — 传到开发板")
        cv.addWidget(self.curr_section)
        self.fig_curr = Figure(figsize=(9, 2.6), facecolor=self.p["fig_bg"])
        self.canvas_curr = FigureCanvas(self.fig_curr)
        self.canvas_curr.setMinimumHeight(200)
        cv.addWidget(self.canvas_curr)
        self.curr_caption = QLabel("")
        self.curr_caption.setObjectName("hint")
        cv.addWidget(self.curr_caption)
        v.addWidget(self.curr_box)

        # before/after viewer
        v.addWidget(self._section("B0 Before / After"))
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Sagittal:"))
        self.sl_x = QSlider(Qt.Horizontal)
        self.sl_x.valueChanged.connect(self._draw_viewer)
        ctrl.addWidget(self.sl_x)
        ctrl.addWidget(QLabel("Coronal:"))
        self.sl_y = QSlider(Qt.Horizontal)
        self.sl_y.valueChanged.connect(self._draw_viewer)
        ctrl.addWidget(self.sl_y)
        ctrl.addWidget(QLabel("Axial:"))
        self.sl_z = QSlider(Qt.Horizontal)
        self.sl_z.valueChanged.connect(self._draw_viewer)
        ctrl.addWidget(self.sl_z)
        v.addLayout(ctrl)

        self.fig_ba = Figure(figsize=(10, 6), facecolor=self.p["fig_bg"])
        self.canvas_ba = FigureCanvas(self.fig_ba)
        self.canvas_ba.setMinimumHeight(400)
        v.addWidget(self.canvas_ba)

        # export
        exp = QHBoxLayout()
        for label, fmt in [("线圈电流 (64ch txt)", "txt"),
                           ("完整结果 (JSON)", "json"),
                           ("残差场图 (NIfTI)", "nii")]:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, f=fmt: self._export(f))
            exp.addWidget(b)
        exp.addStretch()
        v.addLayout(exp)
        return box

    def _section(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("sectionTitle")
        return lbl

    def _make_card(self, label):
        frame = QFrame()
        frame.setObjectName("card")
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(14, 10, 14, 12)
        vl.setSpacing(3)
        lbl = QLabel(label)
        lbl.setObjectName("hint")
        vl.addWidget(lbl)
        val = QLabel("—")
        val.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {self.p['text']};")
        vl.addWidget(val)
        self.cards[label] = val
        return frame

    # ══════════════════════════════════════════════════════════════════
    #  Loading
    # ══════════════════════════════════════════════════════════════════
    # ══════════════════════════════════════════════════════════════════
    #  Run
    # ══════════════════════════════════════════════════════════════════
    def _run(self):
        if self.b0 is None:
            QMessageBox.warning(
                self, "没有数据",
                "请先在「一键匀场」页拖入 DICOM 并完成计算，本页再接着算。")
            return
        joint = self.cb_joint.isChecked() and self.cb_joint.isEnabled()

        for n, box in self.sh_boxes.items():
            self.shim_values[n] = box.value()

        # Validate before putting the UI into its running state. Bailing out
        # afterwards left the run button disabled and the progress bar spinning
        # with nothing behind them, and the page needed a restart to recover.
        st = self._collect_state()
        sh_terms = [n for n, _ in TERMS if not st.is_frozen(n)]
        if not sh_terms:
            QMessageBox.warning(
                self, "无可调项",
                "八个系统匀场项的允许变化都填成了 0，没有任何一项可以调整。\n"
                "请至少给一项一个非零的 ±Δ，或直接用一键匀场页只优化线圈。")
            return

        self.log.clear()
        self.btn_run.setEnabled(False)
        self.progress.setVisible(True)

        self.log.append(f"[Shim 状态] {st.summary()}")
        frozen = [LABEL[n] for n, _ in TERMS if st.is_frozen(n)]
        if frozen:
            self.log.append("[Shim 状态] 固定不动：" + "、".join(frozen))

        roi = self._objective_roi()
        self.worker = SystemShimWorker(
            self.b0, self.basis, self.brain_mask,
            self.affine, self.b0.shape[:3],
            self.coil_maps if joint else None,
            self.coil_names if joint else [],
            self._coil_imax(),
            self.lam_sh.value(), self.lam_pfc.value(), parent=self,
            sh_terms=sh_terms, sh_bounds=st.delta_bounds(sh_terms),
            scanner_baseline=[st.baseline.get(n, 0.0) for n in sh_terms],
            scanner_limits=[st.abs_limits.get(n, 0.0) for n in sh_terms],
            roi_mask=roi, lambda_rest=self.lam_rest.value())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _objective_roi(self):
        """The target region for the objective, or None for whole-brain.

        Reuses whatever the one-click run already registered rather than
        registering again -- the mask is on the same grid as this page's B0,
        because both came from that run. Returning None is a normal outcome
        (whole-brain ROI, or a run that never registered one), not a failure,
        so it says which happened in the log instead of blocking the solve.
        """
        import numpy as _np

        if not self.chk_roi.isChecked():
            self.log.append("[目标区] 未启用——按体素等权，全脑目标")
            return None

        roi = self.app_state.get("roi_mask")
        if roi is None:
            self.log.append(
                "[目标区] 一键匀场没有留下 ROI（可能选的是「全脑」）——退回全脑等权")
            return None

        roi = _np.asarray(roi, dtype=bool)
        if roi.shape != tuple(self.b0.shape[:3]):
            self.log.append(
                f"[目标区] ROI 网格 {roi.shape} 与 B0 {tuple(self.b0.shape[:3])} "
                "不一致——退回全脑等权")
            return None

        brain = _np.asarray(self.brain_mask, dtype=bool)
        n = int((roi & brain).sum())
        if n == 0:
            self.log.append("[目标区] ROI 与脑掩膜无交集——退回全脑等权")
            return None

        frac = 100.0 * n / max(int(brain.sum()), 1)
        name = self.app_state.get("roi_label") or "目标区"
        self.log.append(
            f"[目标区] {name}：{n:,} 体素（占全脑 {frac:.1f}%），"
            f"λ_其余={self.lam_rest.value():g}（1.0 = 两区等权）")
        return roi

    @Slot(str)
    def _on_progress(self, msg):
        self.log.append(msg)

    @Slot(str)
    def _on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.log.append(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Optimization failed", msg)

    @Slot(dict)
    def _stage_stats(self, residual):
        """Same statistics, same key names, as the one-click stage.

        `evaluate_masks` is the one implementation; calling it keeps the two
        stages comparable and stops the schemas drifting apart.
        """
        import numpy as _np

        from core.shim_optimize import evaluate_masks

        b0 = self.app_state.get("b0_vol3d")
        if b0 is None or residual is None:
            return {}
        roi = self.app_state.get("roi_mask")
        if roi is None:
            roi = self.app_state.get("brain_mask")
        if roi is None:
            roi = _np.asarray(b0) != 0
        try:
            return evaluate_masks(_np.asarray(b0), _np.asarray(residual),
                                  _np.asarray(roi, dtype=bool))
        except Exception:
            # A statistics failure must not lose the shim result itself.
            return {}

    def _on_done(self, result):
        self.btn_run.setEnabled(True)
        self.progress.setVisible(False)
        self.result = result

        # Share the currents with the Teensy page. The Results page renders the
        # one-click flow's before/after volume, which does NOT match these
        # currents — drop it rather than let the two be shown side by side.
        self.app_state["optimal_currents"] = result["i_pfc"]
        self.app_state["ch_names_active"] = result["pfc_names"]
        self.app_state["hybrid_result"] = result
        self.app_state["currents_source"] = "hybrid"
        # The system shim is a further stage, not a replacement. Clearing the
        # one-click residual here is what made the results page forget it and
        # fall back to "please run the one-click page first" -- after the user
        # had just run both.
        self.app_state["b0_hybrid_vol"] = result.get("residual")
        # The results page reads one fixed schema -- roi_rmse_before and its
        # siblings. Inventing a second one here is what raised KeyError the
        # moment the stage selector picked this stage.
        self.app_state["hybrid_stats"] = self._stage_stats(
            result.get("residual"))

        joint = len(result.get("pfc_names") or []) > 0
        self.curr_box.setVisible(joint)

        self._fill_cards(result["stats"])
        self._fill_sh_table(result["delta_sh"], result.get("sh_terms"),
                            result.get("scanner_delta"),
                            result.get("local_sh_equiv"),
                            (result.get("stats") or {}).get("centre_frequency_hz"))
        self._report_centre_frequency(result)
        if joint:
            self._draw_currents()

        nx, ny, nz = self.b0.shape
        for sl, n in [(self.sl_x, nx), (self.sl_y, ny), (self.sl_z, nz)]:
            sl.blockSignals(True)
            sl.setRange(0, n - 1)
            sl.setValue(n // 2)
            sl.blockSignals(False)

        self.results_box.setVisible(True)
        self._draw_viewer()
        self.log.append("优化完成 — 系统匀场调整值与线圈电流见下方")

    # ══════════════════════════════════════════════════════════════════
    #  Result rendering
    # ══════════════════════════════════════════════════════════════════
    def _fill_cards(self, stats):
        """Headline figure is RMSE, because RMSE is what the solve minimises.

        The objective is plain least squares over the mask, constant component
        included -- nothing re-locks the centre frequency between typing the
        coefficients in and acquiring. Reporting demeaned spread instead would
        quietly subtract an offset that will still be there on the scanner, and
        so would claim a result nobody measures. std is in the tooltip and both
        are written to result.json.
        """
        # Only two figures, and they measure different things. With the
        # temporal-protection region gone, "hybrid" is the whole brain, so a
        # second spread card would have repeated the first one exactly.
        key = "全脑 RMSE"
        pre, post = stats.get("brain_rmse_pre"), stats.get("brain_rmse_post")
        if pre is not None and post is not None:
            drop = (1 - post / max(pre, 0.1)) * 100
            clr = self.p["success"] if post < pre else self.p["danger"]
            self.cards[key].setText(
                f"{pre:.1f} → {post:.1f} Hz  (↓{drop:.0f}%)" if post < pre
                else f"{pre:.1f} → {post:.1f} Hz  (↑{-drop:.0f}%)")
            sd_pre, sd_post = stats.get("brain_std_pre"), stats.get("brain_std_post")
            tip = ("脑内 RMSE，含常数项——这是本页最小化的量，"
                   "也是下发后实际会得到的场。")
            if sd_pre is not None and sd_post is not None:
                tip += f"\n去均值标准差 {sd_pre:.1f} → {sd_post:.1f} Hz。"
            # The target region is the reason the whole-brain number may be
            # slightly worse than it could be, so the two belong together.
            r_pre, r_post = stats.get("roi_std_pre"), stats.get("roi_std_post")
            if r_pre is not None and r_post is not None:
                tip += (f"\n目标区 std {r_pre:.1f} → {r_post:.1f} Hz"
                        f"（{stats.get('_roi_voxels', 0):,} 体素，"
                        f"λ_其余={stats.get('_roi_lambda_rest', 1.0):g}）。")
            self.cards[key].setToolTip(tip)
            self.cards[key].setStyleSheet(
                f"color: {clr}; font-size: 15px; font-weight: 600;")
        else:
            self.cards[key].setText("—")
            self.cards[key].setStyleSheet(
                f"color: {self.p['text_3']}; font-size: 15px; font-weight: 600;")

    def _fill_sh_table(self, delta_sh, sh_terms=None, scanner_delta=None,
                       local_sh_equiv=None, centre_frequency_hz=None):
        """What is set now, and what to set instead.

        Two columns because that is the whole job at the console: read a row,
        type the right-hand number in. The intermediate quantities -- the model
        delta, the part the coils already supply, the remaining headroom -- are
        how the number was reached, not what anyone acts on, so they live in
        the exported JSON and the log rather than on screen.

        The value shown is `当前值 + scanner_delta`, i.e. it already has the
        coils' contribution removed; entering the model total instead would
        apply the overlap twice.

        The last row is the centre frequency, and it is the one row that is a
        *change* rather than a setting -- the console's f0 is an absolute
        frequency in the hundreds of MHz and this page never reads it, so there
        is no "current value" to add to.
        """
        terms = list(sh_terms) if sh_terms else list(SH2)
        sd = list(scanner_delta) if scanner_delta is not None else list(delta_sh)
        eq = list(local_sh_equiv) if local_sh_equiv is not None else [0.0] * len(terms)

        st = self.shim_state
        has_f0 = centre_frequency_hz is not None
        self.sh_table.setRowCount(len(terms) + (1 if has_f0 else 0))
        breaches = []
        for i, n in enumerate(terms):
            cur = float(self.shim_values.get(n, 0.0))
            target = cur + float(sd[i])
            lim = abs(float(st.abs_limits.get(n, 0.0)))
            over = lim > 0 and abs(target) > lim + 1e-6
            if over:
                breaches.append((n, target, lim))

            cells = [f"{LABEL[n]} ({n}, {UNIT[ORDER[n]]})",
                     f"{cur:+.1f}", f"{target:+.1f}"]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 2:
                    # The value carries its own verdict: green means the
                    # console will take it, red means it will not. That is the
                    # one thing to check per row, so it belongs on the number
                    # rather than in a separate column of headroom figures.
                    item.setForeground(QColor(
                        self.p["danger"] if over else self.p["success"]))
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                    item.setToolTip(
                        (f"超出主机限幅 ±{lim:g}\n" if over else
                         f"在主机限幅 ±{lim:g} 内\n" if lim else "")
                        + f"模型 Δ {float(delta_sh[i]):+.2f}　"
                        f"线圈等效 {float(eq[i]):+.2f}　"
                        f"扫描仪 Δ {float(sd[i]):+.2f}")
                self.sh_table.setItem(i, col, item)

        if has_f0:
            # Sign. b0 is deltaB0 in Hz, i.e. local resonance minus the transmit
            # frequency *the field map was acquired at*, so a residual mean of
            # +20 Hz means the tissue is precessing 20 Hz above that frequency
            # and f0 has to go up by the same amount to meet it. Applying it
            # subtracts the constant from the residual, which is exactly the
            # difference between RMSE and std -- so the RMSE card would fall to
            # the std quoted in its tooltip.
            #
            # Which f0 that offset attaches to is the whole difficulty, and it
            # is why the absolute value is shown whenever the DICOM gave one.
            # The console's displayed f0 is not interchangeable with the field
            # map's: a prescan anywhere between the two re-locks it to the mean
            # resonance of the *unshimmed* field, and adding the offset to that
            # overshoots by the pre-shim mean -- tens of Hz, often larger than
            # the offset itself. An operator reading a bare increment has no
            # way to tell which baseline is in front of them.
            cf = float(centre_frequency_hz)
            row = len(terms)
            f0_hz = self.app_state.get("imaging_frequency_hz")
            try:
                f0_hz = float(f0_hz) if f0_hz else None
            except (TypeError, ValueError):
                f0_hz = None

            if f0_hz:
                target = f0_hz + cf
                cells = ["中心频率 (f0, MHz)",
                         f"{f0_hz / 1e6:.6f}", f"{target / 1e6:.6f}"]
                tip = (f"这一行是绝对频率，可直接填入控制台。\n\n"
                       f"场图采集频率 {f0_hz / 1e6:.6f} MHz "
                       f"({f0_hz:,.0f} Hz)，来自 DICOM (0018,0084)。\n"
                       f"匀场后脑内平均共振频率比它高 {cf:+.1f} Hz，"
                       f"故目标 f0 = {target / 1e6:.6f} MHz "
                       f"({target:,.0f} Hz)。\n\n"
                       "⚠ 增量必须加在上面这个采集频率上，不是控制台当前显示的 f0。"
                       "若两者之间跑过频率校准，控制台的 f0 已被重锁到未匀场场的"
                       "平均共振频率，再加一次增量会多加匀场前的偏置。\n\n"
                       "调整后全脑 RMSE 会降到上方 RMSE 卡片提示中的去均值标准差。"
                       "若 f0 不可调，此行忽略即可，上方 RMSE 已包含这一项。")
            else:
                cells = ["中心频率 (f0, Hz)", "—", f"{cf:+.1f}  (增量)"]
                tip = ("这一行是增量，不是绝对值——本次没有从 DICOM 读到"
                       "采集发射频率 (0018,0084)。\n\n"
                       f"匀场后脑内平均共振频率比采集场图时的 f0 高 {cf:+.1f} Hz，"
                       f"因此把那个 f0 调高 {cf:+.1f} Hz 即可对准；"
                       "调整后全脑 RMSE 会降到上方 RMSE 卡片提示中的去均值标准差。\n\n"
                       "⚠ 基准是采集场图时的 f0，不是控制台当前显示的 f0。"
                       "若两者之间跑过频率校准，两个数已经不同。\n\n"
                       "符号沿用 deltaB0 = 局部共振频率 − 发射频率 的约定。"
                       "若 f0 不可调，此行忽略即可，上方 RMSE 已包含这一项。")

            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if col == 1:
                    item.setForeground(QColor(self.p["text_3"]))
                if col == 2:
                    item.setForeground(QColor(self.p["warning"]))
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                    item.setToolTip(tip)
                self.sh_table.setItem(row, col, item)

        if breaches:
            detail = "、".join(f"{LABEL[n]}（{v:+.0f} > {lim:g}）"
                              for n, v, lim in breaches)
            self.limit_note.setText(f"⚠ 以下项超出主机限幅，扫描仪会拒收：{detail}")
            self._status(self.limit_note, "err")
        else:
            lim = max(abs(float(v)) for v in st.abs_limits.values()) if st.abs_limits else 0
            self.limit_note.setText(f"✓ 所有调整值在主机限幅内 (±{lim:g})")
            self._status(self.limit_note, "ok")

        from core.shim_state import bounds_bind_everything

        if bounds_bind_everything(delta_sh, st.bounds, terms):
            self.log.append(
                "[Shim 状态] ⚠ 八项全部顶到上限——这通常说明 "
                f"{st.bounds_source} 里的单位与场图基组不一致"
                "（一阶 uT/m，二阶 uT/m²，实测 Δ 量级为几十到几百）。")

    def _report_centre_frequency(self, result):
        """State the uniform offset the harmonics could not remove.

        Nothing takes this out. The eight coefficients are typed into the
        console and the next acquisition runs with them, with no prescan in
        between to re-lock f0, so the solve minimises RMSE with the constant
        included and whatever is left here is field that will be there. It is
        already inside the RMSE card above -- this is the same number broken
        out, not a further correction to apply.

        A large value therefore means two things at once: the harmonics ran out
        of authority over the constant component, and the field map was
        acquired well off resonance, at which point the phase unwrapping behind
        it may not have been reliable either.
        """
        self.cf_note.setVisible(True)
        offset = float((result.get("stats") or {}).get("centre_frequency_hz", 0.0))
        a = abs(offset)

        if "中心频率残余" in self.cards:
            colour = (self.p["text_3"] if a < 20 else
                      self.p["warning"] if a <= 50 else self.p["danger"])
            self.cards["中心频率残余"].setText(f"{offset:+.1f} Hz")
            self.cards["中心频率残余"].setToolTip(
                "匀场后残余场的均值，已计入上方 RMSE。"
                "八项球谐里没有零阶项，消不掉它；"
                "若控制台的 f0 可调，见表格最后一行。")
            self.cards["中心频率残余"].setStyleSheet(
                f"color: {colour}; font-size: 15px; font-weight: 600;")

        # Naming the baseline in the line itself, not only in the table's
        # tooltip: this is the sentence the operator reads at the console, and
        # an offset without its baseline is the one way this number gets
        # applied wrongly.
        f0_hz = self.app_state.get("imaging_frequency_hz")
        try:
            f0_hz = float(f0_hz) if f0_hz else None
        except (TypeError, ValueError):
            f0_hz = None
        tail = ("已计入上方 RMSE；若 f0 可调，按表格最后一行调整即可消除"
                + (f"（目标 f0 = {(f0_hz + offset) / 1e6:.6f} MHz，"
                   f"基准为场图采集频率 {f0_hz / 1e6:.6f} MHz）。" if f0_hz
                   else "，基准是采集场图时的 f0，不是控制台当前显示的 f0。"))
        if a < 20.0:
            self.cf_note.setText(f"中心频率残余 {offset:+.1f} Hz，{tail}")
            self._status(self.cf_note, "hint")
        elif a <= 50.0:
            self.cf_note.setText(
                f"中心频率残余 {offset:+.1f} Hz，{tail}"
                "该偏置偏大，建议同时确认场图采集时未明显偏离共振。")
            self._status(self.cf_note, "warn")
        else:
            self.cf_note.setText(
                f"中心频率残余 {offset:+.1f} Hz，{tail}"
                "但该偏置异常大——这种失谐下相位解缠本身可能不可靠，"
                "建议先重新采集场图，而不是直接照此调整 f0。")
            self._status(self.cf_note, "err")

    def _draw_currents(self):
        i_pfc = self.result["i_pfc"]
        names = self.result["pfc_names"]
        imax = self.result["pfc_imax"]

        p = self.p
        self.fig_curr.clear()
        self.fig_curr.set_facecolor(p["fig_bg"])
        ax = self.fig_curr.add_subplot(111)
        style_axes(ax, p)
        colors = [p["series_pos"] if v >= 0 else p["series_neg"] for v in i_pfc]
        ax.bar(range(len(i_pfc)), i_pfc, color=colors, width=0.62)
        ax.axhline(0, color=p["border"], linewidth=0.8)
        for y in (imax, -imax):
            ax.axhline(y, color=p["danger"], linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8,
                           color=p["text_2"])
        ax.set_ylabel("Current (A)", color=p["text_2"], fontsize=9)
        ax.set_ylim(-imax - 0.5, imax + 0.5)
        self.fig_curr.tight_layout()
        self.canvas_curr.draw()

        rms_i = float(np.sqrt(np.mean(i_pfc ** 2)))
        max_i = float(np.max(np.abs(i_pfc)))
        n_bound = int(np.sum(np.abs(np.abs(i_pfc) - imax) < 1e-3))
        warn = (f"  |  ⚠ {n_bound}/{len(i_pfc)} 个通道顶到 ±{imax:g} A 上界"
                if n_bound else "")
        self.curr_caption.setText(
            f"RMS: {rms_i:.3f} A  |  Max |I|: {max_i:.3f} A{warn}")

    def _draw_viewer(self):
        if self.result is None or self.b0 is None:
            return
        residual = self.result["residual"]
        bm = self.brain_mask
        sx, sy, sz = self.sl_x.value(), self.sl_y.value(), self.sl_z.value()

        def cut(vol, axis, idx):
            if axis == 0:
                sl, mk = vol[idx, :, :].T, bm[idx, :, :].T
            elif axis == 1:
                sl, mk = vol[:, idx, :].T, bm[:, idx, :].T
            else:
                sl, mk = vol[:, :, idx].T, bm[:, :, idx].T
            out = sl.astype(float).copy()
            out[~mk] = np.nan
            return out, mk

        p = self.p
        self.fig_ba.clear()
        self.fig_ba.set_facecolor(p["fig_bg"])
        axes = self.fig_ba.subplots(2, 3)
        self.fig_ba.subplots_adjust(hspace=0.2, wspace=0.05,
                                    left=0.02, right=0.9, top=0.9, bottom=0.02)
        im = None
        for r, (vol, label) in enumerate([(self.b0, "Before"),
                                          (residual, "After")]):
            for c, (vn, idx) in enumerate([("Sag", sx), ("Cor", sy), ("Axi", sz)]):
                ax = axes[r, c]
                sl, mk = cut(vol, c, idx)
                im = ax.imshow(sl, cmap="jet", vmin=-CLIM, vmax=CLIM,
                               origin="lower", interpolation="nearest")
                ax.set_facecolor(p["fig_bg"])
                ax.axis("off")
                vals = sl[np.isfinite(sl)]
                if vals.size:
                    s, rm = float(np.std(vals)), float(np.sqrt(np.mean(vals ** 2)))
                    ax.set_title(f"{label} · {vn} {idx}\nstd {s:.1f} | RMSE {rm:.1f} Hz",
                                 color=p["text_2"], fontsize=8)
                else:
                    ax.set_title(f"{label} · {vn} {idx}", color=p["text_2"], fontsize=8)
        if im is not None:
            cb = self.fig_ba.colorbar(im, ax=axes[:, :], shrink=0.8, pad=0.01, aspect=30)
            cb.ax.tick_params(colors=p["text_2"], labelsize=8)
            cb.outline.set_edgecolor(p["border"])
            cb.set_label("Hz", color=p["text_2"], fontsize=9)
        self.canvas_ba.draw()

    # ══════════════════════════════════════════════════════════════════
    #  Export
    # ══════════════════════════════════════════════════════════════════
    def _export(self, fmt):
        if self.result is None:
            return
        r = self.result
        i_pfc, delta_sh = r["i_pfc"], r["delta_sh"]

        if fmt == "txt":
            path, _ = QFileDialog.getSaveFileName(
                self, "导出线圈电流", "coil_currents.txt", "Text (*.txt)")
            if path:
                with open(path, "w") as f:
                    f.write(format_currents_64ch(i_pfc))

        elif fmt == "json":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export result", "hybrid_shim_result.json", "JSON (*.json)")
            if path:
                terms = r.get("sh_terms") or list(SH2)
                sd = r.get("scanner_delta")
                sd = list(sd) if sd is not None else list(delta_sh)
                eq = r.get("local_sh_equiv")
                eq = list(eq) if eq is not None else [0.0] * len(terms)
                _f0 = self.app_state.get("imaging_frequency_hz")
                try:
                    _f0 = float(_f0) if _f0 else None
                except (TypeError, ValueError):
                    _f0 = None
                data = {
                    "sh_terms": terms,
                    "sh_labels": {n: LABEL.get(n, n) for n in terms},
                    "sh_units": {n: UNIT[ORDER[n]] for n in terms},
                    # input / output / final, all three kept: a delta with no
                    # baseline cannot be checked, and a final with no delta
                    # cannot be undone.
                    "sh_baseline": {n: float(self.shim_values.get(n, 0.0))
                                    for n in terms},
                    "sh_model_delta": {n: float(delta_sh[i])
                                       for i, n in enumerate(terms)},
                    "sh_local_equivalent": {n: float(eq[i])
                                            for i, n in enumerate(terms)},
                    "sh_scanner_delta": {n: float(sd[i]) for i, n in enumerate(terms)},
                    "sh_final": {n: float(self.shim_values.get(n, 0.0) + sd[i])
                                 for i, n in enumerate(terms)},
                    # Sign follows deltaB0 = local resonance minus transmit
                    # frequency, so the delta is *added*. Which frequency it is
                    # added to is recorded alongside it rather than left to be
                    # inferred: a delta whose baseline is unstated gets applied
                    # to whatever the console happens to show, and a prescan in
                    # between makes that a different number.
                    "centre_frequency_delta_hz": float(
                        (r.get("stats") or {}).get("centre_frequency_hz", 0.0)),
                    "imaging_frequency_hz": _f0,
                    "centre_frequency_target_hz": (
                        _f0 + float((r.get("stats") or {})
                                    .get("centre_frequency_hz", 0.0))
                        if _f0 else None),
                    "system_shim_state": self.shim_state.to_dict(),
                    "pfc_currents": {r["pfc_names"][i]: float(i_pfc[i])
                                     for i in range(len(i_pfc))},
                    "stats": r["stats"],
                    "params": {"pfc_imax": self._coil_imax(),
                               "lam_sh": float(self.lam_sh.value()),
                               "lam_pfc": float(self.lam_pfc.value()),
                               "objective": "plain_least_squares",
                               "roi_weighted": bool(
                                   (r.get("stats") or {}).get("_roi_weighted")),
                               "lam_rest": float(self.lam_rest.value())},
                    "solver": {"success": r["success"], "nit": r["nit"]},
                }
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)

        elif fmt == "nii":
            path, _ = QFileDialog.getSaveFileName(
                self, "Export residual", "residual_hz.nii.gz", "NIfTI (*.nii.gz)")
            if path:
                nib.save(nib.Nifti1Image(r["residual"].astype(np.float32),
                                         self.affine), path)
