"""Product UI for the idealized, offline linear-encoding simulator.

Scientific definitions and limitations live in
``docs/LINEAR_ENCODING_SIMULATION.md``; this module keeps the working surface
focused on inputs, status and results.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
# Matplotlib's bundled DejaVu Sans has no CJK glyphs, so a Chinese axis label
# printed one warning per character and rendered as boxes. The figures speak
# English; the panel around them stays Chinese, where Qt uses a system font
# that does have the glyphs.
matplotlib.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 11,
    "axes.unicode_minus": False,
})
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget)

from core.coil_registry import list_coils
from core.linear_encoding_export import EXPORT_FILES, export_run
from core.linear_encoding_frames import (FRAME_LABELS, FRAME_TYPES, K_NOTE,
                                         NONLINEAR_NOTE, FrameCache,
                                         decimate, decimation_report,
                                         display_indices, frame_stats,
                                         frame_volume, plane_note,
                                         symmetric_limits)
from core.linear_encoding_model import UnitStatus
from core.linear_encoding_run import MODE_DIRECT, MODE_TARGET
from core.paths import resource
from core.spatial_grid import COVERAGE_THRESHOLD, RAS_AXES
from workers.linear_encoding_worker import LinearEncodingWorker

#: Compact UI badge. The complete limitations remain in the README and export.
BANNER = "离线模型 · 不连接硬件"
SAFETY_TOOLTIP = (
    "IDEALIZED OFFLINE SIMULATION\n"
    "未与扫描仪 ADC、梯度时序或功放同步；电感、电压、带宽及梯度耦合未验证。\n"
    "不能据此宣称 EPI 加速或图像改善。")

DIRECTION_CHOICES = ["RAS-X (R-L)", "RAS-Y (A-P)", "RAS-Z (I-S)", "自定义方向"]


class LinearEncodingWidget(QWidget):
    """Inputs, constraints and results for the offline encoding simulator."""

    request_page = Signal(int)

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = {}
        # Three separate stores, cleared together. `_result` is the
        # serialisable part that goes into JSON; `_runtime` holds the model,
        # the simulation and the full time series, which must never end up in
        # an export; `_frame_cache` holds at most a handful of volumes.
        self._result = None
        self._runtime = None
        self._frame_cache = FrameCache(limit=5)
        self._result_id = 0
        self._region = None
        self._worker = None
        self._export_dir = None
        self._frame_timer = QTimer(self)
        self._frame_timer.setSingleShot(True)
        self._frame_timer.setInterval(75)
        self._frame_timer.timeout.connect(self._draw_slices)

        root = QVBoxLayout(self)
        root.setSpacing(8)

        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("线性编码仿真")
        title.setObjectName("pageTitle")
        subtitle = QLabel("配置波形，查看线圈电流、空间场与相位响应")
        subtitle.setObjectName("pageSubtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)

        self.banner = QLabel(BANNER)
        self.banner.setObjectName("encodingBanner")
        self.banner.setAlignment(Qt.AlignCenter)
        self.banner.setToolTip(SAFETY_TOOLTIP)
        header.addWidget(self.banner)
        self.btn_help = QPushButton("使用说明")
        self.btn_help.setObjectName("ghostBtn")
        self.btn_help.clicked.connect(self._open_readme)
        header.addWidget(self.btn_help)
        root.addLayout(header)

        # The parameter groups are set once and read never; the figures are
        # what the page exists to show. So the settings get a bounded height
        # that scrolls, and every remaining pixel goes to the output tabs.
        setup_inner = QWidget()
        setup = QHBoxLayout(setup_inner)
        setup.setContentsMargins(0, 0, 0, 0)
        setup.setSpacing(10)
        setup.addWidget(self._build_inputs(), 1)
        setup.addWidget(self._build_constraints(), 1)

        self.setup_scroll = QScrollArea()
        self.setup_scroll.setWidgetResizable(True)
        self.setup_scroll.setWidget(setup_inner)
        self.setup_scroll.setFrameShape(QScrollArea.NoFrame)
        self.setup_scroll.setMaximumHeight(300)
        self.setup_scroll.setSizePolicy(QSizePolicy.Expanding,
                                        QSizePolicy.Maximum)
        root.addWidget(self.setup_scroll, 0)

        act_row = QHBoxLayout()
        act_row.addWidget(self._build_actions(), 1)
        self.btn_toggle_setup = QPushButton("收起设置")
        self.btn_toggle_setup.setObjectName("ghostBtn")
        self.btn_toggle_setup.setCheckable(True)
        self.btn_toggle_setup.toggled.connect(self._toggle_setup)
        act_row.addWidget(self.btn_toggle_setup, 0)
        root.addLayout(act_row)

        self.output_tabs = QTabWidget()
        self.output_tabs.addTab(self._build_results(), "结果")
        self.output_tabs.addTab(self._build_plots(), "可视化")
        self.output_tabs.setSizePolicy(QSizePolicy.Expanding,
                                       QSizePolicy.Expanding)
        root.addWidget(self.output_tabs, 1)

        self.btn_coverage.clicked.connect(self.on_coverage)
        self.btn_run.clicked.connect(self.on_run)
        self.btn_export.clicked.connect(self.on_export)

        self._on_direction_changed(0)
        self._on_region_changed(0)
        self._on_mode_changed(0)
        self._refresh_unit_source()
        self._update_ready()

    def _open_readme(self):
        path = resource("docs", "LINEAR_ENCODING_SIMULATION.md")
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(path)):
            self.status.setText("无法打开使用说明：%s" % path)

    # ── A. inputs ───────────────────────────────────────────────────
    def _build_inputs(self):
        box = QGroupBox("① 数据与区域")
        form = QFormLayout(box)

        self.coil_combo = QComboBox()
        for c in list_coils():
            self.coil_combo.addItem("%s (%d ch, ±%.1f A)"
                                    % (c.name, c.channel_count,
                                       c.current_limit_a), c.name)
        self.coil_combo.currentIndexChanged.connect(self._invalidate_region)
        form.addRow("线圈配置", self.coil_combo)

        # Where the unit assertion comes from depends on the coil. A shipped
        # coil carries it in its own config file, next to the calibration it
        # was made about. A coil the user imported carries nothing, so the box
        # starts unticked and they have to say so themselves -- confirming the
        # 12ch maps must not vouch for a folder dropped in later.
        self.unit_confirm = QCheckBox(
            "确认所选线圈场图按 Hz/A 解释，且与当前标定一致")
        self.unit_confirm.toggled.connect(self._update_ready)
        self.unit_source = QLabel("")
        self.unit_source.setObjectName("unitSource")
        urow = QVBoxLayout()
        urow.setContentsMargins(0, 0, 0, 0)
        urow.addWidget(self.unit_confirm)
        urow.addWidget(self.unit_source)
        uw = QWidget(); uw.setLayout(urow)
        form.addRow("场图单位", uw)
        self.coil_combo.currentIndexChanged.connect(self._refresh_unit_source)

        self.region_combo = QComboBox()
        self.region_combo.addItems(["球形 DSV（线圈自身网格）", "ROI 掩膜"])
        self.region_combo.currentIndexChanged.connect(self._on_region_changed)
        self.region_combo.currentIndexChanged.connect(self._invalidate_region)
        form.addRow("区域类型", self.region_combo)

        row = QHBoxLayout()
        self.dsv_spin = QDoubleSpinBox()
        self.dsv_spin.setRange(1.0, 40.0)
        self.dsv_spin.setValue(8.0)
        self.dsv_spin.setSuffix(" cm")
        self.dsv_spin.setDecimals(1)
        self.dsv_spin.valueChanged.connect(self._invalidate_region)
        row.addWidget(self.dsv_spin)
        self.centre_combo = QComboBox()
        self.centre_combo.addItems(["等中心", "线圈能量重心"])
        self.centre_combo.currentIndexChanged.connect(self._invalidate_region)
        row.addWidget(self.centre_combo)
        w = QWidget()
        w.setLayout(row)
        self.dsv_row = w
        form.addRow("DSV 直径 / 球心", w)

        self.roi_state_label = QLabel("")
        self.roi_state_label.setWordWrap(True)
        form.addRow("", self.roi_state_label)

        rrow = QHBoxLayout()
        self.roi_path = QLineEdit()
        self.roi_path.setPlaceholderText(
            "ROI 掩膜 .nii / .nii.gz（最近邻单向重采样到线圈网格）")
        self.roi_path.textChanged.connect(self._invalidate_region)
        rrow.addWidget(self.roi_path, 3)
        self.btn_browse_roi = QPushButton("浏览…")
        self.btn_browse_roi.clicked.connect(self._browse_roi)
        rrow.addWidget(self.btn_browse_roi)
        rw = QWidget(); rw.setLayout(rrow)
        form.addRow("ROI 文件", rw)

        self.coverage_label = QLabel("覆盖率：尚未计算")
        self.coverage_label.setObjectName("coverageLabel")
        form.addRow("DSV 覆盖率", self.coverage_label)

        self.dir_combo = QComboBox()
        self.dir_combo.addItems(DIRECTION_CHOICES)
        self.dir_combo.currentIndexChanged.connect(self._on_direction_changed)
        form.addRow("编码方向", self.dir_combo)

        self.dir_x = QDoubleSpinBox(); self.dir_y = QDoubleSpinBox()
        self.dir_z = QDoubleSpinBox()
        drow = QHBoxLayout()
        for sb, lab in ((self.dir_x, "x"), (self.dir_y, "y"), (self.dir_z, "z")):
            sb.setRange(-1.0, 1.0); sb.setDecimals(3); sb.setSingleStep(0.1)
            drow.addWidget(QLabel(lab)); drow.addWidget(sb)
        self.dir_x.setValue(1.0)
        dw = QWidget(); dw.setLayout(drow)
        self.dir_row = dw
        form.addRow("自定义向量 (RAS)", dw)

        self.dir_combo.setToolTip(
            "方向使用 NIfTI RAS 世界坐标；不等同于扫描仪相位编码方向。")
        return box

    # ── B. waveform and constraints ─────────────────────────────────
    def _build_constraints(self):
        box = QGroupBox("② 波形与编码约束")
        form = QFormLayout(box)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["目标梯度波形（指定方向 + 标量 g(t)）",
                                  "直接指定各通道电流波形（第一版仅支持 CSV）"])
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("仿真模式", self.mode_combo)

        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["sine", "cosine", "triangle", "trapezoid",
                                  "square", "piecewise_linear", "custom_csv"])
        self.wave_combo.currentTextChanged.connect(self._on_waveform_changed)
        form.addRow("波形", self.wave_combo)

        self.amp_spin = QDoubleSpinBox()
        self.amp_spin.setRange(0.0, 1e6); self.amp_spin.setValue(1000.0)
        self.amp_spin.setSuffix(" Hz/m"); self.amp_spin.setDecimals(1)
        form.addRow("幅度", self.amp_spin)

        self.freq_spin = QDoubleSpinBox()
        self.freq_spin.setRange(0.1, 1e5); self.freq_spin.setValue(100.0)
        self.freq_spin.setSuffix(" Hz")
        form.addRow("频率", self.freq_spin)

        self.dur_spin = QDoubleSpinBox()
        self.dur_spin.setRange(0.1, 1000.0); self.dur_spin.setValue(20.0)
        self.dur_spin.setSuffix(" ms"); self.dur_spin.setDecimals(2)
        form.addRow("持续时间", self.dur_spin)

        self.rate_spin = QSpinBox()
        self.rate_spin.setRange(1000, 5_000_000); self.rate_spin.setValue(100_000)
        self.rate_spin.setSuffix(" Sa/s")
        form.addRow("采样率", self.rate_spin)

        self.csv_path = QLineEdit()
        self.csv_path.setPlaceholderText("time_s, ch1, ch2, ...  （单位 A）")
        form.addRow("CSV 电流表", self.csv_path)

        self.static_path = QLineEdit()
        self.static_path.setPlaceholderText("留空表示静态基线为 0 A")
        form.addRow("静态匀场电流", self.static_path)

        self.wave_warning = QLabel("")
        self.wave_warning.setWordWrap(True)
        self.wave_warning.setObjectName("waveWarning")
        form.addRow("", self.wave_warning)

        # The two tolerances live here, not in a collapsed advanced panel: a
        # gradient without both is not a result.
        self.theta_spin = QDoubleSpinBox()
        # tan(90 deg) is not a number and a cone near it constrains nothing.
        # "No cone" is a separate checkbox, not theta = 90.
        self.theta_spin.setRange(0.1, 45.0); self.theta_spin.setValue(5.0)
        self.theta_spin.setSuffix(" °"); self.theta_spin.setDecimals(1)
        trow = QHBoxLayout()
        trow.addWidget(self.theta_spin)
        self.no_cone = QCheckBox("不加方向锥")
        self.no_cone.toggled.connect(
            lambda on: self.theta_spin.setEnabled(not on))
        trow.addWidget(self.no_cone)
        tw = QWidget(); tw.setLayout(trow)
        form.addRow("方向锥 θ_max（必需，≤45°）", tw)

        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setRange(0.01, 2.0); self.eps_spin.setValue(0.30)
        self.eps_spin.setDecimals(2); self.eps_spin.setSingleStep(0.05)
        form.addRow("非线性容差 ε（必需）", self.eps_spin)

        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setRange(0.1, 10.0); self.limit_spin.setValue(3.0)
        self.limit_spin.setSuffix(" A"); self.limit_spin.setDecimals(2)
        self.limit_spin.setEnabled(False)          # production hard limit
        form.addRow("电流限幅", self.limit_spin)

        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["reject（默认，超限不产生输出）",
                                      "uniform scale（整体缩放，保持波形与相对幅度）"])
        form.addRow("超限策略", self.strategy_combo)

        self.strategy_combo.setToolTip(
            "不做逐样本 clipping；uniform scale 只整体缩放动态波形。")
        return box

    def _build_actions(self):
        box = QGroupBox("③ 运行")
        row = QHBoxLayout(box)
        self.btn_coverage = QPushButton("计算区域与覆盖率")
        self.btn_run = QPushButton("运行离线仿真")
        self.btn_export = QPushButton("导出 JSON / CSV / 报告")
        self.btn_export.setEnabled(False)
        for b in (self.btn_coverage, self.btn_run, self.btn_export):
            row.addWidget(b)
        row.addStretch(1)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        row.addWidget(self.status, 2)
        return box

    # ── C. results ──────────────────────────────────────────────────
    def _build_results(self):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)

        self.headline = QLabel("尚未运行。")
        self.headline.setWordWrap(True)
        self.headline.setObjectName("encodingHeadline")
        lay.addWidget(self.headline)

        detail_tabs = QTabWidget()

        self.metric_table = QTableWidget(0, 2)
        self.metric_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metric_table.horizontalHeader().setStretchLastSection(True)
        detail_tabs.addTab(self.metric_table, "关键指标")

        self.current_table = QTableWidget(0, 5)
        self.current_table.setHorizontalHeaderLabels(
            ["通道", "静态 A", "动态峰值 A", "合计峰值 A", "余量 A"])
        self.current_table.horizontalHeader().setStretchLastSection(True)
        detail_tabs.addTab(self.current_table, "通道电流")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        detail_tabs.addTab(self.log, "运行日志")
        lay.addWidget(detail_tabs)
        return box

    def _build_plots(self):
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)

        top = QHBoxLayout()
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Hz/m", "mT/m"])
        self.unit_combo.currentIndexChanged.connect(self._draw_curves)
        top.addWidget(QLabel("梯度单位"))
        top.addWidget(self.unit_combo)
        self.decim_label = QLabel("")
        top.addWidget(self.decim_label, 2)
        lay.addLayout(top)

        self.plot_tabs = QTabWidget()

        grad_page = QWidget()
        grad_lay = QVBoxLayout(grad_page)
        self.fig_grad = Figure(figsize=(7, 3.6), tight_layout=True)
        self.canvas_grad = FigureCanvasQTAgg(self.fig_grad)
        grad_lay.addWidget(self.canvas_grad)
        self.plot_tabs.addTab(grad_page, "梯度波形")

        current_page = QWidget()
        crow = QHBoxLayout()
        self.channel_list = QListWidget()
        self.channel_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.channel_list.setMaximumWidth(230)
        self.channel_list.setMaximumHeight(170)
        self.channel_list.itemSelectionChanged.connect(self._draw_curves)
        crow.addWidget(self.channel_list)
        self.fig_cur = Figure(figsize=(8, 4.0), tight_layout=True)
        self.canvas_cur = FigureCanvasQTAgg(self.fig_cur)
        crow.addWidget(self.canvas_cur, 3)
        current_page.setLayout(crow)
        self.plot_tabs.addTab(current_page, "通道电流")

        k_page = QWidget()
        k_lay = QVBoxLayout(k_page)
        self.fig_k = Figure(figsize=(7, 3.6), tight_layout=True)
        self.canvas_k = FigureCanvasQTAgg(self.fig_k)
        self.canvas_k.setToolTip(K_NOTE)
        k_lay.addWidget(self.canvas_k)
        self.plot_tabs.addTab(k_page, "k 偏移")

        slice_page = QWidget()
        slice_lay = QVBoxLayout(slice_page)
        srow = QHBoxLayout()
        self.frame_combo = QComboBox()
        for ft in FRAME_TYPES:
            self.frame_combo.addItem(FRAME_LABELS[ft][0], ft)
        self.frame_combo.currentIndexChanged.connect(self._request_slices)
        srow.addWidget(QLabel("切面内容"))
        srow.addWidget(self.frame_combo, 2)
        for text, slot in (("首", lambda: self._goto("first")),
                           ("中", lambda: self._goto("middle")),
                           ("末", lambda: self._goto("last")),
                           ("峰值梯度", lambda: self._goto("peak_gradient")),
                           ("峰值相位", lambda: self._goto("peak_phase"))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            srow.addWidget(b)
        slice_lay.addLayout(srow)

        trow = QHBoxLayout()
        tlab = QLabel("时间点")
        tlab.setToolTip("拖动选择要显示的时刻；只改变下方三个切面显示的帧，"
                        "不会重新求解。")
        trow.addWidget(tlab)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setEnabled(False)
        self.time_slider.setToolTip(
            "时间点：选择波形上的一个时刻，下方三个切面显示该时刻的场/相位。")
        # A million-voxel product per pixel of drag would make the slider feel
        # broken. Redraw on release, or after the drag has paused.
        self.time_slider.sliderReleased.connect(self._request_slices)
        self.time_slider.valueChanged.connect(self._on_slider_moved)
        trow.addWidget(self.time_slider, 1)
        slice_lay.addLayout(trow)

        # Which plane index each orthogonal view cuts at. Without these the
        # three panels were stuck at the middle of the array, which for a
        # frontal array is often outside the region that has any field.
        srow2 = QHBoxLayout()
        self.slice_sliders = []
        for k, name in enumerate(("axis 0", "axis 1", "axis 2")):
            lab = QLabel(name)
            srow2.addWidget(lab)
            sl = QSlider(Qt.Horizontal)
            sl.setEnabled(False)
            sl.setToolTip("沿 %s 移动切面位置（只改显示，不重新计算）。" % name)
            sl.valueChanged.connect(self._on_plane_moved)
            sl.sliderReleased.connect(self._request_slices)
            srow2.addWidget(sl, 1)
            self.slice_sliders.append(sl)
        slice_lay.addLayout(srow2)

        self.frame_caption = QLabel("尚未运行。")
        self.frame_caption.setWordWrap(True)
        slice_lay.addWidget(self.frame_caption)

        self.fig_slice = Figure(figsize=(9, 3.8), tight_layout=True)
        self.canvas_slice = FigureCanvasQTAgg(self.fig_slice)
        slice_lay.addWidget(self.canvas_slice)

        self.nl_note = QLabel("")
        self.nl_note.setVisible(False)
        self.frame_combo.setToolTip(NONLINEAR_NOTE)
        self.plot_tabs.addTab(slice_page, "空间场")
        lay.addWidget(self.plot_tabs)
        return box

    # ── plotting ────────────────────────────────────────────────────
    def _title_suffix(self):
        r = self._result or {}
        return ("IDEALIZED | %s | eps=%s theta=%s | requested %s / model "
                "bound %s / actual %s Hz/m | scale %.4f"
                % (r.get("region_name", "—"), r.get("epsilon_nonlinear", "—"),
                   ("无锥" if r.get("theta_max_deg") is None
                    else "%.1f°" % r["theta_max_deg"]),
                   _fmt(r.get("requested_peak_gradient_hz_per_m")),
                   _fmt(r.get("capacity_gradient_hz_per_m")),
                   _fmt(r.get("actual_peak_gradient_hz_per_m")),
                   r.get("uniform_scale", 1.0)))

    def plot_data(self):
        """The decimated series the curves are drawn from. Display only."""
        rt = self._runtime
        if not rt:
            return None
        sim = rt["sim"]
        d = np.asarray(rt["direction"], dtype=float)
        idx = display_indices(sim.t.shape[0],
                              must_include=(rt["peak_index"],))
        scale = 1.0 / 42577.0 if self.unit_combo.currentIndex() == 1 else 1.0
        req = np.asarray(rt.get("requested_waveform")
                         if rt.get("requested_waveform") is not None
                         else np.zeros(sim.t.shape[0]), dtype=float)
        return {
            "idx": idx,
            "t": decimate(sim.t, idx),
            "requested": decimate(req, idx) * scale,
            "actual_dynamic": decimate(sim.g_dynamic @ d, idx) * scale,
            "total_parallel": decimate(sim.g_total @ d, idx) * scale,
            "orthogonal": decimate(
                np.linalg.norm(sim.g_dynamic - np.outer(sim.g_dynamic @ d, d),
                               axis=1), idx) * scale,
            "k_dynamic": decimate(sim.k_dynamic, idx),
            "k_total": decimate(sim.k_total_local_model, idx),
            "i_dynamic": decimate(sim.i_dynamic, idx),
            "i_total": decimate(sim.i_total, idx),
            "unit": self.unit_combo.currentText(),
            "report": decimation_report(sim.t.shape[0], idx),
        }

    def _draw_curves(self, *_):
        pd = self.plot_data()
        if pd is None:
            return
        r = self._result or {}
        rep = pd["report"]
        self.decim_label.setText(
            "完整采样 %d，绘图显示 %d（%s）"
            % (rep["full_samples"], rep["displayed_samples"],
               "等间隔抽取，保留首/末/峰值"))

        ax = self.fig_grad.clear() or self.fig_grad.add_subplot(111)
        ax.plot(pd["t"], pd["requested"], lw=1.4, label="requested")
        ax.plot(pd["t"], pd["actual_dynamic"], lw=1.4,
                label="model-predicted gradient along target (dynamic)")
        ax.plot(pd["t"], pd["total_parallel"], lw=1.0, ls="--",
                label="local coil static+dynamic, along target")
        ax.plot(pd["t"], pd["orthogonal"], lw=1.0, ls=":",
                label="orthogonal gradient magnitude (dynamic)")
        ax.set_xlabel("t (s)"); ax.set_ylabel(pd["unit"])
        ax.set_title(self._title_suffix(), fontsize=9)
        ax.legend(loc="upper right", framealpha=0.85)
        ax.grid(alpha=0.3)
        self.canvas_grad.draw_idle()

        ax = self.fig_cur.clear() or self.fig_cur.add_subplot(111)
        names = r.get("channel_names") or []
        picked = {i.data(Qt.UserRole) for i in self.channel_list.selectedItems()}
        lim = float(r.get("current_limit_a", 3.0))
        # Every channel is drawn, but faintly; a 36-entry legend would cover
        # the data it is labelling. Selecting a channel brings it forward.
        for j in range(pd["i_dynamic"].shape[1]):
            hot = (j in picked)
            ax.plot(pd["t"], pd["i_dynamic"][:, j], lw=1.6 if hot else 0.5,
                    alpha=1.0 if hot else 0.25,
                    label=(names[j] if hot and j < len(names) else None))
            ax.plot(pd["t"], pd["i_total"][:, j], lw=1.2 if hot else 0.4,
                    ls="--", alpha=0.9 if hot else 0.18)
        ax.axhline(lim, color="r", lw=0.8); ax.axhline(-lim, color="r", lw=0.8)
        for v in (r.get("static_currents_a") or []):
            if abs(float(v)) > 0:
                ax.axhline(float(v), color="g", lw=0.5, alpha=0.4)
        ax.set_xlabel("t (s)"); ax.set_ylabel("A")
        ax.set_title("channel currents — solid: dynamic, dashed: total, "
                     "red: +/-%.1f A" % lim, fontsize=9)
        if picked:
            ax.legend(loc="upper right", framealpha=0.85)
        ax.grid(alpha=0.3)
        self.canvas_cur.draw_idle()

        ax = self.fig_k.clear() or self.fig_k.add_subplot(111)
        for a, lab in ((0, "x"), (1, "y"), (2, "z")):
            ax.plot(pd["t"], pd["k_dynamic"][:, a], lw=1.3,
                    label="k_dynamic %s" % lab)
            ax.plot(pd["t"], pd["k_total"][:, a], lw=0.9, ls="--",
                    label="k_total_local %s" % lab)
        ax.set_xlabel("t (s)"); ax.set_ylabel("cycles/m")
        ax.set_title("idealised k offset from the added linear B0 component "
                     "— excludes scanner imaging gradients", fontsize=9)
        ax.legend(ncol=3, framealpha=0.85); ax.grid(alpha=0.3)
        self.canvas_k.draw_idle()

    def _sync_plane_sliders(self, shape):
        """Range the plane sliders to the volume, once per new result."""
        for k, sl in enumerate(self.slice_sliders):
            if sl.maximum() != shape[k] - 1:
                sl.blockSignals(True)
                sl.setRange(0, max(shape[k] - 1, 0))
                sl.setValue(shape[k] // 2)
                sl.blockSignals(False)
            sl.setEnabled(True)

    def _on_plane_moved(self, _value):
        """Redraw after a pause, so dragging does not queue a redraw per pixel."""
        if self._runtime:
            self._frame_timer.start()

    def _on_slider_moved(self, value):
        rt = self._runtime
        if not rt:
            return
        t = float(rt["sim"].t[int(value)])
        self.frame_caption.setText("index %d，t = %.6g s（松开或停顿后刷新切面）"
                                   % (int(value), t))
        self._frame_timer.start()

    def _request_slices(self, *_):
        self._frame_timer.stop()
        self._draw_slices()

    def _goto(self, where):
        rt = self._runtime
        if not rt:
            return
        n = rt["sim"].t.shape[0]
        idx = {"first": 0, "middle": n // 2, "last": n - 1,
               "peak_gradient": rt["peak_index"],
               "peak_phase": rt["peak_phase_index"]}.get(where, 0)
        self.time_slider.setValue(int(idx))
        self._request_slices()

    def current_frame(self):
        """The current volume, from the cache when it is already there."""
        rt = self._runtime
        if not rt:
            return None
        ft = self.frame_combo.currentData()
        k = int(self.time_slider.value())
        key = self._frame_cache.key(self._result_id, ft, k)
        vol = self._frame_cache.get(key)
        if vol is None:
            vol = frame_volume(rt["model"], rt["sim"], ft, k)
            self._frame_cache.put(key, vol)
        return vol, ft, k

    def _draw_slices(self):
        got = self.current_frame()
        if got is None:
            return
        vol, ft, k = got
        rt = self._runtime
        st = frame_stats(vol, ft, k, float(rt["sim"].t[k]))
        lims = symmetric_limits(vol)
        self.frame_caption.setText(
            "%s │ index %d，t = %.6g s │ 有效体素 %d │ min %.3g，max %.3g，"
            "RMS %.3g %s │ %s │ %s"
            % (st["label"], st["index"], st["time_s"], st["valid_voxels"],
               st["min"], st["max"], st["rms"], st["unit"], lims["mode"],
               rt["plane_note"]))

        shape = vol.shape
        self._sync_plane_sliders(shape)
        idx3 = [sl.value() for sl in self.slice_sliders]

        self.fig_slice.clear()
        planes = [("axis 0 = %d" % idx3[0], vol[idx3[0], :, :]),
                  ("axis 1 = %d" % idx3[1], vol[:, idx3[1], :]),
                  ("axis 2 = %d" % idx3[2], vol[:, :, idx3[2]])]
        # One colourbar for the row, not one squeezed into the last panel:
        # sharing it kept the three axes different widths and made the
        # anatomy look misaligned between views.
        gs = self.fig_slice.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.06],
                                         wspace=0.06)
        im = None
        for n, (title, sl) in enumerate(planes):
            ax = self.fig_slice.add_subplot(gs[0, n])
            im = ax.imshow(np.rot90(sl), cmap="jet", vmin=lims["vmin"],
                           vmax=lims["vmax"], interpolation="nearest",
                           aspect="equal", origin="upper")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        cax = self.fig_slice.add_subplot(gs[0, 3])
        if im is not None:
            self.fig_slice.colorbar(im, cax=cax)
            cax.tick_params(labelsize=8)
        self.canvas_slice.draw_idle()
        self.frame_combo.setToolTip(NONLINEAR_NOTE if ft.startswith("nonlinear")
                                    else "选择当前时间点要显示的空间量。")

    # ── behaviour ───────────────────────────────────────────────────
    def _on_region_changed(self, idx):
        dsv = (idx == 0)
        from_state = not dsv
        self.dsv_row.setVisible(dsv)
        self.centre_combo.setVisible(dsv)
        # The ROI comes from the mask the shim pages are already holding. A
        # typed path was only ever a way to point at the same thing, and a
        # folder typed by mistake surfaced as a permissions error.
        self.roi_path.setVisible(False)
        self.btn_browse_roi.setVisible(False)
        self.coverage_label.setVisible(dsv)
        if from_state:
            m = self.app_state.get("roi_mask")
            name = self.app_state.get("roi_name") or "一键匀场 ROI"
            if m is None:
                self.roi_state_label.setText(
                    "还没有 ROI 掩膜。先在一键匀场页跑一次，这里会自动带入它。")
            else:
                import numpy as _np
                self.roi_state_label.setText(
                    "ROI 掩膜「%s」：%d 个体素，来自一键匀场页，直接使用内存中的"
                    "掩膜。" % (name, int(_np.asarray(m).sum())))
        self.roi_state_label.setVisible(from_state)

    def _on_direction_changed(self, idx):
        custom = (idx == len(DIRECTION_CHOICES) - 1)
        self.dir_row.setVisible(custom)

    def _on_waveform_changed(self, kind):
        if kind == "square":
            self.wave_warning.setText(
                "方波不带限；硬件边沿与电压未建模。")
        elif kind == "custom_csv":
            self.wave_warning.setText(
                "CSV：time_s + 对应通道电流（A）。")
        else:
            self.wave_warning.setText("")

    def coil_unit_info(self) -> dict:
        """What the selected coil's own configuration says about its unit."""
        from core.coil_registry import get_coil

        try:
            c = get_coil(self.coil_combo.currentData())
        except Exception:
            return {"status": "unconfirmed", "unit": "Hz/A", "date": "",
                    "note": "", "source": "未知", "imported": True}
        return {
            "status": getattr(c, "field_unit_status", "unconfirmed"),
            "unit": getattr(c, "field_unit", "Hz/A"),
            "date": getattr(c, "field_unit_confirmed_date", ""),
            "note": getattr(c, "field_unit_note", ""),
            "source": str(getattr(c, "source_json", "") or "线圈配置"),
            "imported": bool(getattr(c, "imported", False)),
        }

    def _refresh_unit_source(self, *_):
        """Tick the box only when the coil's own config confirms the unit."""
        info = self.coil_unit_info()
        confirmed = info["status"] == "user_confirmed"
        self.unit_confirm.blockSignals(True)
        self.unit_confirm.setChecked(confirmed)
        self.unit_confirm.blockSignals(False)
        if confirmed:
            self.unit_source.setText(
                "✓ 已确认 · %s" % (info["date"] or "无日期"))
            self.unit_source.setStyleSheet("color:#2f855a;")
            self.unit_source.setToolTip(
                "%s · 来源：%s\n%s"
                % (info["unit"], info["source"], info["note"] or ""))
            self.unit_confirm.setToolTip(self.unit_source.toolTip())
        else:
            # This is what greys out the run button. A single line at the very
            # bottom of the page was too easy to miss, so the button read as
            # broken rather than gated -- say it here, next to the tickbox.
            self.unit_source.setText(
                "⚠ 未确认 → 「运行离线仿真」为灰，勾选左侧复选框后可运行")
            self.unit_source.setStyleSheet("color:#c05621; font-weight:600;")
            self.unit_source.setToolTip(
                "配置文件未记录可核验的场图单位。确认仅对当前线圈生效。")
            self.unit_confirm.setToolTip(
                self.unit_source.toolTip())
        self._invalidate_region()

    def _toggle_setup(self, hidden):
        """Fold the parameter panel away once a run is set up."""
        self.setup_scroll.setVisible(not hidden)
        self.btn_toggle_setup.setText("展开设置" if hidden else "收起设置")

    def _on_mode_changed(self, idx):
        """Direct-current mode does not optimise the field shape."""
        direct = (idx == 1)
        if direct:
            self.wave_combo.setCurrentText("custom_csv")
        self.wave_combo.setEnabled(not direct)
        self.amp_spin.setEnabled(not direct)
        self.freq_spin.setEnabled(not direct)
        self.dur_spin.setEnabled(not direct)
        self.rate_spin.setEnabled(not direct)
        self.eps_spin.setEnabled(not direct)
        self.theta_spin.setEnabled(not direct and not self.no_cone.isChecked())
        self.no_cone.setEnabled(not direct)
        self.csv_path.setEnabled(direct)
        self.wave_warning.setText(
            "直接电流模式 · 仅支持 CSV" if direct else "")

    def _browse_roi(self):
        """Pick a file, so a folder can never be typed in by hand."""
        from PySide6.QtWidgets import QFileDialog

        start = str(self.app_state.get("subject_dir") or "")
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ROI 掩膜", start, "NIfTI (*.nii *.nii.gz);;所有文件 (*)")
        if path:
            self.roi_path.setText(path)

    def _invalidate_region(self, *_):
        """Any input that changes the region invalidates the old coverage.

        Silently keeping a stale mask is how a result ends up describing a
        region the user is no longer looking at.
        """
        self._region = None
        self._clear_results()
        self.coverage_label.setText("覆盖率：输入已更改，请重新计算")
        self.coverage_label.setStyleSheet("")
        self._update_ready()

    def _clear_results(self):
        """Result, runtime objects, frame cache and every plot, together.

        Leaving a stale curve on screen after the inputs changed is how a
        figure ends up describing a region the user is no longer looking at.
        """
        self._result = None
        self._runtime = None
        self._frame_cache.clear()
        self.btn_export.setEnabled(False)
        self.time_slider.setEnabled(False)
        for sl in getattr(self, "slice_sliders", []):
            sl.setEnabled(False)
        self.channel_list.clear()
        self.frame_caption.setText("输入已更改，请重新运行。")
        for fig, canvas in ((self.fig_grad, self.canvas_grad),
                            (self.fig_cur, self.canvas_cur),
                            (self.fig_k, self.canvas_k),
                            (self.fig_slice, self.canvas_slice)):
            fig.clear()
            canvas.draw_idle()

    def mode(self):
        return MODE_DIRECT if self.mode_combo.currentIndex() == 1 else MODE_TARGET

    def collect_params(self) -> dict:
        """Everything the worker needs. No Qt objects cross the boundary."""
        idx = self.region_combo.currentIndex()
        theta = (None if (self.no_cone.isChecked()
                          or self.mode() == MODE_DIRECT)
                 else float(self.theta_spin.value()))
        return {
            "coil": self.coil_combo.currentData(),
            "region_type": "dsv" if idx == 0 else "roi",
            "roi_array": (self.app_state.get("roi_mask") if idx != 0 else None),
            "roi_affine": (self.app_state.get("b0_affine") if idx != 0 else None),
            "roi_label": (self.app_state.get("roi_name") or "一键匀场 ROI"),
            "dsv_cm": float(self.dsv_spin.value()),
            "centre_method": ("isocentre" if self.centre_combo.currentIndex() == 0
                              else "energy_centroid"),
            "roi_path": self.roi_path.text().strip(),
            "mode": self.mode(),
            "direction": [float(v) for v in self.direction_vector()],
            "waveform": self.wave_combo.currentText(),
            "requested_peak_hz_per_m": float(self.amp_spin.value()),
            "frequency_hz": float(self.freq_spin.value()),
            "duration_s": float(self.dur_spin.value()) / 1000.0,
            "sample_rate_hz": float(self.rate_spin.value()),
            "epsilon_nonlinear": float(self.eps_spin.value()),
            "theta_max_deg": theta,
            "csv_path": self.csv_path.text().strip(),
            "static_currents": self._static_currents(),
            "strategy": ("reject" if self.strategy_combo.currentIndex() == 0
                         else "uniform_scale"),
            "unit_status": (UnitStatus.USER_CONFIRMED
                            if self.unit_confirm.isChecked()
                            else UnitStatus.UNCONFIRMED),
            "unit_source": self.coil_unit_info(),
        }

    def _static_currents(self):
        txt = self.static_path.text().strip()
        if not txt:
            return None
        import numpy as _np
        from pathlib import Path as _P
        p = _P(txt)
        if p.is_file():
            return _np.loadtxt(str(p), delimiter=",").ravel().tolist()
        return [float(v) for v in txt.replace(";", ",").split(",") if v.strip()]

    # ── slots ───────────────────────────────────────────────────────
    def _busy(self, on, msg=""):
        for b in (self.btn_coverage, self.btn_run):
            b.setEnabled(not on)
        if on:
            self.btn_export.setEnabled(False)
        self.status.setText(msg)
        if not on:
            self._update_ready()

    def _start(self, job, params, region=None):
        if self._worker is not None and self._worker.isRunning():
            self.status.setText("已有任务在运行。")
            return False
        self._worker = LinearEncodingWorker(job, params, region, parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.failed.connect(self._on_failed)
        self._worker.coverage_finished.connect(self._on_coverage_done)
        self._worker.finished_ok.connect(self._on_run_done)
        self._busy(True, "运行中 …")
        self._worker.start()
        return True

    def on_coverage(self):
        self.log.append("— 计算区域与覆盖率 —")
        self._start(LinearEncodingWorker.JOB_COVERAGE, self.collect_params())

    def on_run(self):
        if not self.unit_confirm.isChecked():
            self.status.setText("未确认场图单位，无法运行正式仿真。")
            return
        if self._region is None:
            self.status.setText("请先计算区域与覆盖率。")
            return
        v = self.direction_vector()
        import numpy as _np
        if float(_np.linalg.norm(v)) <= 0:
            self.status.setText("编码方向为零向量。")
            return
        self.log.append("— 运行离线仿真 —")
        self._start(LinearEncodingWorker.JOB_SIMULATE, self.collect_params(),
                    self._region)

    def on_export(self, directory=None):
        if not self._result:
            self.status.setText("没有可导出的结果。")
            return None
        if directory is None:
            from pathlib import Path as _P
            from PySide6.QtWidgets import QFileDialog
            directory = QFileDialog.getExistingDirectory(self, "选择导出目录")
            if not directory:
                return None
        try:
            written = export_run(self._result, directory)
        except Exception as e:  # noqa: BLE001
            self.status.setText("导出失败：%s" % e)
            return None
        self._export_dir = str(directory)
        self.status.setText("已导出 %d 个文件到 %s"
                            % (len(written), directory))
        self.log.append("导出：" + "、".join(EXPORT_FILES))
        return written

    def _on_progress(self, pct, msg):
        self.status.setText("%d%%  %s" % (pct, msg))
        self.log.append("  %s" % msg)

    def _on_failed(self, msg, detail):
        self._region = self._region if self._result is None else self._region
        # A failure must never leave the previous result on screen pretending
        # to be the new one.
        self._clear_results()
        self.headline.setText("失败：%s" % msg)
        self.headline.setStyleSheet("color:#e53e3e;")
        self.metric_table.setRowCount(0)
        self.current_table.setRowCount(0)
        if detail:
            self.log.append(detail.splitlines()[-1][:200])
        self._busy(False, msg)

    def _on_coverage_done(self, region):
        self._region = region
        self.show_coverage(region)
        self.log.append("区域：%s，%d 体素"
                        % (region.get("region_name"), region.get("voxels", 0)))
        self._busy(False, "区域已就绪。")

    def _on_run_done(self, res):
        sim = res.get("_sim")
        model = res.get("_model")
        d = np.asarray(res.get("direction", [1.0, 0.0, 0.0]), dtype=float)
        phase_peak = int(np.argmax(np.abs(sim.q_dynamic).max(axis=1))) \
            if sim is not None else 0
        self._runtime = {
            "sim": sim, "model": model, "direction": d,
            "peak_index": int(res.get("peak_index", 0)),
            "peak_phase_index": phase_peak,
            "requested_waveform": res.get("requested_waveform_hz_per_m"),
            "plane_note": plane_note(model.affine) if model is not None else "",
        }
        self._result_id += 1
        self._frame_cache.clear()
        self._result = res
        self.show_result(res)

        self.channel_list.clear()
        b = res.get("bounds") or {}
        for j, name in enumerate(res.get("channel_names") or []):
            pc = (b.get("per_channel") or [{}] * (j + 1))[j]
            it = QListWidgetItem(
                "%s  dyn %.2f / tot %.2f A  余量 %.2f"
                % (name, pc.get("dynamic_peak_abs_a", 0.0),
                   pc.get("peak_abs_a", 0.0),
                   min(pc.get("headroom_up_a", 0.0),
                       pc.get("headroom_down_a", 0.0))))
            it.setData(Qt.UserRole, j)
            self.channel_list.addItem(it)

        n = sim.t.shape[0] if sim is not None else 1
        self.time_slider.setEnabled(True)
        self.time_slider.setRange(0, max(n - 1, 0))
        self.time_slider.setValue(self._runtime["peak_index"])
        self._draw_curves()
        self._request_slices()
        self.btn_export.setEnabled(True)
        self._busy(False, "仿真完成。")

    def _update_ready(self, *_):
        busy = self._worker is not None and self._worker.isRunning()
        ok = self.unit_confirm.isChecked() and not busy
        self.btn_run.setEnabled(ok)
        self.btn_coverage.setEnabled(not busy)
        if not self.unit_confirm.isChecked():
            self.status.setText(
                "「运行离线仿真」为灰：尚未确认「%s」的场图单位（见上方设置区）。"
                "可以先计算区域与覆盖率。"
                % (self.coil_combo.currentText() or "所选线圈"))
            self.btn_run.setToolTip(
                "未确认场图单位。勾选设置区的「场图单位」复选框后可运行。")
        elif busy:
            self.btn_run.setToolTip("已有任务在运行。")
        else:
            self.btn_run.setToolTip("")

    def direction_vector(self):
        i = self.dir_combo.currentIndex()
        if i == 0:
            return RAS_AXES["RAS-X"]
        if i == 1:
            return RAS_AXES["RAS-Y"]
        if i == 2:
            return RAS_AXES["RAS-Z"]
        return np.array([self.dir_x.value(), self.dir_y.value(),
                         self.dir_z.value()])

    def show_coverage(self, info: dict):
        """Paint the coverage verdict. Red and TRUNCATED below the threshold."""
        cov = float(info.get("coverage", 0.0))
        if info.get("truncated"):
            self.coverage_label.setText(
                "TRUNCATED  %d / %d = %.1f%%  （<%.0f%%）— 区域为「球 ∩ 有效标定范围」，"
                "不得据此给出最大编码能力结论"
                % (info["valid_sphere_voxels"], info["theoretical_sphere_voxels"],
                   100 * cov, 100 * COVERAGE_THRESHOLD))
            self.coverage_label.setStyleSheet("color:#e53e3e; font-weight:600;")
        else:
            self.coverage_label.setText(
                "完整 DSV  %d / %d = %.1f%%"
                % (info["valid_sphere_voxels"],
                   info["theoretical_sphere_voxels"], 100 * cov))
            self.coverage_label.setStyleSheet("color:#38a169; font-weight:600;")

    def show_result(self, res: dict):
        """Fill the tables. Every headline carries theta and epsilon."""
        self._result = res
        self.btn_export.setEnabled(True)
        # A completed run has no top-level "feasible" flag -- that lives on
        # the capacity solve. Reading it here made every successful run render
        # as unreachable.
        cap = res.get("capacity")
        infeasible = (cap is not None and not cap.get("feasible", True))
        if infeasible or not res.get("actual_peak_gradient_hz_per_m"):
            self.headline.setText(
                "当前场图模型、区域和约束下未找到非零可行解"
                "（ε=%s，θ≤%s，±%.1f A，区域 %s）。"
                % (res.get("epsilon_nonlinear"),
                   ("无锥" if res.get("theta_max_deg") is None
                    else "%.1f°" % res["theta_max_deg"]),
                   self.limit_spin.value(), res.get("region_name", "—")))
            self.headline.setStyleSheet("color:#e53e3e;")
            return
        self.headline.setStyleSheet("")
        self.headline.setText(
            "模型梯度 %.0f Hz/m（%.4f mT/m）  ·  请求 %s  ·  上界 %s Hz/m\n"
            "%s  ·  ε=%s  ·  θ=%s  ·  缩放 %.4f%s"
            % (res.get("actual_peak_gradient_hz_per_m", 0.0),
               res.get("actual_peak_gradient_mt_per_m", 0.0),
               ("%.0f" % res["requested_peak_gradient_hz_per_m"])
               if res.get("requested_peak_gradient_hz_per_m") else "—",
               ("%.0f" % res["capacity_gradient_hz_per_m"])
               if res.get("capacity_gradient_hz_per_m") else "—（直接电流模式）",
               res.get("region_name", "—"),
               res.get("epsilon_nonlinear", "—"),
               ("无锥" if res.get("theta_max_deg") is None
                else "%.1f°" % res["theta_max_deg"]),
               res.get("uniform_scale", 1.0),
               "  ·  含静态基线" if res.get("has_static") else ""))

        rows = [
            ("理想模型预测的沿目标方向梯度", "%.1f Hz/m  (%.4f mT/m)"
             % (res["gradient_parallel_actual_hz_per_m"],
                res["gradient_parallel_actual_mt_per_m"])),
            ("梯度模长", "%.1f Hz/m"
             % res.get("gradient_magnitude_actual_hz_per_m", float("nan"))),
            ("方向误差", "%.2f °" % res["direction_error_deg"]),
            ("正交泄漏比", "%.3f" % res["orthogonal_leakage_ratio"]),
            ("非线性空间场分量比（相对于目标线性梯度）",
             "%.3f" % res.get("nonlinear_spatial_ratio",
                              res.get("nonlinear_residual_ratio", float("nan")))),
            ("非线性空间场分量 RMS", "%.2f Hz"
             % res.get("nonlinear_spatial_rms_hz",
                       res.get("nonlinear_residual_rms_hz", float("nan")))),
            ("拟合 NRMSE（次要）", "%.3f — 同时含正交泄漏与非线性"
             % res.get("target_fit_nrmse", float("nan"))),
            ("电流峰值", "%.3f A" % res.get("current_peak_a", float("nan"))),
            ("达到 ±3 A 电流边界的通道",
             ", ".join(res.get("channels_at_current_bound")
                       or res.get("saturated_channels") or []) or "无"),
            ("均匀缩放因子", "%.4f" % res.get("uniform_scale", 1.0)),
            ("求解器", "%s / %s" % (res.get("solver", "—"),
                                    res.get("status", "—"))),
            ("约束违反（箱 / 非线性 / 锥）",
             "%.2e / %.2e / %.2e" % (res.get("box_violation_a", 0.0),
                                     res.get("nonlinear_constraint_relative_violation", 0.0),
                                     res.get("direction_cone_relative_violation", 0.0))),
            ("clipping", "从未使用"),
            ("电压 / 功率 / 动态可实现性", "UNKNOWN — 缺 R、L、互感、Vmax、H(f)"),
            ("梯度耦合", "not_characterized"),
        ]
        self.metric_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.metric_table.setItem(r, 0, QTableWidgetItem(k))
            self.metric_table.setItem(r, 1, QTableWidgetItem(v))

        cur = res.get("per_channel") or []
        self.current_table.setRowCount(len(cur))
        for r, c in enumerate(cur):
            vals = [c.get("channel", ""), "%.3f" % c.get("static_a", 0.0),
                    "%.3f" % c.get("dynamic_peak_abs_a", 0.0),
                    "%.3f" % c.get("peak_abs_a", 0.0),
                    "%.3f" % min(c.get("headroom_up_a", 0.0),
                                 c.get("headroom_down_a", 0.0))]
            for col, v in enumerate(vals):
                item = QTableWidgetItem(str(v))
                if c.get("saturated"):
                    item.setForeground(Qt.red)
                self.current_table.setItem(r, col, item)

    def shutdown(self):
        """Stop the worker before Qt starts deleting things.

        The page owns no serial port, but it does own a QThread, and a QThread
        destroyed while running aborts the process -- which is exactly the bug
        that made the packaged app crash on every exit.
        """
        w = self._worker
        if w is not None and w.isRunning():
            w.requestInterruption()
            w.quit()
            w.wait(5000)

    def apply_theme(self, dark: bool, p: dict):
        self.p = p
        self.banner.setStyleSheet(
            "background:%s; color:%s; border:1px solid %s; border-radius:12px;"
            "padding:5px 10px; font-size:12px; font-weight:600;"
            % (p.get("accent_soft", "#eef5ff"), p.get("accent", "#0071e3"),
               p.get("accent", "#0071e3")))


def _fmt(v):
    return "—" if v is None else ("%.0f" % v)
