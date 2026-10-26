"""One-shot shimming page: drop a DICOM folder, press once, read the result.

Replaces the old B0 / Optimize / Results trio. Everything the pipeline can
work out for itself -- NIfTI output location, brain mask, channel maps -- is
worked out, so the only decisions left are the two that are genuinely the
user's: which coil, and which region.

Advanced numbers (ΔTE, field strength, λ) stay collapsed; they have sane
defaults and are wrong to put in front of someone on the first run.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QFrame, QFileDialog, QTextEdit, QProgressBar, QGroupBox, QGridLayout,
    QDoubleSpinBox, QSizePolicy, QMessageBox, QToolButton, QDialog,
    QLineEdit, QDialogButtonBox, QCheckBox, QRadioButton, QButtonGroup,
    QSlider,
)
from PySide6.QtCore import Qt, Slot, Signal, QMimeData
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as mcm
from matplotlib.colors import TwoSlopeNorm

from core.theme import palette, style_axes
from core.coil_registry import (list_coils, get_default as default_coil,
                                get_coil, inspect_folder, add_coil,
                                remove_coil, is_removable)
from core.roi_registry import list_rois, load_custom_mask, CUSTOM_KEY, ROIError
from core.pipeline import looks_like_dicom_dir
from widgets.error_card import ErrorCard
from workers.shim_pipeline_worker import ShimPipelineWorker

#: Index of the Teensy page in main_window._NAV. Kept here so the handoff
#: names its destination once rather than at each call site.
#: Index of the Teensy page in main_window._NAV.
_NAV_TEENSY = 2


class DropZone(QFrame):
    """Large drop target for a DICOM folder; also clickable."""

    def __init__(self, on_path, parent=None):
        super().__init__(parent)
        self.on_path = on_path
        self.setAcceptDrops(True)
        self.setObjectName("dropZone")
        self.setMinimumHeight(96)
        self.setCursor(Qt.PointingHandCursor)

        v = QVBoxLayout(self)
        v.setContentsMargins(18, 14, 18, 14)
        v.setSpacing(4)
        self.title = QLabel("把 DICOM 文件夹拖到这里")
        self.title.setAlignment(Qt.AlignCenter)
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        v.addWidget(self.title)
        self.sub = QLabel("")
        self.sub.setAlignment(Qt.AlignCenter)
        self.sub.setObjectName("hint")
        v.addWidget(self.sub)

    def mousePressEvent(self, event):
        d = QFileDialog.getExistingDirectory(self, "选择 DICOM 文件夹")
        if d:
            self.on_path(d)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)

    def dropEvent(self, event: QDropEvent):
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p:
                # A dropped file inside the series is a reasonable mistake --
                # take its folder rather than rejecting the drop.
                self.on_path(p if os.path.isdir(p) else os.path.dirname(p))
                break
        event.acceptProposedAction()

    def show_path(self, path: str, ok: bool, note: str = ""):
        self.title.setText(os.path.basename(str(path).rstrip("\\/")) or str(path))
        self.sub.setText(note or str(path))
        self.setProperty("dropState", "ok" if ok else "bad")
        self.style().unpolish(self)
        self.style().polish(self)


class ShimWidget(QWidget):
    #: Ask the main window to bring another page forward, by nav index.
    request_page = Signal(int)

    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = palette(False)
        self.worker = None
        self.dicom_dir = ""
        self.result = None
        #: A mask file the user imported this session; overrides the ROI key.
        self.custom_mask_path = None
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(10)

        title = QLabel("一键匀场")
        title.setObjectName("pageTitle")
        root.addWidget(title)


        self.drop = DropZone(self._set_dicom_dir)
        root.addWidget(self.drop)

        # ── choices ──
        row = QHBoxLayout()
        row.setSpacing(14)

        row.addWidget(QLabel("线圈:"))
        self.coil_combo = QComboBox()
        self.coil_combo.setMinimumWidth(170)
        for c in list_coils():
            self.coil_combo.addItem(str(c), c.name)
        d = default_coil()
        if d:
            i = self.coil_combo.findData(d.name)
            if i >= 0:
                self.coil_combo.setCurrentIndex(i)
        self.coil_combo.currentIndexChanged.connect(self._update_coil_buttons)
        row.addWidget(self.coil_combo)

        self.btn_add_coil = QToolButton()
        self.btn_add_coil.setText("+")
        self.btn_add_coil.setObjectName("ghostBtn")
        self.btn_add_coil.setToolTip("添加线圈：选择标定文件夹，自动识别通道")
        self.btn_add_coil.clicked.connect(self._import_coil)
        row.addWidget(self.btn_add_coil)

        self.btn_del_coil = QToolButton()
        self.btn_del_coil.setText("−")
        self.btn_del_coil.setObjectName("ghostBtn")
        self.btn_del_coil.setToolTip("删除选中的线圈（仅限自己导入的）")
        self.btn_del_coil.clicked.connect(self._delete_coil)
        row.addWidget(self.btn_del_coil)
        # Reflect the initial selection: without this the delete button starts
        # enabled on a built-in coil, inviting a click that can only be refused.
        self._update_coil_buttons()

        row.addWidget(QLabel("模式:"))
        self.rb_static = QRadioButton("静态")
        self.rb_dynamic = QRadioButton("动态")
        self.rb_static.setChecked(True)
        self.rb_static.setToolTip("整个脑区一组电流")
        self.rb_dynamic.setToolTip("沿层方向逐层求解，输出电流表")
        grp = QButtonGroup(self)
        grp.addButton(self.rb_static)
        grp.addButton(self.rb_dynamic)
        self._mode_group = grp
        self.rb_dynamic.toggled.connect(self._on_mode_changed)
        row.addWidget(self.rb_static)
        row.addWidget(self.rb_dynamic)

        row.addWidget(QLabel("区域:"))
        self.roi_combo = QComboBox()
        self.roi_combo.setMinimumWidth(150)
        for r in list_rois():
            self.roi_combo.addItem(r.label, r.key)
            self.roi_combo.setItemData(self.roi_combo.count() - 1,
                                       r.description, Qt.ToolTipRole)
        #: What the box read before dynamic mode forced it to whole brain.
        self._static_roi_key = None
        self.roi_combo.currentIndexChanged.connect(self._on_roi_changed)
        row.addWidget(self.roi_combo)

        self.btn_adv = QToolButton()
        self.btn_adv.setText("高级设置 ▾")
        self.btn_adv.setCheckable(True)
        self.btn_adv.setObjectName("ghostBtn")
        self.btn_adv.toggled.connect(self._toggle_advanced)
        row.addWidget(self.btn_adv)

        row.addStretch()

        self.btn_run = QPushButton("开始匀场")
        self.btn_run.setObjectName("primaryBtn")
        self.btn_run.setMinimumHeight(40)
        self.btn_run.setMinimumWidth(140)
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run)
        row.addWidget(self.btn_run)
        root.addLayout(row)

        root.addWidget(self._build_advanced())

        # ── progress + log ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        root.addWidget(self.progress)

        self.error_card = ErrorCard()
        root.addWidget(self.error_card)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(150)
        self.log.setLineWrapMode(QTextEdit.NoWrap)
        root.addWidget(self.log)

        # ── results ──
        self.results_box = self._build_results()
        self.results_box.setVisible(False)
        root.addWidget(self.results_box, stretch=1)

    def _build_advanced(self) -> QGroupBox:
        box = QGroupBox("高级设置")
        box.setVisible(False)
        self.adv_box = box
        g = QGridLayout(box)

        self.auto_params = QCheckBox("自动读取 ΔTE / 场强")
        self.auto_params.setChecked(True)
        self.auto_params.setToolTip(
            "从 DICOM header 读 EchoTime 与 MagneticFieldStrength。"
            "取消勾选才使用下面填的值。")
        self.auto_params.toggled.connect(
            lambda on: (self.dte.setEnabled(not on), self.field.setEnabled(not on)))
        g.addWidget(self.auto_params, 1, 0, 1, 4)

        g.addWidget(QLabel("ΔTE (s):"), 0, 0)
        self.dte = QDoubleSpinBox()
        self.dte.setDecimals(5)                 # before setValue -- see b0_widget
        self.dte.setRange(0.0001, 0.1)
        self.dte.setSingleStep(0.0001)
        self.dte.setValue(0.00246)
        self.dte.setToolTip("回波间隔，单位秒。整张场图与它成反比。")
        self.dte.setEnabled(False)
        g.addWidget(self.dte, 0, 1)

        g.addWidget(QLabel("场强 (T):"), 0, 2)
        self.field = QDoubleSpinBox()
        self.field.setRange(0.5, 14.0)
        self.field.setValue(5.0)
        self.field.setEnabled(False)
        g.addWidget(self.field, 0, 3)

        g.addWidget(QLabel("λ (非ROI约束):"), 0, 4)
        self.lam = QDoubleSpinBox()
        self.lam.setDecimals(2)
        self.lam.setRange(0.0, 2.0)
        self.lam.setSingleStep(0.05)
        self.lam.setValue(0.5)
        self.lam.setToolTip(
            "目标函数是 mean(残差²_ROI) + λ·mean(残差²_非ROI)。"
            "λ 越小越偏向 ROI，其余脑区退化越多；选「全脑」时无影响。")
        g.addWidget(self.lam, 0, 5)

        g.setColumnStretch(6, 1)
        return box

    def _toggle_advanced(self, on: bool):
        self.adv_box.setVisible(on)
        self.btn_adv.setText("高级设置 ▴" if on else "高级设置 ▾")

    def _build_results(self) -> QGroupBox:
        box = QGroupBox("结果")
        v = QVBoxLayout(box)

        self.cards = {}
        cards = QHBoxLayout()
        for key in ["ROI RMSE", "ROI STD", "全脑 RMSE", "耗时"]:
            cards.addWidget(self._make_card(key))
        v.addLayout(cards)

        # How much context to show. Cropping to the ROI loses the anatomy the
        # eye needs to judge whether the region even landed in the right place,
        # so the default keeps the whole brain and outlines the ROI on top.
        modes = QHBoxLayout()
        modes.addWidget(QLabel("显示:"))
        self.view_mode = QComboBox()
        self.view_mode.addItem("全脑 + ROI 轮廓", "highlight")
        self.view_mode.addItem("全脑", "brain")
        self.view_mode.addItem("仅 ROI", "roi")
        self.view_mode.currentIndexChanged.connect(self._draw)
        modes.addWidget(self.view_mode)
        modes.addWidget(QLabel("切片:"))
        self.slice_at = QComboBox()
        self.slice_at.addItem("ROI 中心", "roi")
        self.slice_at.addItem("体积中心", "volume")
        self.slice_at.currentIndexChanged.connect(self._draw)
        modes.addWidget(self.slice_at)
        modes.addStretch()
        v.addLayout(modes)

        self.fig = Figure(figsize=(11, 5), facecolor=self.p["fig_bg"])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(300)
        v.addWidget(self.canvas, stretch=1)

        # ── dynamic-only: the current table, and one slice out of it ──
        self.dyn_box = QWidget()
        dv = QVBoxLayout(self.dyn_box)
        dv.setContentsMargins(0, 0, 0, 0)

        self.fig_heat = Figure(figsize=(11, 4), facecolor=self.p["fig_bg"])
        self.canvas_heat = FigureCanvas(self.fig_heat)
        self.canvas_heat.setMinimumHeight(260)
        # Clicking a row is the natural way to ask "what does that layer do";
        # a spin box beside the image would make the reader do the lookup.
        self.canvas_heat.mpl_connect("button_press_event", self._on_heat_click)
        dv.addWidget(self.canvas_heat, stretch=1)

        pick = QHBoxLayout()
        self.slice_caption = QLabel("")
        self.slice_caption.setStyleSheet("font-weight: 600;")
        pick.addWidget(self.slice_caption)
        pick.addStretch()
        pick.addWidget(QLabel("层:"))
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimumWidth(240)
        self.slice_slider.valueChanged.connect(self._draw_slice_card)
        pick.addWidget(self.slice_slider)
        dv.addLayout(pick)

        self.fig_slice = Figure(figsize=(11, 2.4), facecolor=self.p["fig_bg"])
        self.canvas_slice = FigureCanvas(self.fig_slice)
        self.canvas_slice.setMinimumHeight(180)
        dv.addWidget(self.canvas_slice)

        self.dyn_box.setVisible(False)
        v.addWidget(self.dyn_box, stretch=1)

        actions = QHBoxLayout()
        self.btn_teensy = QPushButton("写入放大器")
        self.btn_teensy.setObjectName("primaryBtn")
        self.btn_teensy.setToolTip(
            "把电流交给「Teensy 上传」页并切换过去，在那里连接串口后上传。")
        self.btn_teensy.clicked.connect(self._send_to_teensy)
        self.btn_export = QPushButton("导出结果")
        self.btn_export.clicked.connect(self._export)
        actions.addWidget(self.btn_teensy)
        actions.addWidget(self.btn_export)
        actions.addStretch()
        self.result_note = QLabel("")
        self.result_note.setObjectName("hint")
        actions.addWidget(self.result_note)
        v.addLayout(actions)
        return box

    def _make_card(self, label):
        f = QFrame()
        f.setObjectName("card")
        vl = QVBoxLayout(f)
        vl.setContentsMargins(14, 10, 14, 12)
        vl.setSpacing(3)
        t = QLabel(label)
        t.setObjectName("hint")
        vl.addWidget(t)
        val = QLabel("—")
        val.setStyleSheet(f"font-size: 16px; font-weight: 600; color: {self.p['text']};")
        vl.addWidget(val)
        self.cards[label] = val
        return f

    # ── coil management ───────────────────────────────────────────────
    def _refresh_coils(self, select: str = ""):
        keep = select or self.coil_combo.currentData()
        self.coil_combo.blockSignals(True)
        self.coil_combo.clear()
        for c in list_coils():
            self.coil_combo.addItem(str(c), c.name)
        i = self.coil_combo.findData(keep)
        if i < 0:
            d = default_coil()
            i = self.coil_combo.findData(d.name) if d else 0
        self.coil_combo.setCurrentIndex(max(i, 0))
        self.coil_combo.blockSignals(False)
        self._update_coil_buttons()

    def _update_coil_buttons(self):
        c = get_coil(self.coil_combo.currentData())
        self.btn_del_coil.setEnabled(bool(c and is_removable(c)))

    def _import_coil(self):
        folder = QFileDialog.getExistingDirectory(self, "选择线圈标定文件夹")
        if not folder:
            return

        # Inspect before asking for anything: no point making someone fill in a
        # form for a folder that was never going to work.
        info = inspect_folder(folder)
        if info["usable"] == 0:
            QMessageBox.warning(self, "无法作为线圈导入", info["problem"])
            return
        if info["missing"]:
            if QMessageBox.warning(
                    self, "部分通道缺少场图",
                    info["problem"]
                    + f"\n\n仍可导入 {info['usable']}/{info['channels']} "
                      "个通道，继续吗？",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return

        dlg = QDialog(self)
        dlg.setWindowTitle("添加线圈")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"识别到 {info['usable']} 个可用通道"))
        v.addWidget(QLabel(folder))
        form = QGridLayout()
        form.addWidget(QLabel("名称:"), 0, 0)
        name_edit = QLineEdit(info["suggested_name"])
        name_edit.selectAll()
        form.addWidget(name_edit, 0, 1)
        form.addWidget(QLabel("电流上限 (A):"), 1, 0)
        lim = QDoubleSpinBox()
        lim.setDecimals(1)
        lim.setRange(0.1, 20.0)
        lim.setSingleStep(0.5)
        lim.setValue(3.0)
        form.addWidget(lim, 1, 1)
        v.addLayout(form)
        hint = QLabel("场图会复制到用户目录，程序更新后仍然保留。")
        hint.setObjectName("hint")
        v.addWidget(hint)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec() != QDialog.Accepted:
            return

        try:
            coil = add_coil(folder, name_edit.text(), lim.value(),
                            progress_cb=self.log.append)
        except ValueError as e:
            QMessageBox.warning(self, "导入失败", str(e))
            return
        self.log.append(f"[线圈] 已导入「{coil.name}」，{coil.channel_count} 通道")
        self._refresh_coils(select=coil.name)

    def _delete_coil(self):
        c = get_coil(self.coil_combo.currentData())
        if c is None:
            return
        if not is_removable(c):
            QMessageBox.information(self, "无法删除",
                                    f"「{c.name}」是软件内置的线圈，不能删除。")
            return
        if QMessageBox.question(
                self, "删除线圈",
                f"删除「{c.name}」及其已复制的场图？\n原始文件夹不受影响。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No) != QMessageBox.Yes:
            return
        remove_coil(c.name)
        self.log.append(f"[线圈] 已删除「{c.name}」")
        self._refresh_coils()

    # ── input ─────────────────────────────────────────────────────────
    def _set_dicom_dir(self, path: str):
        ok, why = looks_like_dicom_dir(path)
        self.dicom_dir = path if ok else ""
        self.btn_run.setEnabled(ok)
        self.drop.show_path(path, ok, "" if ok else why)
        if not ok:
            self.log.append(f"[输入] {why}")

    # ── run ───────────────────────────────────────────────────────────
    def _run(self):
        if not self.dicom_dir:
            return
        coil = get_coil(self.coil_combo.currentData())
        if coil is None:
            QMessageBox.warning(self, "没有线圈配置",
                                "没有可用的线圈。请检查 config/coils/ 下的配置文件。")
            return

        self.log.clear()
        self.results_box.setVisible(False)
        self.btn_run.setEnabled(False)
        self.btn_run.setText("匀场中…")
        self.progress.setVisible(True)

        self.error_card.clear()
        dynamic = self.rb_dynamic.isChecked()
        roi_key = self.roi_combo.currentData()
        self.worker = ShimPipelineWorker(
            self.dicom_dir, coil=coil, mode="dynamic" if dynamic else "static",
            roi="whole_brain" if roi_key == CUSTOM_KEY else roi_key,
            custom_mask_path=(self.custom_mask_path
                              if roi_key == CUSTOM_KEY else None),
            delta_te=self.dte.value(), field_t=self.field.value(),
            auto_params=self.auto_params.isChecked(),
            lambda_penalty=self.lam.value(), parent=self)
        self.worker.progress.connect(self.log.append)
        self.worker.finished.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _reset_button(self):
        self.btn_run.setEnabled(bool(self.dicom_dir))
        self.btn_run.setText("开始匀场")
        self.progress.setVisible(False)

    # ── Dynamic result rendering ─────────────────────────────────────
    def _draw_heatmap(self):
        """The whole current table at a glance: slices down, channels across.

        A diverging map centred on zero, because the sign is the point -- which
        channels push and which pull, and where along the head that flips. A
        sequential map would hide the sign change that makes the table dynamic.
        """
        r = self.result
        p = self.p
        cur = r.slice_currents
        self.fig_heat.clear()
        self.fig_heat.set_facecolor(p["fig_bg"])
        ax = self.fig_heat.add_subplot(111)

        lim = float(np.abs(cur).max()) or 1.0
        im = ax.imshow(cur, aspect="auto", cmap="RdBu_r",
                       vmin=-lim, vmax=lim, interpolation="nearest",
                       extent=[0.5, cur.shape[1] + 0.5, cur.shape[0] + 0.5, 0.5])
        ax.set_xlabel("Channel")
        ax.set_ylabel("Slice")
        ax.set_title(f"Current table  {cur.shape[0]} slices x {cur.shape[1]} "
                     f"channels  (click a row for that slice)", fontsize=10)
        names = list(r.channel_names or [])
        step = max(1, cur.shape[1] // 12)
        ax.set_xticks(range(1, cur.shape[1] + 1, step))
        if names:
            ax.set_xticklabels([names[i] for i in range(0, cur.shape[1], step)],
                               rotation=45, ha="right", fontsize=8)
        cb = self.fig_heat.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label("Current (A)")

        # Skipped slices carry no information; hatching them stops a band of
        # exact zeros from reading as "these layers need no current".
        solved = r.slice_solved
        if solved is not None and not solved.all():
            for z in np.where(~solved)[0]:
                ax.axhspan(z + 0.5, z + 1.5, color=p["fig_bg"], alpha=0.75, lw=0)

        self._heat_ax = ax
        style_axes(ax, p)
        self.fig_heat.tight_layout()
        self.canvas_heat.draw()

    def _on_heat_click(self, event):
        if event.inaxes is None or self.result is None:
            return
        z = int(round(event.ydata)) - 1
        n = self.result.slice_currents.shape[0]
        self.slice_slider.setValue(max(0, min(n - 1, z)))

    def _draw_slice_card(self):
        r = self.result
        if r is None or r.slice_currents is None:
            return
        p = self.p
        z = int(self.slice_slider.value())
        cur = r.slice_currents[z]
        n = r.slice_currents.shape[0]

        if r.slice_solved is not None and not r.slice_solved[z]:
            self.slice_caption.setText(
                f"Slice {z + 1} / {n}   too few brain voxels, not solved "
                f"(currents 0)")
        else:
            self.slice_caption.setText(
                f"Slice {z + 1} / {n}  |  RMSE {r.slice_rmse_before[z]:.1f} → "
                f"{r.slice_rmse_after[z]:.1f} Hz  |  current "
                f"{cur.min():+.2f} ~ {cur.max():+.2f} A")

        self.fig_slice.clear()
        self.fig_slice.set_facecolor(p["fig_bg"])
        ax = self.fig_slice.add_subplot(111)
        names = list(r.channel_names or []) or [f"ch{i+1}" for i in range(len(cur))]
        colours = [p["danger"] if v >= 0 else p["accent"] for v in cur]
        ax.bar(range(len(cur)), cur, color=colours)
        ax.axhline(0, color=p["text_3"], lw=0.8)
        ax.set_xticks(range(len(cur)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Current (A)")
        # Fixed to the table's full range, so bars stay comparable between
        # slices instead of every slice being auto-scaled to look the same.
        lim = float(np.abs(r.slice_currents).max()) or 1.0
        ax.set_ylim(-lim * 1.1, lim * 1.1)
        style_axes(ax, p)
        self.fig_slice.tight_layout()
        self.canvas_slice.draw()

        if getattr(self, "_heat_ax", None) is not None:
            for ln in list(getattr(self, "_heat_marks", [])):
                try:
                    ln.remove()
                except Exception:
                    pass
            self._heat_marks = [
                self._heat_ax.axhline(z + 1, color=self.p["text"], lw=1.2, ls="--")]
            self.canvas_heat.draw()

    # ── Custom ROI ───────────────────────────────────────────────────
    def _on_mode_changed(self, dynamic: bool):
        """Dynamic mode shims every slice of the head, so an anatomical ROI
        has nothing to select -- the sequence steps through slices, not regions.

        The box is reset to whole brain, not merely greyed out. `run_pipeline`
        already forces `roi = whole_brain` for dynamic runs, so leaving a
        disabled box reading 「前额叶」 showed a region that was not going to be
        used and could not be changed. The previous choice is restored on the
        way back to static, so switching modes to look at the table does not
        cost the user their ROI selection.
        """
        if dynamic:
            if self._static_roi_key is None:
                self._static_roi_key = self.roi_combo.currentData()
            self._select_roi("whole_brain")
        elif self._static_roi_key is not None:
            self._select_roi(self._static_roi_key)
            self._static_roi_key = None
        self.roi_combo.setEnabled(not dynamic)
        self.roi_combo.setToolTip(
            "动态匀场逐层求解，固定使用全脑；静态模式下可选脑区。" if dynamic else "")
        self.btn_run.setText("开始动态匀场" if dynamic else "开始匀场")

    def _select_roi(self, key):
        """Move the box to `key` without re-triggering the custom-file dialog."""
        idx = self.roi_combo.findData(key)
        if idx < 0:
            return
        self.roi_combo.blockSignals(True)
        self.roi_combo.setCurrentIndex(idx)
        self.roi_combo.blockSignals(False)

    def _on_roi_changed(self, _index: int):
        """Picking 「自定义…」 asks for the file straight away.

        The mask cannot be validated until a B0 map exists, so it is only read
        here far enough to fail fast on an unreadable file; the shape check
        happens in the worker, against the grid that run actually produced.
        """
        if self.roi_combo.currentData() != CUSTOM_KEY:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 ROI mask 文件", "",
            "NIfTI 图像 (*.nii *.nii.gz);;所有文件 (*)")
        if not path:
            self._select_roi("whole_brain")
            return
        self.custom_mask_path = path
        self.roi_combo.setItemText(self.roi_combo.currentIndex(),
                                   f"自定义：{Path(path).name}")
        self.log.append(f"[ROI] 已选择自定义 mask：{path}")

    def _select_roi(self, key: str):
        idx = self.roi_combo.findData(key)
        if idx >= 0:
            self.roi_combo.blockSignals(True)
            self.roi_combo.setCurrentIndex(idx)
            self.roi_combo.blockSignals(False)

    # ── Outcome ──────────────────────────────────────────────────────
    # Three parameters, matching the worker's signal. Declaring two made Qt
    # drop the code silently, so the card always showed an empty one.
    @Slot(str, str, str)
    def _on_failed(self, message: str, technical: str, code: str = ""):
        self._reset_button()
        self.log.append("\n[失败] " + message)
        if technical:
            self.log.append(f"[技术细节] {technical}")
        self.error_card.show_error(message, technical, code)

    @Slot(object)
    def _on_done(self, result):
        self._reset_button()
        self.result = result

        self.error_card.clear()
        for w in getattr(result, "warnings", []) or []:
            self.error_card.show_warning(w.get("headline", ""),
                                         w.get("detail", ""), w.get("code", ""))

        # Hand the currents to the Teensy page.
        self.app_state["optimal_currents"] = result.currents
        self.app_state["ch_names_active"] = result.channel_names
        self.app_state["currents_source"] = "one_click"
        self.app_state["b0_vol3d"] = result.b0
        self.app_state["b0_affine"] = result.affine
        self.app_state["b0_shimmed_vol"] = result.residual
        self.app_state["shim_stats"] = result.stats

        # Everything the 系统匀场 page needs, so it never re-reads a DICOM,
        # re-runs BET, or re-resamples the coil maps. `coil_maps` in particular
        # is already on this subject's grid -- loading it again there would
        # repeat several seconds of work to arrive at the same array.
        self.app_state["b0_data"] = result.b0
        self.app_state["brain_mask"] = result.brain_mask
        self.app_state["roi_mask"] = result.roi_mask
        self.app_state["coil"] = get_coil(self.coil_combo.currentData())
        self.app_state["coil_maps"] = result.coil_maps
        self.app_state["coil_channel_names"] = result.channel_names
        self.app_state["subject"] = result.subject
        self.app_state["subject_dir"] = str(result.nifti_dir)
        # The field map's own transmit frequency, so the 系统匀场 page can name
        # an absolute f0 rather than an offset whose baseline the operator has
        # to remember. None when the DICOM did not give one consistent value.
        self.app_state["imaging_frequency_hz"] = result.imaging_frequency_hz

        s = result.stats

        self.cards["ROI RMSE"].setText(
            f"{s['roi_rmse_before']:.1f} → {s['roi_rmse_after']:.1f} Hz")
        self.cards["ROI STD"].setText(
            f"{s['roi_std_before']:.1f} → {s['roi_std_after']:.1f} Hz")
        self.cards["全脑 RMSE"].setText(
            f"{s['brain_rmse_before']:.1f} → {s['brain_rmse_after']:.1f} Hz")

        self.cards["耗时"].setText(f"{result.total_seconds:.1f} s")
        for k in ("ROI RMSE", "ROI STD", "全脑 RMSE"):
            before, after = self.cards[k].text().split(" → ")
            better = float(after.split()[0]) < float(before)
            colour = self.p["success"] if better else self.p["danger"]
            self.cards[k].setStyleSheet(
                f"font-size: 16px; font-weight: 600; color: {colour};")

        dynamic = getattr(result, "mode", "static") == "dynamic"
        self.dyn_box.setVisible(dynamic)
        self.canvas.setVisible(not dynamic)
        self.view_mode.setEnabled(not dynamic)
        self.slice_at.setEnabled(not dynamic)
        if dynamic:
            n = result.slice_currents.shape[0]
            self.slice_slider.blockSignals(True)
            self.slice_slider.setRange(0, n - 1)
            solved = np.where(result.slice_solved)[0]
            self.slice_slider.setValue(int(solved[len(solved) // 2]) if len(solved)
                                       else n // 2)
            self.slice_slider.blockSignals(False)
            self._draw_heatmap()
            self._draw_slice_card()

        note = f"{result.subject} · {result.coil_name} · {result.roi_name}"
        if dynamic:
            note = (f"{result.subject} · {result.coil_name} · 动态 · "
                    f"{int(result.slice_solved.sum())}/{result.slice_currents.shape[0]} 层")
        if result.channels_at_limit:
            note += f" · ⚠ {result.channels_at_limit} 个通道触及电流上限"
        self.result_note.setText(note)
        if result.report_path:
            self.log.append(f"[记录] {result.report_path}")

        self._draw()
        self.results_box.setVisible(True)

    # ── plots ─────────────────────────────────────────────────────────
    def _draw(self):
        r = self.result
        if r is None or getattr(r, "mode", "static") == "dynamic":
            return
        p = self.p
        self.fig.clear()
        self.fig.set_facecolor(p["fig_bg"])
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.28,
                                   wspace=0.06, left=0.03, right=0.97,
                                   top=0.92, bottom=0.12)

        roi = r.roi_mask
        brain = r.brain_mask if r.brain_mask is not None else (r.b0 != 0)
        mode = self.view_mode.currentData()
        # Scale on the ROI: that is the region being optimised, and letting the
        # whole brain set the window would flatten the part that matters.
        vals = r.b0[roi]
        lim = float(max(abs(np.percentile(vals, 2)), abs(np.percentile(vals, 98)))) or 1.0
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        cmap = mcm.get_cmap("jet").copy()
        cmap.set_bad(color=p["fig_bg"])

        # Centring on the volume can miss the ROI entirely -- the coronal
        # mid-slice of this data contains zero PFC voxels, which reads as a
        # failed registration when it is only a badly chosen slice.
        if self.slice_at.currentData() == "roi" and roi.any():
            idx = [int(round(v)) for v in np.argwhere(roi).mean(0)]
        else:
            idx = [n // 2 for n in r.b0.shape]

        show = brain if mode in ("brain", "highlight") else roi

        def cut(vol, axis):
            sl = [slice(None)] * 3
            sl[axis] = idx[axis]
            out = np.asarray(vol[tuple(sl)]).T[::-1].astype(float)
            mk = np.asarray(show[tuple(sl)]).T[::-1]
            out[~mk] = np.nan
            return out

        def roi_slice(axis):
            sl = [slice(None)] * 3
            sl[axis] = idx[axis]
            return np.asarray(roi[tuple(sl)]).T[::-1]

        im = None
        for col, (name, axis) in enumerate([("Sag", 0), ("Cor", 1), ("Axi", 2)]):
            ax = self.fig.add_subplot(gs[0, col])
            half = np.concatenate([cut(r.b0, axis), cut(r.residual, axis)], axis=1)
            im = ax.imshow(half, cmap=cmap, norm=norm, interpolation="nearest")
            if mode == "highlight":
                rs = roi_slice(axis)
                if rs.any():
                    both = np.concatenate([rs, rs], axis=1).astype(float)
                    ax.contour(both, levels=[0.5], colors=["#000000"],
                               linewidths=1.4)
            ax.axis("off")
            n_roi = int(roi_slice(axis).sum())
            ax.set_title(f"{name} {idx[axis]}   before | after"
                         + (f"   ROI {n_roi} px" if mode == "highlight" else ""),
                         color=p["text_2"], fontsize=9)
        if im is not None:
            cb = self.fig.colorbar(im, ax=self.fig.axes[:3], shrink=0.82,
                                   pad=0.01, aspect=28)
            cb.ax.tick_params(colors=p["text_2"], labelsize=8)
            cb.set_label("Hz", color=p["text_2"], fontsize=9)

        ax = self.fig.add_subplot(gs[1, :2])
        style_axes(ax, p)
        colours = [p["series_pos"] if v >= 0 else p["series_neg"] for v in r.currents]
        ax.bar(range(len(r.currents)), r.currents, color=colours, width=0.62)
        limit = max(abs(r.currents).max(), 0.1) * 1.25
        ax.set_xticks(range(len(r.channel_names)))
        ax.set_xticklabels(r.channel_names, rotation=45, ha="right",
                           fontsize=7, color=p["text_2"])
        ax.set_ylabel("Current (A)", color=p["text_2"], fontsize=9)
        ax.set_ylim(-limit, limit)
        ax.axhline(0, color=p["border"], linewidth=0.8)

        ax = self.fig.add_subplot(gs[1, 2])
        style_axes(ax, p)
        ax.hist(r.b0[roi].ravel(), bins=60, alpha=0.65,
                color=p["series_neg"], label="before")
        ax.hist(r.residual[roi].ravel(), bins=60, alpha=0.65,
                color=p["series_pos"], label="after")
        ax.set_xlabel("B0 (Hz)", color=p["text_2"], fontsize=9)
        ax.legend(fontsize=8, facecolor=p["surface"], edgecolor=p["border"],
                  labelcolor=p["text_2"])
        self.canvas.draw()

    # ── actions ───────────────────────────────────────────────────────
    def _send_to_teensy(self):
        """Hand the currents to the amplifier page and go there.

        A handoff rather than an upload: the serial port, the board map and the
        verify step all live on the Teensy page, which is maintained separately.
        Duplicating any of it here would mean two places to fix when the
        hardware changes.
        """
        if self.result is None:
            return
        from core import amp_bridge

        coil = get_coil(self.coil_combo.currentData())
        table = getattr(self.result, "slice_currents", None)
        amp_bridge.publish(
            self.app_state,
            currents=self.result.currents,
            channel_names=self.result.channel_names,
            coil_name=self.result.coil_name,
            current_limit_a=getattr(coil, "current_limit_a", 0.0),
            source="one_click_shim",
            report_path=self.result.report_path,
            slice_currents=table)
        if table is not None:
            self.log.append(
                f"[放大器] 电流表已发送（{len(table)} 层 × {len(table[0])} 通道）")
        else:
            self.log.append("[放大器] 电流已发送到「Teensy 上传」页")
        self.request_page.emit(_NAV_TEENSY)

    def _export(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", f"{self.result.subject}_shim.json", "JSON (*.json)")
        if not path:
            return
        import json
        r = self.result
        data = {
            "subject": r.subject,
            "coil": r.coil_name,
            "roi": r.roi_name,
            "currents_a": {n: float(v) for n, v in zip(r.channel_names, r.currents)},
            "channels_at_limit": r.channels_at_limit,
            "stats_hz": r.stats,
            "seconds": r.seconds,
            "files": {k: str(v) for k, v in r.files.items()},
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False),
                              encoding="utf-8")
        self.log.append(f"[导出] {path}")

    # ── theme ─────────────────────────────────────────────────────────
    def apply_theme(self, dark: bool, p: dict):
        self.p = p
        self.fig.set_facecolor(p["fig_bg"])
        for k, lbl in self.cards.items():
            if lbl.text() == "—":
                lbl.setStyleSheet(
                    f"font-size: 16px; font-weight: 600; color: {p['text']};")
        if self.result is not None:
            self._draw()
        else:
            self.canvas.draw()
