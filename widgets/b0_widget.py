"""B0 Map Processing Widget — Load NIfTI/MAT + DICOM pipeline + slice viewer.

Layout priority: the tri-plane viewer gets maximum vertical space.
"""
import os
import glob as _glob
import numpy as np
import nibabel as nib

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QPushButton, QFileDialog, QGroupBox,
    QComboBox, QSlider, QDoubleSpinBox, QFrame,
    QLineEdit, QTextEdit, QCheckBox, QMessageBox,
    QGridLayout, QSplitter, QSizePolicy,
)
from PySide6.QtCore import Qt, Slot

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as mcm
from matplotlib.colors import Normalize, TwoSlopeNorm
import matplotlib.pyplot as plt

from core.physics import load_b0_mat, load_b0_nifti
from workers.pipeline_worker import PipelineWorker


from core.theme import palette


def _status(label, kind):
    """Tag a status label so the global stylesheet colours it."""
    label.setObjectName({"ok": "statusOk", "err": "statusErr",
                         "warn": "statusWarn"}.get(kind, "hint"))
    label.style().unpolish(label)
    label.style().polish(label)


class B0Widget(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = palette(False)
        self.pipeline_worker = None
        self._build_ui()

    # ══════════════════════════════════════════════════════════════════
    #  Build UI — viewer-first layout
    # ══════════════════════════════════════════════════════════════════
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 8)
        layout.setSpacing(8)

        # ── Header: title + tabs in one row ──
        header = QHBoxLayout()
        title = QLabel("B0 Map Processing")
        title.setObjectName("pageTitle")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # ── Tabs (compact) ──
        self.tabs = QTabWidget()
        self.tabs.setMinimumHeight(300)
        self.tabs.addTab(self._build_load_tab(), "Load B0 Map")
        self.tabs.addTab(self._build_pipeline_tab(), "DICOM Pipeline")
        layout.addWidget(self.tabs)

        # ── Metric cards ──
        self.cards_row = QHBoxLayout()
        self.cards_row.setSpacing(8)
        self.metric_labels = {}
        for key in ["Mean (Hz)", "Std (Hz)", "Peak-Peak", "Voxels", "Shape"]:
            card = self._make_metric_card(key)
            self.cards_row.addWidget(card)
        layout.addLayout(self.cards_row)

        # ── Viewer controls (slider + colormap) ──
        ctrl = QHBoxLayout()
        ctrl.setSpacing(12)
        ctrl.addWidget(QLabel("Sag X:"))
        self.sl_x = QSlider(Qt.Horizontal)
        self.sl_x.valueChanged.connect(self._update_viewer)
        ctrl.addWidget(self.sl_x, 2)
        ctrl.addWidget(QLabel("Cor Y:"))
        self.sl_y = QSlider(Qt.Horizontal)
        self.sl_y.valueChanged.connect(self._update_viewer)
        ctrl.addWidget(self.sl_y, 2)
        ctrl.addWidget(QLabel("Axi Z:"))
        self.sl_z = QSlider(Qt.Horizontal)
        self.sl_z.valueChanged.connect(self._update_viewer)
        ctrl.addWidget(self.sl_z, 2)
        ctrl.addWidget(QLabel("Cmap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["jet", "hot", "viridis", "plasma", "inferno",
                                   "turbo", "RdBu_r", "seismic", "gray"])
        self.cmap_combo.currentTextChanged.connect(self._update_viewer)
        ctrl.addWidget(self.cmap_combo)
        layout.addLayout(ctrl)

        # ── Matplotlib canvas — takes ALL remaining vertical space ──
        self.fig = Figure(facecolor=self.p["fig_bg"])
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(350)
        layout.addWidget(self.canvas, stretch=1)   # stretch=1 makes it fill

        # Initial placeholder
        self._draw_placeholder()

    def _make_metric_card(self, label):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setMinimumWidth(110)
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(8, 4, 8, 4)
        vl.setSpacing(2)
        lbl = QLabel(label)
        lbl.setObjectName("hint")
        vl.addWidget(lbl)
        val = QLabel("—")
        val.setStyleSheet("font-size: 15px; font-weight: bold;")
        vl.addWidget(val)
        self.metric_labels[label] = val
        return frame

    # ── Load tab ──
    def _build_load_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 4)

        # File row
        row = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("B0 NIfTI file path (.nii / .nii.gz) or .mat")
        self.path_input.returnPressed.connect(self._load_b0)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._browse_b0)
        btn_load = QPushButton("Load")
        btn_load.setObjectName("primaryBtn")
        btn_load.clicked.connect(self._load_b0)
        row.addWidget(self.path_input, stretch=1)
        row.addWidget(btn_browse)
        row.addWidget(btn_load)
        layout.addLayout(row)

        # Quick-load from pipeline
        self.quickload_row = QHBoxLayout()
        self.quickload_combo = QComboBox()
        self.quickload_combo.setVisible(False)
        btn_quickload = QPushButton("Quick Load")
        btn_quickload.setVisible(False)
        btn_quickload.clicked.connect(self._quick_load_pipeline)
        self.quickload_row.addWidget(QLabel("Pipeline output:"))
        self.quickload_row.addWidget(self.quickload_combo, stretch=1)
        self.quickload_row.addWidget(btn_quickload)
        self.btn_quickload = btn_quickload
        layout.addLayout(self.quickload_row)

        self.load_status = QLabel("")
        layout.addWidget(self.load_status)
        return w

    # ── Pipeline tab ──
    def _build_pipeline_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(8, 8, 8, 4)

        # Parameters row
        pr = QHBoxLayout()
        pr.addWidget(QLabel("Field (T):"))
        self.field_spin = QDoubleSpinBox()
        self.field_spin.setRange(0.5, 14.0); self.field_spin.setValue(5.0)
        pr.addWidget(self.field_spin)
        pr.addWidget(QLabel("ΔTE (s):"))
        self.dte_spin = QDoubleSpinBox()
        # setDecimals FIRST. QDoubleSpinBox defaults to 2 decimals, so calling
        # setValue(0.00246) before it rounds the value to 0.00, which is then
        # clamped up to the 0.0001 minimum -- silently making every B0 map
        # 24.6x too large (std ~1400 Hz instead of ~57 Hz).
        self.dte_spin.setDecimals(5)
        self.dte_spin.setRange(0.0001, 0.1)
        self.dte_spin.setSingleStep(0.0001)
        self.dte_spin.setValue(0.00246)
        self.dte_spin.setToolTip("Echo spacing in SECONDS (e.g. 0.00246 for 2.46 ms)")
        pr.addWidget(self.dte_spin)
        pr.addWidget(QLabel("BET f:"))
        self.betf_spin = QDoubleSpinBox()
        self.betf_spin.setRange(0.01, 0.9); self.betf_spin.setValue(0.1); self.betf_spin.setSingleStep(0.01)
        pr.addWidget(self.betf_spin)
        pr.addStretch()
        layout.addLayout(pr)

        # Folder row
        fr1 = QHBoxLayout()
        fr1.addWidget(QLabel("DICOM:"))
        self.dicom_input = QLineEdit()
        btn_d = QPushButton("...")
        btn_d.clicked.connect(lambda: self._browse_folder(self.dicom_input))
        fr1.addWidget(self.dicom_input, stretch=1); fr1.addWidget(btn_d)
        layout.addLayout(fr1)

        fr2 = QHBoxLayout()
        fr2.addWidget(QLabel("Output:"))
        self.nifti_input = QLineEdit()
        btn_n = QPushButton("...")
        btn_n.clicked.connect(lambda: self._browse_folder(self.nifti_input))
        fr2.addWidget(self.nifti_input, stretch=1); fr2.addWidget(btn_n)
        layout.addLayout(fr2)

        # Run row
        rr = QHBoxLayout()
        self.btn_run_pipeline = QPushButton("Run B0 Pipeline")
        self.btn_run_pipeline.setObjectName("primaryBtn")
        self.btn_run_pipeline.clicked.connect(self._run_pipeline)
        self.chk_coreg = QCheckBox("Co-registration after")
        self.chk_coreg.setChecked(True)
        rr.addWidget(self.btn_run_pipeline); rr.addWidget(self.chk_coreg); rr.addStretch()
        layout.addLayout(rr)

        self.pipeline_log = QTextEdit()
        self.pipeline_log.setReadOnly(True)
        # The pipeline prints ~20 lines per subject; 80 px showed three of them
        # and hid the step that actually failed.
        self.pipeline_log.setMinimumHeight(180)
        self.pipeline_log.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pipeline_log.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(self.pipeline_log, stretch=1)
        return w

    # ══════════════════════════════════════════════════════════════════
    #  Viewer
    # ══════════════════════════════════════════════════════════════════
    def _draw_placeholder(self):
        """Show a hint when no B0 is loaded."""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor(self.p["fig_bg"])
        ax.text(0.5, 0.5, "Load a B0 map to display",
                ha="center", va="center", color=self.p["text_3"], fontsize=15)
        ax.axis("off")
        self.canvas.draw()

    def _setup_viewer(self, vol3d):
        nx, ny, nz = vol3d.shape
        for sl, mx in [(self.sl_x, nx), (self.sl_y, ny), (self.sl_z, nz)]:
            sl.blockSignals(True)
            sl.setRange(0, mx - 1)
            sl.setValue(mx // 2)
            sl.blockSignals(False)
        self._update_viewer()

    def _update_viewer(self):
        vol3d = self.app_state.get("b0_vol3d")
        if vol3d is None:
            return

        sx, sy, sz = self.sl_x.value(), self.sl_y.value(), self.sl_z.value()
        cmap_name = self.cmap_combo.currentText()

        bm3d = vol3d != 0
        nz_vals = vol3d[bm3d]
        if len(nz_vals) == 0:
            return

        # Robust, symmetric window. A few unwrapping outliers at the brain
        # edge used to drag the 98th percentile into the thousands and flatten
        # the whole map to one colour; clipping at |p2|,|p98| and mirroring the
        # scale about zero keeps the field structure visible and makes the two
        # ends of the bar directly comparable.
        p2 = float(np.percentile(nz_vals, 2))
        p98 = float(np.percentile(nz_vals, 98))
        lim = max(abs(p2), abs(p98))
        if not np.isfinite(lim) or lim <= 0:
            lim = float(np.max(np.abs(nz_vals))) or 1.0
        vmin, vmax = -lim, lim
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap = mcm.get_cmap(cmap_name).copy()
        cmap.set_bad(color=self.p["fig_bg"])

        def msl(vol, axis, idx):
            if axis == 0:   sl, mk = vol[idx,:,:].T[::-1], bm3d[idx,:,:].T[::-1]
            elif axis == 1: sl, mk = vol[:,idx,:].T[::-1], bm3d[:,idx,:].T[::-1]
            else:           sl, mk = vol[:,:,idx].T[::-1], bm3d[:,:,idx].T[::-1]
            out = sl.astype(float).copy(); out[mk == 0] = np.nan; return out

        self.fig.clear()
        # 1×4 layout: 3 slices + colorbar
        gs = self.fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.06)
        self.fig.subplots_adjust(left=0.01, right=0.93, top=0.92, bottom=0.02)

        slices = [
            (msl(vol3d, 0, sx), f"Sagittal  X={sx}"),
            (msl(vol3d, 1, sy), f"Coronal  Y={sy}"),
            (msl(vol3d, 2, sz), f"Axial  Z={sz}"),
        ]
        im = None
        for i, (sl, title) in enumerate(slices):
            ax = self.fig.add_subplot(gs[0, i])
            im = ax.imshow(sl, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
            ax.set_facecolor(self.p["fig_bg"])
            ax.set_title(title, color=self.p["text_2"], fontsize=11, pad=6)
            ax.axis("off")

        # Colorbar
        if im is not None:
            cax = self.fig.add_subplot(gs[0, 3])
            cb = self.fig.colorbar(im, cax=cax)
            cb.ax.tick_params(colors=self.p["text_2"], labelsize=9)
            cb.set_label("Hz", color=self.p["text_2"], fontsize=10)

        # State the numbers the picture is built from, and flag a window that
        # is physically implausible for a head B0 map (usually a wrong dTE or
        # a failed unwrap) instead of just drawing it.
        std = float(np.std(nz_vals))
        note = (f"scale ±{lim:.0f} Hz   (p2 {p2:+.0f} / p98 {p98:+.0f}, "
                f"std {std:.0f} Hz, {nz_vals.size:,} vox)")
        colour = self.p["text_2"]
        if lim > 1000 or std > 300:
            note += "   ⚠ unusually wide — check ΔTE units (s) and unwrapping"
            colour = self.p["danger"]
        self.fig.text(0.01, 0.015, note, color=colour, fontsize=8, ha="left")

        self.canvas.draw()

        # Metric cards
        self.metric_labels["Mean (Hz)"].setText(f"{np.mean(nz_vals):.1f}")
        self.metric_labels["Std (Hz)"].setText(f"{np.std(nz_vals):.1f}")
        self.metric_labels["Peak-Peak"].setText(f"{np.ptp(nz_vals):.1f}")
        self.metric_labels["Voxels"].setText(f"{len(nz_vals):,}")
        self.metric_labels["Shape"].setText(f"{vol3d.shape[0]}×{vol3d.shape[1]}×{vol3d.shape[2]}")

    # ══════════════════════════════════════════════════════════════════
    #  Load actions
    # ══════════════════════════════════════════════════════════════════
    def _browse_b0(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select B0 File", "",
            "NIfTI (*.nii *.nii.gz);;MAT (*.mat);;All (*)")
        if path:
            self.path_input.setText(path)
            self._load_b0()

    def _browse_folder(self, line_edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            line_edit.setText(path)

    def _load_b0(self):
        path = self.path_input.text().strip()
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self, "Error", "Please select a valid file")
            return

        try:
            if path.endswith(".mat"):
                coords, b0 = load_b0_mat(path)
                self.app_state.update(
                    b0_data=b0, b0_coords=coords,
                    b0_vol3d=None, b0_affine=None, b0_label=os.path.basename(path))
                self.load_status.setText(f"✓ {os.path.basename(path)} — {len(b0):,} voxels")
                _status(self.load_status, "ok")
                self._draw_placeholder()
            else:
                coords, b0, vol3d, affine = load_b0_nifti(path)
                self.app_state.update(
                    b0_data=b0, b0_coords=coords,
                    b0_vol3d=vol3d, b0_affine=affine,
                    b0_label=os.path.basename(path),
                    subject_dir=os.path.dirname(os.path.abspath(path)))
                self.load_status.setText(
                    f"✓ {os.path.basename(path)} — {len(b0):,} voxels, {vol3d.shape}")
                _status(self.load_status, "ok")
                self._setup_viewer(vol3d)
        except Exception as e:
            self.load_status.setText(f"✗ {e}")
            _status(self.load_status, "err")

    def _quick_load_pipeline(self):
        path = self.quickload_combo.currentData()
        if path and os.path.isfile(path):
            self.path_input.setText(path)
            self._load_b0()

    # ══════════════════════════════════════════════════════════════════
    #  DICOM Pipeline
    # ══════════════════════════════════════════════════════════════════
    def _run_pipeline(self):
        dicom_dir = self.dicom_input.text().strip()
        nifti_dir = self.nifti_input.text().strip()
        if not dicom_dir or not os.path.isdir(dicom_dir):
            QMessageBox.warning(self, "Error", "Please select valid DICOM directory")
            return
        if not nifti_dir:
            QMessageBox.warning(self, "Error", "Please select output directory")
            return

        # deltaB0 = phase / (2*pi*dTE), so dTE scales the entire field map.
        # A wrong value produces a plausible-looking map with every number off
        # by that factor, which is far more dangerous than a crash -- state it
        # up front and refuse the extremes.
        dte = self.dte_spin.value()
        if dte <= 0.0005 or dte >= 0.05:
            if QMessageBox.warning(
                    self, "Check ΔTE",
                    f"ΔTE = {dte:.5f} s ({dte*1000:.2f} ms).\n\n"
                    "Typical dual-echo spacing is 1-10 ms, i.e. 0.001-0.010 s.\n"
                    "This value scales the whole B0 map: too small a ΔTE makes "
                    "every field value proportionally too large.\n\nRun anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No) != QMessageBox.Yes:
                return

        self.pipeline_log.clear()
        self.btn_run_pipeline.setEnabled(False)
        self.btn_run_pipeline.setText("Running...")
        self.pipeline_log.append(
            f"ΔTE = {dte:.5f} s ({dte*1000:.2f} ms)   "
            f"field = {self.field_spin.value():.2f} T   "
            f"Larmor = {42.58 * self.field_spin.value():.2f} MHz")

        self.pipeline_worker = PipelineWorker(
            dicom_dir, nifti_dir,
            dte, 42.58 * self.field_spin.value(),
            bet_f=self.betf_spin.value(),
            run_coreg=self.chk_coreg.isChecked(), parent=self)
        self.pipeline_worker.progress.connect(self._on_pipeline_log)
        self.pipeline_worker.finished.connect(self._on_pipeline_done)
        self.pipeline_worker.error.connect(self._on_pipeline_error)
        self.pipeline_worker.start()

    @Slot(str)
    def _on_pipeline_log(self, msg):
        self.pipeline_log.append(msg)

    @Slot(dict)
    def _on_pipeline_done(self, results):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_run_pipeline.setText("Run B0 Pipeline")
        ok = sum(1 for v in results.values() if v.get("status") == "ok")
        self.pipeline_log.append(f"\n=== Done: {ok} scans processed ===")

        # Populate quick-load combo
        self.quickload_combo.clear()
        for d, info in results.items():
            if info.get("status") == "ok":
                for key in ["deltaB0_hz", "deltaB0_ppm"]:
                    p = info.get("files", {}).get(key, "")
                    if p and os.path.isfile(p):
                        label = f"{d} → {os.path.basename(p)}"
                        self.quickload_combo.addItem(label, p)
        if self.quickload_combo.count() > 0:
            self.quickload_combo.setVisible(True)
            self.btn_quickload.setVisible(True)
            # Auto-load first result
            self.quickload_combo.setCurrentIndex(0)
            self._quick_load_pipeline()

    @Slot(str)
    def _on_pipeline_error(self, msg):
        self.btn_run_pipeline.setEnabled(True)
        self.btn_run_pipeline.setText("Run B0 Pipeline")
        self.pipeline_log.append(f"\n[ERROR] {msg}")

    def apply_theme(self, dark: bool, p: dict):
        self.p = p
        self.fig.set_facecolor(p["fig_bg"])
        vol = self.app_state.get("b0_vol3d")
        if vol is None:
            self._draw_placeholder()
        else:
            self._update_viewer()
