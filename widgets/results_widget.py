"""Results Widget — Before/After analysis, histogram, current chart, export."""
import os
import json
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QSlider, QComboBox,
    QFileDialog, QFrame,
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.theme import palette, style_axes


class ResultsWidget(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.p = palette(False)
        #: Set once the user chooses a stage themselves; until then the view
        #: follows whatever finished most recently.
        self._stage_pinned = False
        self._last_known_stage = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("Results & Analysis")
        title.setObjectName("pageTitle")
        layout.addWidget(title)

        # Which stage is being compared against the unshimmed baseline. Hidden
        # while only one stage exists, so a single-stage run looks exactly as
        # it did before.
        layout.addWidget(self._build_stage_row())

        # Metric cards row
        self.cards_layout = QHBoxLayout()
        self.card_labels = {}
        for key in ["ROI RMSE", "ROI STD", "BRAIN RMSE", "BRAIN STD"]:
            card = self._make_card(key)
            self.cards_layout.addWidget(card)
        layout.addLayout(self.cards_layout)

        # Slice controls
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Sag X:"))
        self.sl_x = QSlider(Qt.Horizontal)
        self.sl_x.valueChanged.connect(self._update_viewer)
        ctrl_row.addWidget(self.sl_x)
        ctrl_row.addWidget(QLabel("Cor Y:"))
        self.sl_y = QSlider(Qt.Horizontal)
        self.sl_y.valueChanged.connect(self._update_viewer)
        ctrl_row.addWidget(self.sl_y)
        ctrl_row.addWidget(QLabel("Axi Z:"))
        self.sl_z = QSlider(Qt.Horizontal)
        self.sl_z.valueChanged.connect(self._update_viewer)
        ctrl_row.addWidget(self.sl_z)
        ctrl_row.addWidget(QLabel("Cmap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["jet", "seismic", "bwr", "coolwarm", "RdBu_r", "viridis"])
        self.cmap_combo.currentTextChanged.connect(self._update_viewer)
        ctrl_row.addWidget(self.cmap_combo)
        layout.addLayout(ctrl_row)

        # Before/After canvas
        self.fig_ba = Figure(figsize=(10, 5.5), facecolor=self.p["fig_bg"])
        self.canvas_ba = FigureCanvas(self.fig_ba)
        self.canvas_ba.setMinimumHeight(350)
        layout.addWidget(self.canvas_ba)

        # Histogram + Current chart
        bottom = QHBoxLayout()
        self.fig_hist = Figure(figsize=(5, 3), facecolor=self.p["fig_bg"])
        self.canvas_hist = FigureCanvas(self.fig_hist)
        bottom.addWidget(self.canvas_hist)

        self.fig_curr = Figure(figsize=(5, 3), facecolor=self.p["fig_bg"])
        self.canvas_curr = FigureCanvas(self.fig_curr)
        bottom.addWidget(self.canvas_curr)
        layout.addLayout(bottom)

        # Export buttons
        export_row = QHBoxLayout()
        btn_json = QPushButton("Export JSON")
        btn_json.clicked.connect(lambda: self._export("json"))
        btn_txt = QPushButton("Export TXT")
        btn_txt.clicked.connect(lambda: self._export("txt"))
        btn_mat = QPushButton("Export .mat")
        btn_mat.clicked.connect(lambda: self._export("mat"))
        export_row.addWidget(btn_json)
        export_row.addWidget(btn_txt)
        export_row.addWidget(btn_mat)
        export_row.addStretch()
        layout.addLayout(export_row)

        # Placeholder
        self.placeholder = QLabel("请先在「一键匀场」页面完成优化")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setObjectName("hint")
        self.placeholder.setStyleSheet("font-size: 15px; padding: 40px;")
        layout.addWidget(self.placeholder)

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
        val.setStyleSheet(f"color: {self.p['text']}; font-size: 16px; font-weight: 600;")
        vl.addWidget(val)
        self.card_labels[label] = val
        return frame

    def _build_stage_row(self):
        from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget

        row = QHBoxLayout()
        self.stage_label = QLabel("显示阶段")
        row.addWidget(self.stage_label)
        self.stage_combo = QComboBox()

        def _picked(*_):
            self._stage_pinned = True
            self._refresh()

        self.stage_combo.currentIndexChanged.connect(_picked)
        row.addWidget(self.stage_combo)
        self.stage_summary = QLabel("")
        self.stage_summary.setWordWrap(True)
        row.addWidget(self.stage_summary, 2)
        w = QWidget()
        w.setLayout(row)
        return w

    def showEvent(self, event):
        """Refresh when page becomes visible."""
        super().showEvent(event)
        self._refresh()

    def available_stages(self):
        """(key, label, volume, stats) for every stage that has been run.

        Shimming here is cumulative: the one-click solve and the system shim
        are two stages over the same baseline, and both are worth comparing
        against it and against each other. Whichever have been run are listed;
        none is discarded when the next one arrives.
        """
        out = []
        one = self.app_state.get("b0_shimmed_vol")
        if one is not None and self.app_state.get("shim_stats"):
            out.append(("one_click", "一键匀场", one,
                        self.app_state.get("shim_stats")))
        hyb = self.app_state.get("b0_hybrid_vol")
        if hyb is not None:
            out.append(("hybrid", "系统匀场（SH + 线圈）", hyb,
                        self.app_state.get("hybrid_stats") or {}))
        return out

    def _sync_stage_selector(self, stages):
        """Keep the selector in step without losing the user's choice."""
        keys = [s[0] for s in stages]
        current = self.stage_combo.currentData()
        # A newly finished stage is what the user just asked for, so it becomes
        # the one on screen -- unless they had picked a stage themselves, in
        # which case moving the view out from under them would be worse.
        new_stage = bool(keys) and keys[-1] != self._last_known_stage
        self.stage_combo.blockSignals(True)
        self.stage_combo.clear()
        for key, label, _, _ in stages:
            self.stage_combo.addItem(label, key)
        if current in keys and self._stage_pinned and not new_stage:
            self.stage_combo.setCurrentIndex(keys.index(current))
        else:
            self.stage_combo.setCurrentIndex(len(keys) - 1)   # newest stage
            if new_stage:
                self._stage_pinned = False
        self.stage_combo.blockSignals(False)
        self._last_known_stage = keys[-1] if keys else None
        self.stage_combo.setVisible(len(stages) > 1)
        self.stage_label.setVisible(len(stages) > 1)
        return self.stage_combo.currentData()

    def _refresh(self):
        vol3d = self.app_state.get("b0_vol3d")
        w_opt = self.app_state.get("optimal_currents")
        stages = self.available_stages()

        has_data = vol3d is not None and w_opt is not None and bool(stages)
        if not has_data:
            self.placeholder.setText("请先在「一键匀场」或「系统匀场」页面完成优化")
        self.placeholder.setVisible(not has_data)
        if not has_data:
            return

        key = self._sync_stage_selector(stages)
        chosen = next((s for s in stages if s[0] == key), stages[-1])
        B0_shim, stats = chosen[2], chosen[3]
        self.stage_summary.setText(
            "对比：未匀场  →  %s      （已完成 %d 个阶段：%s）"
            % (chosen[1], len(stages), "、".join(s[1] for s in stages)))

        # Update cards
        def _update_card(key, before, after):
            delta = after - before
            arrow = "▼" if delta < 0 else "▲"
            clr = self.p["success"] if delta < 0 else self.p["danger"]
            self.card_labels[key].setText(f"{before:.1f} → {after:.1f}")
            self.card_labels[key].setStyleSheet(f"color: {clr}; font-size: 16px; font-weight: 600;")

        # A stage whose statistics could not be computed shows a dash rather
        # than taking the page down with a KeyError.
        for label, kb, ka in (("ROI RMSE", "roi_rmse_before", "roi_rmse_after"),
                              ("ROI STD", "roi_std_before", "roi_std_after"),
                              ("BRAIN RMSE", "brain_rmse_before",
                               "brain_rmse_after"),
                              ("BRAIN STD", "brain_std_before",
                               "brain_std_after")):
            if kb in stats and ka in stats:
                _update_card(label, stats[kb], stats[ka])
            else:
                self.card_labels[label].setText("—")

        # Setup sliders
        nx, ny, nz = vol3d.shape
        self.sl_x.setRange(0, nx - 1)
        self.sl_x.setValue(nx // 2)
        self.sl_y.setRange(0, ny - 1)
        self.sl_y.setValue(ny // 2)
        self.sl_z.setRange(0, nz - 1)
        self.sl_z.setValue(nz // 2)

        self._update_viewer()
        self._draw_histogram()
        self._draw_currents()

    def _update_viewer(self):
        vol3d = self.app_state.get("b0_vol3d")
        B0_shim = self.app_state.get("b0_shimmed_vol")
        if vol3d is None or B0_shim is None:
            return

        sx = self.sl_x.value()
        sy = self.sl_y.value()
        sz = self.sl_z.value()
        cmap_name = self.cmap_combo.currentText()

        bm3d = vol3d != 0
        nz_vals = vol3d[bm3d]
        if len(nz_vals) == 0:
            return

        vmin = float(np.percentile(nz_vals, 2))
        vmax = float(np.percentile(nz_vals, 98))
        norm = (TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
                if vmin < 0 < vmax else Normalize(vmin=vmin, vmax=vmax))
        cmap = mcm.get_cmap(cmap_name).copy()
        cmap.set_bad(color=self.p["fig_bg"])

        def masked_slice(vol, axis, idx):
            if axis == 0:
                sl, mk = vol[idx, :, :].T[::-1], bm3d[idx, :, :].T[::-1]
            elif axis == 1:
                sl, mk = vol[:, idx, :].T[::-1], bm3d[:, idx, :].T[::-1]
            else:
                sl, mk = vol[:, :, idx].T[::-1], bm3d[:, :, idx].T[::-1]
            out = sl.astype(float).copy()
            out[mk == 0] = np.nan
            return out

        self.fig_ba.clear()
        axes = self.fig_ba.subplots(2, 3)
        self.fig_ba.subplots_adjust(hspace=0.08, wspace=0.03, left=0.01, right=0.92, top=0.95, bottom=0.02)

        for row, (vol, label) in enumerate([(vol3d, "Before"), (B0_shim, "After")]):
            for col, (ax_idx, title) in enumerate([
                (0, f"Sag {sx}"), (1, f"Cor {sy}"), (2, f"Axi {sz}")
            ]):
                sl = masked_slice(vol, col, [sx, sy, sz][col])
                axes[row, col].imshow(sl, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                axes[row, col].set_facecolor(self.p["fig_bg"])
                axes[row, col].set_title(f"{label} {title}", color=self.p["text_2"], fontsize=9)
                axes[row, col].axis("off")

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cb = self.fig_ba.colorbar(sm, ax=axes[:, :], shrink=0.75, pad=0.01, aspect=30)
        cb.ax.tick_params(colors=self.p["text_2"], labelsize=8)
        cb.set_label("Hz", color=self.p["text_2"], fontsize=9)
        self.canvas_ba.draw()

    def _draw_histogram(self):
        vol3d = self.app_state.get("b0_vol3d")
        B0_shim = self.app_state.get("b0_shimmed_vol")
        if vol3d is None or B0_shim is None:
            return

        bm3d = vol3d != 0
        self.fig_hist.clear()
        ax = self.fig_hist.add_subplot(111)
        style_axes(ax, self.p)
        ax.hist(vol3d[bm3d].ravel(), bins=80, alpha=0.6, color=self.p["series_neg"], label="Before")
        ax.hist(B0_shim[bm3d].ravel(), bins=80, alpha=0.6, color=self.p["series_pos"], label="After")
        ax.set_xlabel("B0 (Hz)", color=self.p["text_2"], fontsize=9)
        ax.set_ylabel("Voxels", color=self.p["text_2"], fontsize=9)
        ax.legend(fontsize=8, facecolor=self.p["surface"],
                  edgecolor=self.p["border"], labelcolor=self.p["text_2"])
        ax.set_title("B0 Distribution", color=self.p["text"], fontsize=10)
        self.fig_hist.tight_layout()
        self.canvas_hist.draw()

    def _draw_currents(self):
        w_opt = self.app_state.get("optimal_currents")
        names = self.app_state.get("ch_names_active", [])
        if w_opt is None:
            return

        if not names:
            names = [f"ch{i+1}" for i in range(len(w_opt))]

        self.fig_curr.clear()
        ax = self.fig_curr.add_subplot(111)
        style_axes(ax, self.p)

        colors = [self.p["series_pos"] if v >= 0 else self.p["series_neg"] for v in w_opt]
        ax.bar(range(len(w_opt)), w_opt, color=colors, width=0.7)
        ax.axhline(3.0, color=self.p["danger"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.axhline(-3.0, color=self.p["danger"], linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=self.p["text_2"])
        ax.set_ylabel("Current (A)", color=self.p["text_2"], fontsize=9)
        ax.set_title("Optimal Currents", color=self.p["text"], fontsize=10)
        ax.set_ylim(-3.5, 3.5)
        self.fig_curr.tight_layout()
        self.canvas_curr.draw()

    def _export(self, fmt):
        w_opt = self.app_state.get("optimal_currents")
        names = self.app_state.get("ch_names_active", [])
        if w_opt is None:
            return
        if not names:
            names = [f"ch{i+1}" for i in range(len(w_opt))]

        if fmt == "json":
            path, _ = QFileDialog.getSaveFileName(self, "Export JSON", "shim_currents.json", "JSON (*.json)")
            if path:
                data = {n: float(v) for n, v in zip(names, w_opt)}
                with open(path, "w") as f:
                    json.dump(data, f, indent=2)
        elif fmt == "txt":
            path, _ = QFileDialog.getSaveFileName(self, "Export TXT", "shim_currents.txt", "Text (*.txt)")
            if path:
                with open(path, "w") as f:
                    f.write("\n".join(f"{v:.6f}" for v in w_opt))
        elif fmt == "mat":
            path, _ = QFileDialog.getSaveFileName(self, "Export MAT", "shim_results.mat", "MAT (*.mat)")
            if path:
                from scipy.io import savemat
                vol3d = self.app_state.get("b0_vol3d")
                B0_shim = self.app_state.get("b0_shimmed_vol")
                savemat(path, {"currents": w_opt, "channel_names": names,
                               "b0_before": vol3d, "b0_after": B0_shim})

    def apply_theme(self, dark: bool, p: dict):
        self.p = p
        for fig in (self.fig_ba, self.fig_hist, self.fig_curr):
            fig.set_facecolor(p["fig_bg"])
        for val in self.card_labels.values():
            if val.text() == "—":
                val.setStyleSheet(f"color: {p['text']}; font-size: 16px; font-weight: 600;")
        self._refresh()
