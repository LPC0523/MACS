"""Optimize Widget — One-Click PFC Shimming.

Steps: Load B0 → PFC Mask (registration) → Channel Maps → Optimize
"""

# ─────────────────────────────────────────────────────────────────────────
# LEGACY / NOT USED BY CURRENT QT APP
#
# 未加入 main_window 的页面栈；其优化流程已由一键匀场页取代。
#
# 替代模块：widgets/shim_widget.py + core/pipeline.py
#
# 保留原因：路径与 import 兼容，且其中的实现细节仍有参考价值。修改这里不会影响
# 运行中的程序——先确认调用方是谁再动手。
# ─────────────────────────────────────────────────────────────────────────

import os
import re
import glob as _glob
import numpy as np
import nibabel as nib

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox,
    QLineEdit, QTextEdit, QSlider, QDoubleSpinBox,
    QMessageBox, QProgressBar,
)
from PySide6.QtCore import Qt, Slot

from core.data_context import natural_key, resample_to_target
from core.paths import find_atlases
from workers.optimization_worker import OptimizationWorker


from core.theme import palette


def _status(label, kind):
    """Tag a status label so the global stylesheet colours it."""
    label.setObjectName({"ok": "statusOk", "err": "statusErr",
                         "warn": "statusWarn"}.get(kind, "hint"))
    label.style().unpolish(label)
    label.style().polish(label)


class OptimizeWidget(QWidget):
    def __init__(self, app_state: dict, parent=None):
        super().__init__(parent)
        self.app_state = app_state
        self.opt_worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)

        title = QLabel("一键前额叶匀场")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        layout.addWidget(QLabel("加载 B0 场图 → 配准 PFC mask → 加载通道图 → 一键优化"))

        # ── Step 1: B0 ──
        s1 = QGroupBox("Step 1: B0 目标场图 (Hz)")
        s1l = QHBoxLayout(s1)
        self.b0_path = QLineEdit()
        self.b0_path.setPlaceholderText("B0 Hz NIfTI path (auto-filled from B0 page)")
        btn_b0 = QPushButton("...")
        btn_b0.clicked.connect(self._browse_b0)
        self.b0_status = QLabel("")
        s1l.addWidget(self.b0_path)
        s1l.addWidget(btn_b0)
        s1l.addWidget(self.b0_status)
        layout.addWidget(s1)

        # ── Step 2: PFC Mask ──
        s2 = QGroupBox("Step 2: PFC ROI Mask")
        s2l = QVBoxLayout(s2)
        mask_row = QHBoxLayout()
        self.mask_path = QLineEdit()
        self.mask_path.setPlaceholderText("PFC mask NIfTI (native space or will be registered)")
        btn_mask = QPushButton("...")
        btn_mask.clicked.connect(self._browse_mask)
        btn_reg = QPushButton("一键配准")
        btn_reg.setObjectName("primaryBtn")
        btn_reg.clicked.connect(self._run_registration)
        mask_row.addWidget(self.mask_path)
        mask_row.addWidget(btn_mask)
        mask_row.addWidget(btn_reg)
        s2l.addLayout(mask_row)
        self.mask_status = QLabel("")
        s2l.addWidget(self.mask_status)
        layout.addWidget(s2)

        # ── Step 3: Channel Maps ──
        s3 = QGroupBox("Step 3: 线圈通道场图")
        s3l = QHBoxLayout(s3)
        self.ch_path = QLineEdit()
        self.ch_path.setPlaceholderText("Channel map folder (contains ch1_nii/, ch2_nii/, ...)")
        btn_ch = QPushButton("...")
        btn_ch.clicked.connect(self._browse_ch)
        self.ch_status = QLabel("")
        s3l.addWidget(self.ch_path)
        s3l.addWidget(btn_ch)
        s3l.addWidget(self.ch_status)
        layout.addWidget(s3)

        # ── Step 4: Lambda ──
        s4 = QGroupBox("Step 4: 惩罚系数 λ")
        s4l = QHBoxLayout(s4)
        s4l.addWidget(QLabel("λ = "))
        self.lambda_spin = QDoubleSpinBox()
        self.lambda_spin.setRange(0.0, 2.0)
        self.lambda_spin.setValue(0.5)
        self.lambda_spin.setSingleStep(0.01)
        s4l.addWidget(self.lambda_spin)
        s4l.addWidget(QLabel("(越大=非ROI约束越强)  |  电流约束: ±3.0A  |  L-BFGS-B"))
        s4l.addStretch()
        layout.addWidget(s4)

        # ── Run button ──
        self.btn_optimize = QPushButton("一键优化")
        self.btn_optimize.setObjectName("primaryBtn")
        self.btn_optimize.setMinimumHeight(42)
        self.btn_optimize.clicked.connect(self._run_optimization)
        layout.addWidget(self.btn_optimize)

        # ── Progress / Log ──
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # indeterminate
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(180)
        layout.addWidget(self.log_text)

        # ── Results summary ──
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        layout.addWidget(self.results_label)

        layout.addStretch()

    # ══════════════════════════════════════════════════════════════════════════
    # Browse actions
    # ══════════════════════════════════════════════════════════════════════════

    def _browse_b0(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select B0 Hz NIfTI", "",
                                              "NIfTI (*.nii *.nii.gz)")
        if path:
            self.b0_path.setText(path)
            self._load_b0(path)

    def _browse_mask(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PFC Mask NIfTI", "",
                                              "NIfTI (*.nii *.nii.gz)")
        if path:
            self.mask_path.setText(path)
            self._load_mask(path)

    def _browse_ch(self):
        path = QFileDialog.getExistingDirectory(self, "Select Channel Map Folder")
        if path:
            self.ch_path.setText(path)
            self._check_channels(path)

    # ══════════════════════════════════════════════════════════════════════════
    # Data loading
    # ══════════════════════════════════════════════════════════════════════════

    def _load_b0(self, path):
        try:
            img = nib.load(path)
            data = img.get_fdata(dtype=np.float32)
            self.app_state["b0_vol3d"] = data
            self.app_state["b0_affine"] = img.affine.copy()
            self.app_state["subject_dir"] = os.path.dirname(os.path.abspath(path))
            nz = (data != 0).sum()
            self.b0_status.setText(f"✓ {data.shape} | {nz:,} voxels")
            _status(self.b0_status, "ok")
        except Exception as e:
            self.b0_status.setText(f"✗ {e}")
            _status(self.b0_status, "err")

    def _load_mask(self, path):
        try:
            img = nib.load(path)
            mask_data = img.get_fdata(dtype=np.float32)
            # Resample to B0 grid if needed
            b0_vol = self.app_state.get("b0_vol3d")
            b0_aff = self.app_state.get("b0_affine")
            if b0_vol is not None and mask_data.shape[:3] != b0_vol.shape[:3]:
                mask_data = resample_to_target(
                    mask_data, img.affine, b0_aff, b0_vol.shape[:3], order=0)
            mask_bin = (mask_data > 0.5).astype(np.float32)
            self.app_state["pfc_mask"] = mask_bin
            n_roi = int(mask_bin.sum())
            self.mask_status.setText(f"✓ ROI: {n_roi:,} voxels")
            _status(self.mask_status, "ok")
        except Exception as e:
            self.mask_status.setText(f"✗ {e}")
            _status(self.mask_status, "err")

    def _check_channels(self, folder):
        ch_dirs = sorted(_glob.glob(os.path.join(folder, "ch*_nii")), key=natural_key)
        if ch_dirs:
            self.ch_status.setText(f"✓ {len(ch_dirs)} channels detected")
            _status(self.ch_status, "ok")
        else:
            self.ch_status.setText("✗ No ch*_nii folders found")
            _status(self.ch_status, "err")

    def _run_registration(self):
        """One-click MNI→native PFC mask registration using ANTsPy."""
        self.log_text.clear()
        self.log_text.append("Starting MNI → native registration...")

        # Find atlas files (bundled under atlases/, works frozen and from source)
        mni_path, pfc_path = find_atlases()

        if not mni_path or not pfc_path:
            missing = []
            if not mni_path:
                missing.append("1MNI152_T1_1mm_brain.nii.gz")
            if not pfc_path:
                missing.append("pfc_mask_v3.nii.gz")
            self.log_text.append(
                f"[ERROR] Atlas files not found: {', '.join(missing)}. "
                "Place them in the application's atlases/ folder.")
            return

        # Find magnitude
        subj_dir = self.app_state.get("subject_dir", "")
        if not subj_dir or not os.path.isdir(subj_dir):
            QMessageBox.warning(self, "Error", "请先加载B0（自动定位被试文件夹）")
            return

        mag_path = ""
        exclude = ("mask", "phase", "deltab0", "unwrap", "brain", "pfc", "ppm")
        for pat in ["*Echo2*.nii*", "*Echo1*.nii*", "*_e2.nii*", "*_e1.nii*", "*mag*.nii*"]:
            hits = [f for f in _glob.glob(os.path.join(subj_dir, pat))
                    if not any(ex in os.path.basename(f).lower() for ex in exclude)]
            if hits:
                mag_path = hits[0]
                break

        if not mag_path:
            QMessageBox.warning(self, "Error", f"No magnitude found in {subj_dir}")
            return

        self.log_text.append(f"Magnitude: {os.path.basename(mag_path)}")
        self.log_text.append(f"MNI T1: {os.path.basename(mni_path)}")
        self.log_text.append(f"PFC mask: {os.path.basename(pfc_path)}")

        try:
            import ants
            self.log_text.append("Running ANTsPy registration...")

            # BET: find brain mask
            bet_mask = ""
            for pat in ["*bet*mask*.nii*", "*brain*mask*.nii*", "*_mask.nii*"]:
                hits = _glob.glob(os.path.join(subj_dir, pat))
                if hits:
                    bet_mask = hits[0]
                    break

            # Create mag_brain
            mag_img = nib.load(mag_path)
            mag_data = np.asanyarray(mag_img.dataobj, dtype=np.float32)
            if bet_mask:
                mask_bet = nib.load(bet_mask).get_fdata().astype(np.float32)
                if mask_bet.shape[:3] != mag_data.shape[:3]:
                    mask_bet = resample_to_target(
                        mask_bet, nib.load(bet_mask).affine,
                        mag_img.affine, mag_data.shape[:3], order=0)
                mag_brain = mag_data * (mask_bet > 0.5).astype(np.float32)
            else:
                mag_brain = mag_data

            # Save temp mag_brain
            reg_dir = os.path.join(subj_dir, "pfc_registration")
            os.makedirs(reg_dir, exist_ok=True)
            mag_brain_path = os.path.join(reg_dir, "mag_brain.nii.gz")
            nib.save(nib.Nifti1Image(mag_brain, mag_img.affine), mag_brain_path)

            # Register
            fixed = ants.image_read(mag_brain_path)
            moving = ants.image_read(mni_path)
            mask_mni = ants.image_read(pfc_path)

            reg_result = ants.registration(
                fixed=fixed, moving=moving,
                type_of_transform="Affine",
                aff_metric="mattes", aff_sampling=32,
                aff_iterations=(2100, 1200, 1200, 100),
                aff_shrink_factors=(6, 4, 2, 1),
                aff_smoothing_sigmas=(3, 2, 1, 0),
                verbose=False)

            # Apply to PFC mask
            mask_native = ants.apply_transforms(
                fixed=fixed, moving=mask_mni,
                transformlist=reg_result["fwdtransforms"],
                interpolator="nearestNeighbor")

            mask_path = os.path.join(reg_dir, "pfc_mask_native.nii.gz")
            ants.image_write(mask_native, mask_path)

            # Resample to B0 grid
            mask_data = mask_native.numpy().astype(np.float32)
            b0_vol = self.app_state.get("b0_vol3d")
            b0_aff = self.app_state.get("b0_affine")
            if b0_vol is not None and mask_data.shape[:3] != b0_vol.shape[:3]:
                mask_nib = nib.load(mask_path)
                mask_data = resample_to_target(
                    mask_data, mask_nib.affine, b0_aff, b0_vol.shape[:3], order=0)

            mask_bin = (mask_data > 0.5).astype(np.float32)
            self.app_state["pfc_mask"] = mask_bin
            n_roi = int(mask_bin.sum())

            self.mask_path.setText(mask_path)
            self.mask_status.setText(f"✓ ROI: {n_roi:,} voxels (registered)")
            _status(self.mask_status, "ok")
            self.log_text.append(f"Registration complete! ROI: {n_roi:,} voxels")

        except ImportError:
            self.log_text.append("[ERROR] ANTsPy not installed. Run: pip install antspyx")
        except Exception as e:
            self.log_text.append(f"[ERROR] {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Optimization
    # ══════════════════════════════════════════════════════════════════════════

    def _run_optimization(self):
        b0_vol = self.app_state.get("b0_vol3d")
        b0_aff = self.app_state.get("b0_affine")
        mask = self.app_state.get("pfc_mask")
        ch_folder = self.ch_path.text().strip()

        if b0_vol is None:
            QMessageBox.warning(self, "Missing", "请先加载 B0 场图")
            return
        if mask is None:
            QMessageBox.warning(self, "Missing", "请先加载或配准 PFC mask")
            return
        if not ch_folder or not os.path.isdir(ch_folder):
            QMessageBox.warning(self, "Missing", "请指定通道场图文件夹")
            return

        # Load channels
        self.log_text.clear()
        self.log_text.append("Loading channel maps...")
        try:
            B0_array, ch_names = self._load_channels(
                ch_folder, b0_aff, b0_vol.shape[:3])
            self.log_text.append(f"Loaded {len(ch_names)} channels")
        except Exception as e:
            self.log_text.append(f"[ERROR] {e}")
            return

        # Start optimization worker
        self.btn_optimize.setEnabled(False)
        self.progress.setVisible(True)

        self.opt_worker = OptimizationWorker(
            b0_vol, b0_aff, mask, B0_array, ch_names,
            lambda_penalty=self.lambda_spin.value(), parent=self)
        self.opt_worker.progress.connect(self._on_opt_progress)
        self.opt_worker.finished.connect(self._on_opt_done)
        self.opt_worker.error.connect(self._on_opt_error)
        self.opt_worker.start()

    def _load_channels(self, folder, ref_affine, ref_shape):
        """Load and resample channel maps."""
        entries = sorted(
            [e for e in os.listdir(folder)
             if os.path.isdir(os.path.join(folder, e)) and re.match(r'ch\d+', e, re.I)],
            key=natural_key)
        if not entries:
            raise FileNotFoundError(f"No ch*_nii folders in {folder}")

        B0_list, names = [], []
        for ch_dir_name in entries:
            ch_dir = os.path.join(folder, ch_dir_name)
            ch_label = ch_dir_name.replace("_nii", "")

            resampled = _glob.glob(os.path.join(ch_dir, "*resampled*.nii*"))
            delta = _glob.glob(os.path.join(ch_dir, "*deltaFieldmap.nii*"))
            any_nii = _glob.glob(os.path.join(ch_dir, "*.nii*"))

            load_path = (resampled or delta or any_nii or [None])[0]
            if not load_path:
                continue

            img = nib.load(load_path)
            if img.shape[:3] != tuple(ref_shape):
                data = resample_to_target(
                    img.get_fdata(dtype=np.float32), img.affine, ref_affine, ref_shape, order=1)
                # Save resampled
                out_path = os.path.join(ch_dir, f"{ch_dir_name}_resampled_to_brain.nii.gz")
                nib.save(nib.Nifti1Image(data, ref_affine), out_path)
            else:
                data = img.get_fdata(dtype=np.float32)

            B0_list.append(data)
            names.append(ch_label)
            self.log_text.append(f"  ✓ {ch_label}: {data.shape}")

        if not B0_list:
            raise FileNotFoundError("No channel maps loaded")
        return np.stack(B0_list, axis=-1), names

    @Slot(str)
    def _on_opt_progress(self, msg):
        self.log_text.append(msg)

    @Slot(dict)
    def _on_opt_done(self, result):
        self.btn_optimize.setEnabled(True)
        self.progress.setVisible(False)

        # Store results in app_state
        self.app_state["optimal_currents"] = result["currents"]
        self.app_state["ch_names_active"] = result["ch_names"]
        self.app_state["currents_source"] = "oneclick"
        self.app_state["shim_stats"] = result["stats"]
        self.app_state["b0_shimmed_vol"] = result["residual"]

        stats = result["stats"]
        self.results_label.setText(
            f"优化完成!\n"
            f"ROI RMSE: {stats['roi_rmse_before']:.1f} → {stats['roi_rmse_after']:.1f} Hz\n"
            f"ROI STD: {stats['roi_std_before']:.1f} → {stats['roi_std_after']:.1f} Hz\n"
            f"迭代: {result['n_iter']}  |  Loss: {result['loss']:.3f}")
        _status(self.results_label, "ok")

    @Slot(str)
    def _on_opt_error(self, msg):
        self.btn_optimize.setEnabled(True)
        self.progress.setVisible(False)
        self.log_text.append(f"[ERROR] {msg}")
        QMessageBox.critical(self, "Optimization Failed", msg)

    # ══════════════════════════════════════════════════════════════════════════
    # Auto-sync from app_state (called when switching to this page)
    # ══════════════════════════════════════════════════════════════════════════

    def showEvent(self, event):
        """Sync B0 path from app_state when page becomes visible."""
        super().showEvent(event)
        if self.app_state.get("b0_vol3d") is not None and not self.b0_path.text():
            label = self.app_state.get("b0_label", "")
            self.b0_status.setText(f"✓ {label} (from B0 page)")
            _status(self.b0_status, "ok")

