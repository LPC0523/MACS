"""Background worker for shimming optimization (runs in QThread)."""

# ─────────────────────────────────────────────────────────────────────────
# LEGACY / NOT USED BY CURRENT QT APP
#
# 只被 optimize_widget 使用，随该页面一起停用。
#
# 替代模块：workers/shim_pipeline_worker.py
#
# 保留原因：路径与 import 兼容，且其中的实现细节仍有参考价值。修改这里不会影响
# 运行中的程序——先确认调用方是谁再动手。
# ─────────────────────────────────────────────────────────────────────────

import numpy as np
from scipy.optimize import minimize
from PySide6.QtCore import QThread, Signal


class OptimizationWorker(QThread):
    """Run MSE shimming optimization in background thread."""

    progress = Signal(str)       # Log messages
    finished = Signal(dict)      # Results dict
    error = Signal(str)          # Error message

    def __init__(self, B0_target, target_affine, mask, B0_array, ch_names,
                 lambda_penalty=0.5, parent=None):
        super().__init__(parent)
        self.B0_target = B0_target
        self.target_affine = target_affine
        self.mask = mask
        self.B0_array = B0_array
        self.ch_names = ch_names
        self.lambda_penalty = lambda_penalty

    def run(self):
        try:
            B0_target = self.B0_target
            mask = self.mask
            B0_array = self.B0_array
            n_channels = B0_array.shape[-1]
            lam = self.lambda_penalty

            self.progress.emit(f"Building matrices... {n_channels} channels")

            # Build A matrix & b vector
            b_flat = B0_target.flatten()
            mask_flat = mask.flatten()
            roi_idx = mask_flat > 0.5
            nonroi_idx = mask_flat <= 0.5

            b = b_flat[roi_idx].astype(np.float64)
            A = B0_array.reshape(-1, n_channels)[roi_idx].astype(np.float64)
            A_nonroi = B0_array.reshape(-1, n_channels)[nonroi_idx].astype(np.float64)

            self.progress.emit(f"ROI: {len(b):,} voxels | non-ROI: {A_nonroi.shape[0]:,}")

            # MSE loss function
            def loss_mse(w):
                roi_term = np.mean((A @ w + b) ** 2)
                nonroi_term = np.mean((A_nonroi @ w) ** 2)
                return roi_term + lam * nonroi_term

            # Optimize
            bounds = [(-3.0, 3.0)] * n_channels
            initial_weights = np.ones(n_channels) / n_channels

            self.progress.emit(f"Optimizing MSE (L-BFGS-B, λ={lam})...")
            res = minimize(loss_mse, initial_weights, bounds=bounds, method='L-BFGS-B')
            shim_currents = res.x

            # Evaluate
            B0_comp = np.tensordot(B0_array, shim_currents, axes=([3], [0]))
            residual_map = B0_target + B0_comp

            roi_mask_bool = mask > 0.5
            brain_bool = B0_target != 0

            stats = {
                "roi_rmse_before":   float(np.sqrt(np.mean(B0_target[roi_mask_bool] ** 2))),
                "roi_rmse_after":    float(np.sqrt(np.mean(residual_map[roi_mask_bool] ** 2))),
                "roi_std_before":    float(np.std(B0_target[roi_mask_bool])),
                "roi_std_after":     float(np.std(residual_map[roi_mask_bool])),
                "roi_ptp_before":    float(np.ptp(B0_target[roi_mask_bool])),
                "roi_ptp_after":     float(np.ptp(residual_map[roi_mask_bool])),
                "brain_rmse_before": float(np.sqrt(np.mean(B0_target[brain_bool] ** 2))),
                "brain_rmse_after":  float(np.sqrt(np.mean(residual_map[brain_bool] ** 2))),
                "brain_std_before":  float(np.std(B0_target[brain_bool])),
                "brain_std_after":   float(np.std(residual_map[brain_bool])),
            }

            self.progress.emit(
                f"Done! ROI RMSE: {stats['roi_rmse_before']:.1f} → {stats['roi_rmse_after']:.1f} Hz")

            self.finished.emit({
                "currents": shim_currents,
                "residual": residual_map,
                "stats": stats,
                "ch_names": self.ch_names,
                "loss": float(loss_mse(shim_currents)),
                "n_iter": res.nit,
                "success": res.success,
            })

        except Exception as e:
            self.error.emit(str(e))
