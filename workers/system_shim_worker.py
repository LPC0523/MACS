"""Background worker for system-shim optimisation, with or without a coil."""
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.hybrid_shim import joint_optimize


class SystemShimWorker(QThread):
    """Joint spherical-harmonic + local-coil solve, off the UI thread."""

    progress = Signal(str)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, b0, basis, brain_mask, affine, shape,
                 coil_maps, coil_names, pfc_imax, lam_sh, lam_pfc, parent=None,
                 sh_terms=None, sh_bounds=None,
                 scanner_baseline=None, scanner_limits=None,
                 roi_mask=None, lambda_rest=1.0):
        super().__init__(parent)
        self.b0 = b0
        self.basis = basis
        self.brain_mask = brain_mask
        self.affine = affine
        self.shape = shape
        # Already on this subject's grid, straight from the one-click run.
        self.coil_maps = coil_maps
        self.coil_names = list(coil_names or [])
        self.pfc_imax = pfc_imax
        self.lam_sh = lam_sh
        self.lam_pfc = lam_pfc
        # Which system harmonics may be retuned, and by how much. None keeps
        # the original 2nd-order-only behaviour.
        self.sh_terms = sh_terms
        self.sh_bounds = sh_bounds
        self.scanner_baseline = scanner_baseline
        self.scanner_limits = scanner_limits
        # Target region for the objective's weighting. None = whole brain, one
        # vote per voxel, which lets a 5.7% region lose its own array.
        self.roi_mask = roi_mask
        self.lambda_rest = lambda_rest

    def run(self):
        try:
            import numpy as _np

            if self.coil_maps is not None and len(self.coil_names):
                pfc_names = list(self.coil_names)
                n_pfc = len(pfc_names)
                pfc_array = _np.asarray(self.coil_maps, dtype=_np.float64)
                self.progress.emit(f"线圈场图：{n_pfc} 通道（复用一键匀场的结果）")
            else:
                # SH-only: an empty coil block rather than a second solver, so
                # the objective and the bounds handling are the same code in
                # both modes.
                pfc_names = []
                n_pfc = 0
                pfc_array = _np.zeros((*self.shape, 0), dtype=_np.float64)
                self.progress.emit("仅系统 SH 模式：不使用局部线圈")

            # Whole-brain RMSE, with no region held back. `joint_optimize`
            # subtracts the temporal mask from its target region and penalises
            # degradation there; passing an empty one turns both off, which is
            # what "不保护颞叶" means -- rather than a second code path.
            temporal = _np.zeros_like(self.brain_mask)

            (delta_sh, i_pfc, residual, stats, res,
             scanner_delta, local_sh_equiv) = joint_optimize(
                self.b0, self.basis, self.brain_mask, temporal,
                pfc_array, n_pfc, self.pfc_imax,
                0.0, self.lam_sh, self.lam_pfc,
                log_cb=self.progress.emit,
                sh_terms=self.sh_terms, sh_bounds=self.sh_bounds,
                scanner_baseline=self.scanner_baseline,
                scanner_limits=self.scanner_limits,
                roi_mask=self.roi_mask, lambda_rest=self.lambda_rest)

            for k in sorted(stats):
                if not k.startswith("_"):
                    self.progress.emit(f"  {k}: {stats[k]:.1f}")

            self.finished.emit({
                "sh_terms": list(self.sh_terms) if self.sh_terms else None,
                "delta_sh": delta_sh,
                # What the scanner is told, after removing the component the
                # coils already produce -- see hybrid_shim.local_sh_equivalent.
                "scanner_delta": scanner_delta,
                "local_sh_equiv": local_sh_equiv,
                "i_pfc": i_pfc,
                "residual": residual,
                "stats": stats,
                "pfc_names": pfc_names,
                "pfc_imax": self.pfc_imax,
                "success": bool(res.success),
                "nit": int(res.nit),
            })
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
