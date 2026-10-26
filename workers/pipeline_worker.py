"""Background worker for DICOM → B0 pipeline."""
from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """Run DICOM→deltaB0 pipeline in background thread."""

    progress = Signal(str)
    finished = Signal(dict)  # pipeline results
    error = Signal(str)

    def __init__(self, dicom_dir, nifti_dir, delta_te, larmor_freq,
                 bet_f=0.1, run_coreg=True, parent=None):
        super().__init__(parent)
        self.dicom_dir = dicom_dir
        self.nifti_dir = nifti_dir
        self.delta_te = delta_te
        self.larmor_freq = larmor_freq
        self.bet_f = bet_f
        self.run_coreg = run_coreg

    def run(self):
        try:
            from core.B0_cal_algorithm import (
                process_dicom_to_deltaB0, get_reference_file, corigistrator)
            import os

            os.makedirs(self.nifti_dir, exist_ok=True)

            results = process_dicom_to_deltaB0(
                self.dicom_dir, self.nifti_dir,
                self.delta_te, self.larmor_freq,
                bet_f=self.bet_f,
                log_cb=lambda msg: self.progress.emit(str(msg)))

            if self.run_coreg:
                self.progress.emit("\n── Co-registration ──")
                corigistrator(
                    self.nifti_dir, get_reference_file(self.nifti_dir),
                    log_cb=lambda msg: self.progress.emit(str(msg)))

            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))
