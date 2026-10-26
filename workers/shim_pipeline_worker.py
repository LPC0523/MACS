"""Background worker for the one-shot shimming pipeline."""
from PySide6.QtCore import QThread, Signal

from core.pipeline import run_pipeline, PipelineError


class ShimPipelineWorker(QThread):
    """DICOM folder -> optimised currents, off the UI thread."""

    progress = Signal(str)
    finished = Signal(object)               # PipelineResult
    # (human message, technical detail, error code) -- the code lets the UI
    # show a stable identifier without parsing the prose.
    failed = Signal(str, str, str)

    def __init__(self, dicom_dir, *, output_dir=None, coil=None,
                 roi="whole_brain", custom_mask_path=None, mode="static",
                 delta_te=0.00246, field_t=5.0, auto_params=True,
                 lambda_penalty=0.5, parent=None):
        super().__init__(parent)
        self.dicom_dir = dicom_dir
        self.output_dir = output_dir
        self.coil = coil
        self.roi = roi
        self.custom_mask_path = custom_mask_path
        self.mode = mode
        self.delta_te = delta_te
        self.field_t = field_t
        self.auto_params = auto_params
        self.lambda_penalty = lambda_penalty

    def run(self):
        try:
            result = run_pipeline(
                self.dicom_dir,
                output_dir=self.output_dir,
                coil=self.coil,
                roi=self.roi,
                custom_mask_path=self.custom_mask_path,
                mode=self.mode,
                delta_te=self.delta_te,
                field_t=self.field_t,
                auto_params=self.auto_params,
                lambda_penalty=self.lambda_penalty,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(result)
        except PipelineError as e:
            # Already phrased for the user by the pipeline.
            self.failed.emit(e.message, e.technical, getattr(e, "code", ""))
        except Exception as e:
            from core.errors import classify

            a = classify(e)
            self.failed.emit(a.headline, a.detail, a.code)
