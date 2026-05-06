"""FPR-based model calibration using sklearn IsotonicRegression."""

from fpr_model_calibration.calibration import (
    fit_calibration_pipeline,
    fpr_to_calibrated,
)

__all__ = ["fit_calibration_pipeline", "fpr_to_calibrated"]

__version__ = "0.1.0"
