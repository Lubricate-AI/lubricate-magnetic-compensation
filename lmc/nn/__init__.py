"""Neural network modules for aeromagnetic compensation (optional, research-grade)."""

from lmc.nn.supervised import (
    NNCalibrationResult,
    NNConfig,
    calibrate_nn,
    compensate_nn,
    predict_nn,
)

__all__ = [
    "NNConfig",
    "NNCalibrationResult",
    "calibrate_nn",
    "predict_nn",
    "compensate_nn",
]
