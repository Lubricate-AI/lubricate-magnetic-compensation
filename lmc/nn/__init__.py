"""Neural network modules for aeromagnetic compensation (optional, research-grade)."""

from lmc.nn.pinn import (
    PINNCalibrationResult,
    PINNConfig,
    calibrate_pinn,
    compensate_pinn,
    predict_pinn,
)
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
    "PINNConfig",
    "PINNCalibrationResult",
    "calibrate_pinn",
    "predict_pinn",
    "compensate_pinn",
]
