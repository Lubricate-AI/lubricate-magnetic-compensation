# tests/unit/nn/test_pinn.py
"""Unit tests for lmc.nn.pinn."""

from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from lmc.calibration import CalibrationResult
from lmc.nn.pinn import PINNCalibrationResult, PINNConfig


def test_pinn_config_defaults() -> None:
    cfg = PINNConfig()
    assert cfg.hidden_layer_sizes == (64, 64)
    assert cfg.activation == "relu"
    assert cfg.max_iter == 500
    assert cfg.n_estimators == 20
    assert cfg.random_state == 42
    assert cfg.physics_lambda == 1e-3
    assert cfg.tl_model_terms == "c"
    assert cfg.nn_feature_terms == "b"


def test_pinn_config_custom() -> None:
    cfg = PINNConfig(hidden_layer_sizes=(32,), n_estimators=5, physics_lambda=0.1)
    assert cfg.hidden_layer_sizes == (32,)
    assert cfg.n_estimators == 5
    assert cfg.physics_lambda == 0.1


def test_pinn_calibration_result_stores_fields() -> None:
    # Build a minimal CalibrationResult
    dummy_coef = np.zeros(3)
    dummy_sv = np.array([1.0, 0.5, 0.1])
    tl = CalibrationResult(
        coefficients=dummy_coef,
        residuals=np.array([0.1, -0.1]),
        condition_number=2.0,
        singular_values=dummy_sv,
        n_terms=3,
    )
    scaler = StandardScaler()
    scaler.fit([[1.0, 2.0]])  # pyright: ignore[reportUnknownMemberType]
    estimators = [MLPRegressor(max_iter=10).fit([[1.0, 2.0]], [0.5])]  # pyright: ignore[reportUnknownMemberType]
    result = PINNCalibrationResult(
        tl_result=tl,
        estimators=estimators,
        input_scaler=scaler,
        tl_residuals=np.array([0.2, -0.3]),
        pinn_residuals=np.array([0.05, -0.04]),
        n_nn_features=2,
        n_estimators=1,
    )
    assert result.n_nn_features == 2
    assert result.n_estimators == 1
    assert len(result.estimators) == 1
    assert result.tl_residuals.shape == (2,)
    assert result.pinn_residuals.shape == (2,)
