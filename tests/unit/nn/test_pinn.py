# tests/unit/nn/test_pinn.py
"""Unit tests for lmc.nn.pinn."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from lmc.calibration import CalibrationResult
from lmc.columns import (
    COL_ALT,
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_DELTA_B,
    COL_LAT,
    COL_LON,
    COL_TIME,
)
from lmc.nn.pinn import (
    PINNCalibrationResult,
    PINNConfig,
    _extract_pinn_features,  # pyright: ignore[reportPrivateUsage]
    calibrate_pinn,
    predict_pinn,
)
from lmc.segmentation import Segment


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


def _make_df(n: int, rng: np.random.Generator) -> pl.DataFrame:
    """Synthetic calibration DataFrame with physically valid B vectors."""
    raw = rng.standard_normal((n, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cos = raw / norms
    b_total = 50_000.0
    bx = cos[:, 0] * b_total
    by = cos[:, 1] * b_total
    bz = cos[:, 2] * b_total
    delta_b = 200.0 * cos[:, 0] + 150.0 * cos[:, 1] ** 2 - 80.0 * cos[:, 2]
    return pl.DataFrame(
        {
            COL_TIME: np.arange(n, dtype=np.float64),
            COL_LAT: np.zeros(n),
            COL_LON: np.zeros(n),
            COL_ALT: np.full(n, 300.0),
            COL_BTOTAL: np.full(n, b_total),
            COL_BX: bx,
            COL_BY: by,
            COL_BZ: bz,
            COL_DELTA_B: delta_b,
        }
    )


def test_extract_pinn_features_shape_b_model() -> None:
    rng = np.random.default_rng(0)
    df = _make_df(100, rng)
    X = _extract_pinn_features(df, "b")
    assert X.shape == (100, 9), f"Expected (100, 9) for b-model, got {X.shape}"
    assert X.dtype == np.float64


def test_extract_pinn_features_shape_a_model() -> None:
    rng = np.random.default_rng(1)
    df = _make_df(50, rng)
    X = _extract_pinn_features(df, "a")
    assert X.shape == (50, 3), f"Expected (50, 3) for a-model, got {X.shape}"


def test_calibrate_pinn_returns_result() -> None:
    rng = np.random.default_rng(10)
    df = _make_df(200, rng)
    seg = Segment(start_idx=0, end_idx=200, maneuver="pitch", heading="N")
    cfg = PINNConfig(n_estimators=3, max_iter=50)
    result = calibrate_pinn(df, [seg], cfg)
    assert isinstance(result, PINNCalibrationResult)
    assert result.n_estimators == 3
    assert result.tl_residuals.shape == (200,)
    assert result.pinn_residuals.shape == (200,)
    assert result.n_nn_features == 9  # b-model default


def test_calibrate_pinn_empty_segments_raises() -> None:
    rng = np.random.default_rng(11)
    df = _make_df(50, rng)
    cfg = PINNConfig()
    with pytest.raises(ValueError, match="segments must be non-empty"):
        calibrate_pinn(df, [], cfg)


def test_calibrate_pinn_missing_delta_b_raises() -> None:
    rng = np.random.default_rng(12)
    df = _make_df(50, rng).drop(COL_DELTA_B)
    seg = Segment(start_idx=0, end_idx=50, maneuver="pitch", heading="N")
    cfg = PINNConfig()
    with pytest.raises(ValueError, match="delta_B"):
        calibrate_pinn(df, [seg], cfg)


def test_calibrate_pinn_zero_estimators_raises() -> None:
    rng = np.random.default_rng(13)
    df = _make_df(50, rng)
    seg = Segment(start_idx=0, end_idx=50, maneuver="pitch", heading="N")
    cfg = PINNConfig(n_estimators=0)
    with pytest.raises(ValueError, match="n_estimators"):
        calibrate_pinn(df, [seg], cfg)


def test_calibrate_pinn_pinn_residuals_smaller_than_tl() -> None:
    """PINN should reduce residuals compared to TL alone."""
    rng = np.random.default_rng(14)
    df = _make_df(300, rng)
    seg = Segment(start_idx=0, end_idx=300, maneuver="pitch", heading="N")
    # Use tl_model_terms="a" (3 permanent terms) so TL leaves residuals for NN
    cfg = PINNConfig(n_estimators=5, max_iter=200, tl_model_terms="a")
    result = calibrate_pinn(df, [seg], cfg)
    tl_rmse = float(np.sqrt(np.mean(result.tl_residuals**2)))
    pinn_rmse = float(np.sqrt(np.mean(result.pinn_residuals**2)))
    assert pinn_rmse <= tl_rmse, (
        f"PINN RMSE ({pinn_rmse:.3f}) should be <= TL RMSE ({tl_rmse:.3f})"
    )


def test_predict_pinn_shape_and_uncertainty() -> None:
    rng = np.random.default_rng(20)
    df = _make_df(200, rng)
    seg = Segment(start_idx=0, end_idx=200, maneuver="pitch", heading="N")
    cfg = PINNConfig(n_estimators=5, max_iter=50)
    result = calibrate_pinn(df, [seg], cfg)

    test_df = _make_df(50, np.random.default_rng(99))
    mean_pred, std_pred = predict_pinn(test_df, result, cfg)
    assert mean_pred.shape == (50,)
    assert std_pred.shape == (50,)
    assert np.all(std_pred >= 0.0)


def test_predict_pinn_uncertainty_is_nontrivial() -> None:
    """Bootstrap ensemble std should be positive for most samples."""
    rng = np.random.default_rng(21)
    df = _make_df(300, rng)
    seg = Segment(start_idx=0, end_idx=300, maneuver="pitch", heading="N")
    cfg = PINNConfig(n_estimators=10, max_iter=200)
    result = calibrate_pinn(df, [seg], cfg)

    _, std_pred = predict_pinn(df, result, cfg)
    assert np.mean(std_pred > 0.0) > 0.9
