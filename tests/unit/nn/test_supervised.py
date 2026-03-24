"""Unit tests for lmc.nn.supervised."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

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
    COL_TMI_COMPENSATED,
)
from lmc.nn.supervised import (
    NNCalibrationResult,
    NNConfig,
    _extract_nn_features,  # pyright: ignore[reportPrivateUsage]
    calibrate_nn,
    compensate_nn,
    predict_nn,
)
from lmc.segmentation import Segment

# ---------------------------------------------------------------------------
# Synthetic data helper
# ---------------------------------------------------------------------------


def _make_df(n: int, rng: np.random.Generator) -> pl.DataFrame:
    raw = rng.standard_normal((n, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cos = raw / norms
    b_total = 50_000.0
    bx = cos[:, 0] * b_total
    by = cos[:, 1] * b_total
    bz = cos[:, 2] * b_total
    # Synthetic interference: simple nonlinear function of direction cosines
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


# ---------------------------------------------------------------------------
# Task 2: NNConfig
# ---------------------------------------------------------------------------


def test_nn_config_defaults() -> None:
    cfg = NNConfig()
    assert cfg.hidden_layer_sizes == (64, 64)
    assert cfg.activation == "relu"
    assert cfg.max_iter == 500
    assert cfg.n_estimators == 20
    assert cfg.random_state == 42


def test_nn_config_custom() -> None:
    cfg = NNConfig(hidden_layer_sizes=(100, 100, 50), n_estimators=10)
    assert cfg.hidden_layer_sizes == (100, 100, 50)
    assert cfg.n_estimators == 10


# ---------------------------------------------------------------------------
# Task 3: NNCalibrationResult
# ---------------------------------------------------------------------------


def test_nn_calibration_result_stores_fields() -> None:
    scaler = StandardScaler()
    scaler.fit([[1.0, 2.0, 3.0]])  # pyright: ignore[reportUnknownMemberType]
    estimators = [MLPRegressor(max_iter=10).fit([[1.0, 2.0, 3.0]], [0.5])]  # pyright: ignore[reportUnknownMemberType]
    residuals = np.array([0.1, -0.2])
    result = NNCalibrationResult(
        estimators=estimators,
        input_scaler=scaler,
        residuals=residuals,
        n_features=3,
        n_estimators=1,
    )
    assert result.n_features == 3
    assert result.n_estimators == 1
    assert len(result.estimators) == 1
    assert result.residuals.shape == (2,)


# ---------------------------------------------------------------------------
# Task 4: _extract_nn_features
# ---------------------------------------------------------------------------


def test_extract_nn_features_shape() -> None:
    df = pl.DataFrame(
        {
            COL_BX: [1.0, 2.0, 3.0],
            COL_BY: [4.0, 5.0, 6.0],
            COL_BZ: [7.0, 8.0, 9.0],
            "other_col": [0.0, 0.0, 0.0],
        }
    )
    X = _extract_nn_features(df)
    assert X.shape == (3, 3)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X[:, 0], [1.0, 2.0, 3.0])


def test_extract_nn_features_missing_column() -> None:
    df = pl.DataFrame({COL_BX: [1.0], COL_BY: [2.0]})  # missing COL_BZ
    with pytest.raises(ValueError, match="B_z"):
        _extract_nn_features(df)


# ---------------------------------------------------------------------------
# Task 5: calibrate_nn
# ---------------------------------------------------------------------------


def test_calibrate_nn_returns_result() -> None:
    rng = np.random.default_rng(0)
    df = _make_df(200, rng)
    seg = Segment(start_idx=0, end_idx=200, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=3, max_iter=50)
    result = calibrate_nn(df, [seg], cfg)
    assert isinstance(result, NNCalibrationResult)
    assert result.n_estimators == 3
    assert result.n_features == 3
    assert result.residuals.shape == (200,)


def test_calibrate_nn_empty_segments_raises() -> None:
    rng = np.random.default_rng(1)
    df = _make_df(50, rng)
    cfg = NNConfig()
    with pytest.raises(ValueError, match="segments must be non-empty"):
        calibrate_nn(df, [], cfg)


def test_calibrate_nn_missing_delta_b_raises() -> None:
    rng = np.random.default_rng(2)
    df = _make_df(50, rng).drop(COL_DELTA_B)
    seg = Segment(start_idx=0, end_idx=50, maneuver="pitch", heading="N")
    cfg = NNConfig()
    with pytest.raises(ValueError, match="delta_B"):
        calibrate_nn(df, [seg], cfg)


def test_calibrate_nn_zero_estimators_raises() -> None:
    rng = np.random.default_rng(7)
    df = _make_df(50, rng)
    seg = Segment(start_idx=0, end_idx=50, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=0)
    with pytest.raises(ValueError, match="n_estimators"):
        calibrate_nn(df, [seg], cfg)


# ---------------------------------------------------------------------------
# Task 6: predict_nn
# ---------------------------------------------------------------------------


def test_predict_nn_shape_and_uncertainty() -> None:
    rng = np.random.default_rng(3)
    df = _make_df(200, rng)
    seg = Segment(start_idx=0, end_idx=200, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=5, max_iter=50)
    result = calibrate_nn(df, [seg], cfg)

    test_df = _make_df(50, np.random.default_rng(99))
    mean_pred, std_pred = predict_nn(test_df, result)
    assert mean_pred.shape == (50,)
    assert std_pred.shape == (50,)
    assert np.all(std_pred >= 0.0)


def test_predict_nn_uncertainty_is_nontrivial() -> None:
    """Ensemble std should be positive for most samples, confirming UQ is active."""
    rng = np.random.default_rng(4)
    df = _make_df(300, rng)
    seg = Segment(start_idx=0, end_idx=300, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=10, max_iter=200)
    result = calibrate_nn(df, [seg], cfg)

    _, std_pred = predict_nn(df, result)
    # At least 90% of predictions should have non-zero ensemble std,
    # confirming that bootstrap diversity produces meaningful uncertainty estimates.
    assert np.mean(std_pred > 0.0) > 0.9


# ---------------------------------------------------------------------------
# Task 7: compensate_nn
# ---------------------------------------------------------------------------


def test_compensate_nn_adds_column() -> None:
    rng = np.random.default_rng(5)
    df = _make_df(200, rng)
    seg = Segment(start_idx=0, end_idx=200, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=3, max_iter=50)
    result = calibrate_nn(df, [seg], cfg)

    compensated = compensate_nn(df, result)
    assert COL_TMI_COMPENSATED in compensated.columns
    assert len(compensated) == len(df)


def test_compensate_nn_missing_btotal_raises() -> None:
    rng = np.random.default_rng(8)
    df = _make_df(100, rng)
    seg = Segment(start_idx=0, end_idx=100, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=3, max_iter=50)
    result = calibrate_nn(df, [seg], cfg)

    df_no_btotal = df.drop(COL_BTOTAL)
    with pytest.raises(ValueError, match="B_total"):
        compensate_nn(df_no_btotal, result)


def test_compensate_nn_values_are_btotal_minus_prediction() -> None:
    rng = np.random.default_rng(6)
    df = _make_df(100, rng)
    seg = Segment(start_idx=0, end_idx=100, maneuver="pitch", heading="N")
    cfg = NNConfig(n_estimators=3, max_iter=50)
    result = calibrate_nn(df, [seg], cfg)

    compensated = compensate_nn(df, result)
    mean_pred, _ = predict_nn(df, result)
    b_total = df[COL_BTOTAL].to_numpy()
    expected = b_total - mean_pred
    np.testing.assert_allclose(
        compensated[COL_TMI_COMPENSATED].to_numpy(), expected, rtol=1e-10
    )


# ---------------------------------------------------------------------------
# Task 8: public exports from lmc.nn
# ---------------------------------------------------------------------------


def test_public_exports() -> None:
    from lmc.nn import (  # noqa: PLC0415
        NNCalibrationResult,
        NNConfig,
        calibrate_nn,
        compensate_nn,
        predict_nn,
    )

    assert callable(calibrate_nn)
    assert callable(predict_nn)
    assert callable(compensate_nn)
    assert NNCalibrationResult is not None
    assert NNConfig is not None
