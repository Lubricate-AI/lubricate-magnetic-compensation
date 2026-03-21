"""Unit tests for lmc.calibration."""

from __future__ import annotations

import math
import warnings

import numpy as np
import polars as pl
import pytest

from lmc.calibration import calibrate
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
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import Segment

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_CONFIG_A = PipelineConfig(model_terms="a")
_CONFIG_B = PipelineConfig(model_terms="b")
_CONFIG_C = PipelineConfig(model_terms="c")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n_rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Generate a valid magnetometer DataFrame with random (but normalised) B-vectors.

    The direction cosines vary across rows so the resulting A-matrix has
    full column rank.  Returns a DataFrame without ``COL_DELTA_B``; callers
    should add that column after building the feature matrix.
    """
    # Draw random unit vectors for direction cosines.
    raw = rng.standard_normal((n_rows, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms  # shape (n_rows, 3) — each row is a unit vector

    b_total = 50_000.0  # nT (Earth-like magnitude)
    bx = cosines[:, 0] * b_total
    by = cosines[:, 1] * b_total
    bz = cosines[:, 2] * b_total

    return pl.DataFrame(
        {
            COL_TIME: np.arange(n_rows, dtype=np.float64),
            COL_LAT: np.full(n_rows, 45.0),
            COL_LON: np.full(n_rows, -75.0),
            COL_ALT: np.full(n_rows, 300.0),
            COL_BTOTAL: np.full(n_rows, b_total),
            COL_BX: bx,
            COL_BY: by,
            COL_BZ: bz,
        }
    )


def _make_synthetic_data(
    c_true: np.ndarray,
    config: PipelineConfig,
    noise_std: float = 0.0,
    n_rows: int = 80,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build a (df, segments) pair for calibration tests.

    ``COL_DELTA_B`` is set to ``A @ c_true + noise`` where ``A`` is the
    feature matrix computed from the random magnetometer data.

    Parameters
    ----------
    c_true:
        Ground-truth coefficient vector; its length determines ``model_terms``.
    config:
        Pipeline configuration (controls term set, ridge, etc.).
    noise_std:
        Standard deviation of additive Gaussian noise on ``COL_DELTA_B``.
    n_rows:
        Number of synthetic data rows to generate.
    seed:
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    base_df = _make_synthetic_df(n_rows, rng)

    A = build_feature_matrix(base_df, config).to_numpy()
    delta_b = A @ c_true
    if noise_std > 0.0:
        delta_b = delta_b + rng.normal(0.0, noise_std, size=delta_b.shape)

    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))

    # Single segment covering all rows.
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=n_rows)]
    return df, segments


# ---------------------------------------------------------------------------
# Exact recovery tests — OLS on noise-free data
# ---------------------------------------------------------------------------


def test_exact_recovery_terms_a() -> None:
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    np.testing.assert_allclose(result.coefficients, c_true, atol=1e-10)


def test_exact_recovery_terms_b() -> None:
    c_true = np.array([1.0, -2.0, 0.5, 0.3, -0.1, 0.7, -0.4, 0.2, -0.8])
    df, segments = _make_synthetic_data(c_true, _CONFIG_B)
    result = calibrate(df, segments, _CONFIG_B)
    np.testing.assert_allclose(result.coefficients, c_true, atol=1e-10)


def test_exact_recovery_terms_c() -> None:
    c_true = np.arange(1, 19, dtype=np.float64) * 0.1
    df, segments = _make_synthetic_data(c_true, _CONFIG_C)
    result = calibrate(df, segments, _CONFIG_C)
    np.testing.assert_allclose(result.coefficients, c_true, atol=1e-10)


# ---------------------------------------------------------------------------
# Residual tests
# ---------------------------------------------------------------------------


def test_residuals_near_zero_noise_free() -> None:
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    np.testing.assert_allclose(result.residuals, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# n_terms tests
# ---------------------------------------------------------------------------


def test_n_terms_matches_model_a() -> None:
    c_true = np.zeros(3)
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    assert result.n_terms == 3


def test_n_terms_matches_model_b() -> None:
    c_true = np.zeros(9)
    df, segments = _make_synthetic_data(c_true, _CONFIG_B)
    result = calibrate(df, segments, _CONFIG_B)
    assert result.n_terms == 9


def test_n_terms_matches_model_c() -> None:
    c_true = np.zeros(18)
    df, segments = _make_synthetic_data(c_true, _CONFIG_C)
    result = calibrate(df, segments, _CONFIG_C)
    assert result.n_terms == 18


# ---------------------------------------------------------------------------
# Ridge regression test
# ---------------------------------------------------------------------------


def test_ridge_recovers_reasonable() -> None:
    """Ridge introduces bias but should return plausible coefficients."""
    c_true = np.array([1.0, -2.0, 0.5])
    config_ridge = PipelineConfig(model_terms="a", use_ridge=True, ridge_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config_ridge)
    result = calibrate(df, segments, config_ridge)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    # Ridge is biased, but the sign pattern and rough magnitude should survive
    # for a well-conditioned system with small alpha.
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)


# ---------------------------------------------------------------------------
# Condition number tests
# ---------------------------------------------------------------------------


def test_condition_number_reported() -> None:
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    assert math.isfinite(result.condition_number)
    assert result.condition_number > 0.0


def test_condition_number_warning() -> None:
    """Calibrating with a near-singular A-matrix should emit a UserWarning."""
    # Build a near-singular A by repeating the same row (rank 1 → huge cond number).
    rng = np.random.default_rng(7)
    base_df = _make_synthetic_df(1, rng)
    single_row = base_df.to_dicts()[0]
    # Stack 20 identical rows → rank-1 A, condition number → ∞
    n_rows = 20
    repeated_df = pl.DataFrame([single_row] * n_rows).with_columns(
        pl.Series(COL_TIME, np.arange(n_rows, dtype=np.float64))
    )
    delta_b = np.ones(n_rows)
    df_singular = repeated_df.with_columns(
        pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64)
    )
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=n_rows)]

    config_low_threshold = PipelineConfig(
        model_terms="a", condition_number_threshold=1.0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrate(df_singular, segments, config_low_threshold)

    assert any(issubclass(w.category, UserWarning) for w in caught), (
        "Expected a UserWarning about the condition number."
    )


# ---------------------------------------------------------------------------
# Error condition tests
# ---------------------------------------------------------------------------


def test_empty_segments_raises() -> None:
    c_true = np.array([1.0, -2.0, 0.5])
    df, _ = _make_synthetic_data(c_true, _CONFIG_A)
    with pytest.raises(ValueError, match="segments must be non-empty"):
        calibrate(df, [], _CONFIG_A)


def test_missing_delta_b_raises() -> None:
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    df_no_delta = df.drop(COL_DELTA_B)
    with pytest.raises(ValueError, match=COL_DELTA_B):
        calibrate(df_no_delta, segments, _CONFIG_A)


def test_zero_length_segment_raises() -> None:
    """Zero-length segment (start_idx == end_idx) should raise ValueError."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, _ = _make_synthetic_data(c_true, _CONFIG_A)
    bad_seg = Segment(maneuver="steady", heading="N", start_idx=5, end_idx=5)
    with pytest.raises(ValueError, match="invalid bounds"):
        calibrate(df, [bad_seg], _CONFIG_A)


def test_out_of_range_segment_raises() -> None:
    """Out-of-range segment (end_idx > len(df)) should raise ValueError."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, _ = _make_synthetic_data(c_true, _CONFIG_A)
    bad_seg = Segment(maneuver="steady", heading="N", start_idx=0, end_idx=len(df) + 1)
    with pytest.raises(ValueError, match="invalid bounds"):
        calibrate(df, [bad_seg], _CONFIG_A)


def test_calibration_result_diagnostic_fields_default_none() -> None:
    """OLS calibration should have None diagnostics (no regularization)."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    assert result.selected_alpha is None
    assert result.effective_dof is None


def test_calibration_result_existing_sites_unbroken() -> None:
    """Direct instantiation with keyword args still works without new fields."""
    from lmc.calibration import CalibrationResult

    r = CalibrationResult(
        coefficients=np.zeros(3),
        residuals=np.zeros(5),
        condition_number=1.0,
        n_terms=3,
    )
    assert r.selected_alpha is None
    assert r.effective_dof is None


def test_lasso_recovers_reasonable() -> None:
    """LASSO introduces bias but should return plausible, finite coefficients."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)


def test_lasso_populates_diagnostics() -> None:
    """LASSO result should have selected_alpha and effective_dof populated."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.selected_alpha == pytest.approx(1e-3)  # pyright: ignore[reportUnknownMemberType]
    assert result.effective_dof is not None
    assert 0.0 <= result.effective_dof <= result.n_terms


def test_lasso_zeroes_weak_terms_under_strong_regularization() -> None:
    """With high alpha, LASSO should zero out at least some coefficients."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=10.0)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    # At least one coefficient zeroed; effective_dof < n_terms
    assert result.effective_dof is not None
    assert result.effective_dof < result.n_terms


def test_elastic_net_recovers_reasonable() -> None:
    """ElasticNet should return plausible, finite coefficients."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=1e-3,
        elastic_net_l1_ratio=0.5,
    )
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)


def test_elastic_net_populates_diagnostics() -> None:
    config = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=1e-3,
        elastic_net_l1_ratio=0.5,
    )
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.selected_alpha == pytest.approx(1e-3)  # pyright: ignore[reportUnknownMemberType]
    assert result.effective_dof is not None
    assert 0.0 <= result.effective_dof <= result.n_terms


def test_elastic_net_l1_ratio_1_behaves_like_lasso() -> None:
    """l1_ratio=1.0 is pure L1, so strong regularization should zero some coefs."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=10.0,
        elastic_net_l1_ratio=1.0,
    )
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.effective_dof is not None
    assert result.effective_dof < result.n_terms


def _make_multicollinear_df(
    c_true: np.ndarray,
    config: PipelineConfig,
    n_rows: int = 100,
    seed: int = 99,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build data where direction cosines are nearly constant (multicollinear).

    All aircraft headings point nearly the same direction, so the feature
    matrix columns are highly correlated, creating an ill-conditioned system.
    """
    rng = np.random.default_rng(seed)
    # All rows point nearly the same direction — creates multicollinearity.
    base_direction = np.array([0.6, 0.8, 0.0])
    noise = rng.normal(0, 0.001, size=(n_rows, 3))
    directions = base_direction + noise
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    cosines = directions / norms

    b_total = 50_000.0
    bx = cosines[:, 0] * b_total
    by = cosines[:, 1] * b_total
    bz = cosines[:, 2] * b_total

    base_df = pl.DataFrame(
        {
            COL_TIME: np.arange(n_rows, dtype=np.float64),
            COL_LAT: np.full(n_rows, 45.0),
            COL_LON: np.full(n_rows, -75.0),
            COL_ALT: np.full(n_rows, 300.0),
            COL_BTOTAL: np.full(n_rows, b_total),
            COL_BX: bx,
            COL_BY: by,
            COL_BZ: bz,
        }
    )

    A = build_feature_matrix(base_df, config).to_numpy()
    delta_b = A @ c_true + rng.normal(0, 0.1, size=n_rows)
    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=n_rows)]
    return df, segments


def test_ridge_more_stable_than_ols_on_multicollinear_data() -> None:
    """Ridge coefficients have smaller L2 norm than OLS on ill-conditioned data."""
    c_true = np.array([1.0, -2.0, 0.5])
    config_ols = PipelineConfig(model_terms="a")
    config_ridge = PipelineConfig(model_terms="a", use_ridge=True, ridge_alpha=1.0)

    df, segments = _make_multicollinear_df(c_true, config_ols)

    result_ols = calibrate(df, segments, config_ols)
    result_ridge = calibrate(df, segments, config_ridge)

    ols_norm = np.linalg.norm(result_ols.coefficients)
    ridge_norm = np.linalg.norm(result_ridge.coefficients)
    assert ridge_norm < ols_norm, (
        f"Expected ridge norm {ridge_norm:.3f} < OLS norm {ols_norm:.3f}"
    )


def test_lasso_more_stable_than_ols_on_multicollinear_data() -> None:
    """LASSO coefficients have smaller L1 norm than OLS on ill-conditioned data."""
    c_true = np.array([1.0, -2.0, 0.5])
    config_ols = PipelineConfig(model_terms="a")
    config_lasso = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1.0)

    df, segments = _make_multicollinear_df(c_true, config_ols)

    result_ols = calibrate(df, segments, config_ols)
    result_lasso = calibrate(df, segments, config_lasso)

    ols_l1 = float(np.sum(np.abs(result_ols.coefficients)))
    lasso_l1 = float(np.sum(np.abs(result_lasso.coefficients)))
    assert lasso_l1 < ols_l1, (
        f"Expected LASSO L1 norm {lasso_l1:.3f} < OLS L1 norm {ols_l1:.3f}"
    )


def test_elastic_net_more_stable_than_ols_on_multicollinear_data() -> None:
    """ElasticNet coefficients have smaller norm than OLS on ill-conditioned data."""
    c_true = np.array([1.0, -2.0, 0.5])
    config_ols = PipelineConfig(model_terms="a")
    config_en = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=1.0,
        elastic_net_l1_ratio=0.5,
    )

    df, segments = _make_multicollinear_df(c_true, config_ols)

    result_ols = calibrate(df, segments, config_ols)
    result_en = calibrate(df, segments, config_en)

    ols_norm = np.linalg.norm(result_ols.coefficients)
    en_norm = np.linalg.norm(result_en.coefficients)
    assert en_norm < ols_norm, (
        f"Expected ElasticNet norm {en_norm:.3f} < OLS norm {ols_norm:.3f}"
    )
