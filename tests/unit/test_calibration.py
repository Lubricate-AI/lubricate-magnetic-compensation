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


def test_ridge_populates_diagnostics() -> None:
    """Ridge result should have selected_alpha and effective_dof populated."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_ridge=True, ridge_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.selected_alpha == pytest.approx(1e-3)  # pyright: ignore[reportUnknownMemberType]
    assert result.effective_dof is not None
    assert 0.0 <= result.effective_dof <= result.n_terms


def test_ridge_effective_dof_decreases_with_stronger_regularization() -> None:
    """Larger ridge alpha should shrink effective_dof toward zero."""
    c_true = np.array([1.0, -2.0, 0.5])
    config_weak = PipelineConfig(model_terms="a", use_ridge=True, ridge_alpha=1e-6)
    config_strong = PipelineConfig(model_terms="a", use_ridge=True, ridge_alpha=1e3)
    df_weak, segments = _make_synthetic_data(c_true, config_weak)
    df_strong, _ = _make_synthetic_data(c_true, config_strong)
    result_weak = calibrate(df_weak, segments, config_weak)
    result_strong = calibrate(df_strong, segments, config_strong)
    assert result_weak.effective_dof is not None
    assert result_strong.effective_dof is not None
    assert result_strong.effective_dof < result_weak.effective_dof


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
        singular_values=np.zeros(3),
        n_terms=3,
    )
    assert r.selected_alpha is None
    assert r.effective_dof is None


def test_lasso_recovers_reasonable() -> None:
    """LASSO introduces bias but should return plausible, finite coefficients.

    Uses a small lasso_alpha (unnormalized convention) so the scaled sklearn
    alpha remains weak and near-exact recovery is expected.
    """
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-5)
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


# ---------------------------------------------------------------------------
# Alpha convention tests: unnormalized (ridge) convention
# ---------------------------------------------------------------------------


def test_lasso_uses_n_samples_scaled_alpha_internally() -> None:
    """Non-CV LASSO should pass lasso_alpha * n_samples to sklearn Lasso.

    sklearn's Lasso normalizes by n_samples internally, so to maintain
    the unnormalized (ridge) alpha convention, we scale up before passing.
    """
    c_true = np.array([1.0, -2.0, 0.5])
    n_rows = 80
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config, n_rows=n_rows)

    result = calibrate(df, segments, config)

    # Recompute manually with the expected scaled alpha
    A = build_feature_matrix(df.slice(0, n_rows), config).to_numpy()
    dB = df[COL_DELTA_B].to_numpy().astype(np.float64)
    from sklearn.linear_model import Lasso as _Lasso

    expected = _Lasso(alpha=1e-3 * n_rows, fit_intercept=False, max_iter=10_000)
    expected.fit(A, dB)
    np.testing.assert_allclose(result.coefficients, expected.coef_, atol=1e-10)


def test_lasso_selected_alpha_is_user_convention_not_scaled() -> None:
    """CalibrationResult.selected_alpha should be the user-facing alpha, not scaled."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=2e-4)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    # selected_alpha must reflect what the user passed, not the sklearn-scaled value
    assert result.selected_alpha == pytest.approx(2e-4)  # pyright: ignore[reportUnknownMemberType]


def test_lasso_cv_selected_alpha_in_unnormalized_convention() -> None:
    """CV LASSO selected_alpha must be in unnormalized (ridge) convention.

    LassoCV returns alpha in sklearn's convention (normalized by n_samples).
    We must divide by n_samples to bring it into the same convention as ridge.
    """
    df, segments = _make_multicollinear_df_for_cv()
    n_rows = len(df)
    config = PipelineConfig(model_terms="a", use_lasso=True, use_cv=True, cv_folds=5)
    result = calibrate(df, segments, config)

    # Reproduce what LassoCV returns in sklearn convention
    A = build_feature_matrix(df, config).to_numpy()
    dB = df[COL_DELTA_B].to_numpy().astype(np.float64)
    from sklearn.linear_model import LassoCV as _LassoCV
    from sklearn.model_selection import TimeSeriesSplit as _TSS

    cv = _TSS(n_splits=5)
    model_cv = _LassoCV(cv=cv, fit_intercept=False, max_iter=10_000)
    model_cv.fit(A, dB)

    # After fix: selected_alpha = model_cv.alpha_ / n_rows
    expected_user_alpha = float(model_cv.alpha_) / n_rows
    assert result.selected_alpha == pytest.approx(expected_user_alpha, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]


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
    noise = rng.normal(0, 1e-5, size=(n_rows, 3))
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


# ---------------------------------------------------------------------------
# Low-sample warning tests
# ---------------------------------------------------------------------------


def test_low_sample_warning_c_model() -> None:
    """C-model with < 10,000 samples should emit a UserWarning about sample count."""
    c_true = np.arange(1, 19, dtype=np.float64) * 0.1
    df, segments = _make_synthetic_data(c_true, _CONFIG_C, n_rows=80)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrate(df, segments, _CONFIG_C)

    sample_warnings = [
        w for w in caught if "10,000" in str(w.message) or "10000" in str(w.message)
    ]
    assert len(sample_warnings) == 1, (
        f"Expected exactly one low-sample warning, got {len(sample_warnings)}"
    )


def test_no_low_sample_warning_when_sufficient() -> None:
    """C-model with >= 10,000 samples should NOT emit a low-sample warning."""
    c_true = np.arange(1, 19, dtype=np.float64) * 0.1
    df, segments = _make_synthetic_data(c_true, _CONFIG_C, n_rows=10_000)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrate(df, segments, _CONFIG_C)

    sample_warnings = [
        w for w in caught if "10,000" in str(w.message) or "10000" in str(w.message)
    ]
    assert len(sample_warnings) == 0, (
        f"Expected no low-sample warning with 10,000 rows, got {len(sample_warnings)}"
    )


def test_no_low_sample_warning_for_b_model() -> None:
    """B-model should never emit a low-sample warning regardless of row count."""
    c_true = np.array([1.0, -2.0, 0.5, 0.3, -0.1, 0.7, -0.4, 0.2, -0.8])
    df, segments = _make_synthetic_data(c_true, _CONFIG_B, n_rows=80)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calibrate(df, segments, _CONFIG_B)

    sample_warnings = [
        w for w in caught if "10,000" in str(w.message) or "10000" in str(w.message)
    ]
    assert len(sample_warnings) == 0


# ---------------------------------------------------------------------------
# Singular value tests
# ---------------------------------------------------------------------------


def test_singular_values_present_and_correct_shape() -> None:
    """CalibrationResult should contain singular values matching n_terms."""
    c_true = np.arange(1, 19, dtype=np.float64) * 0.1
    df, segments = _make_synthetic_data(c_true, _CONFIG_C, n_rows=80)
    result = calibrate(df, segments, _CONFIG_C)
    assert hasattr(result, "singular_values")
    assert result.singular_values.shape == (result.n_terms,)
    assert result.singular_values.dtype == np.float64


def test_singular_values_sorted_descending_and_positive() -> None:
    """Singular values should be non-negative and sorted largest-first."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A, n_rows=80)
    result = calibrate(df, segments, _CONFIG_A)

    assert np.all(result.singular_values >= 0.0)
    # Check descending order
    assert np.all(result.singular_values[:-1] >= result.singular_values[1:])


def test_condition_number_matches_singular_values() -> None:
    """Condition number should equal max(sv) / min(sv)."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A, n_rows=80)
    result = calibrate(df, segments, _CONFIG_A)

    expected_cond = result.singular_values[0] / result.singular_values[-1]
    np.testing.assert_allclose(result.condition_number, expected_cond, rtol=1e-10)


# ---------------------------------------------------------------------------
# Cross-validation alpha selection tests
# ---------------------------------------------------------------------------


def _make_multicollinear_df_for_cv(
    n_rows: int = 200, seed: int = 0
) -> tuple[pl.DataFrame, list[Segment]]:
    """Return (df, segments) with a nearly collinear A-matrix.

    Uses a larger n_rows than typical to ensure TimeSeriesSplit has enough
    data in each fold (at least n_terms rows per fold).
    """
    rng = np.random.default_rng(seed)
    # Make all three direction cosine vectors nearly identical to force
    # collinearity in the feature matrix.
    raw = rng.standard_normal((n_rows, 3))
    raw[:, 1] = raw[:, 0] + rng.normal(0, 0.01, n_rows)
    raw[:, 2] = raw[:, 0] + rng.normal(0, 0.01, n_rows)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms
    b_total = 50_000.0
    base_df = pl.DataFrame(
        {
            COL_TIME: np.arange(n_rows, dtype=np.float64),
            COL_LAT: np.full(n_rows, 45.0),
            COL_LON: np.full(n_rows, -75.0),
            COL_ALT: np.full(n_rows, 300.0),
            COL_BTOTAL: np.full(n_rows, b_total),
            COL_BX: cosines[:, 0] * b_total,
            COL_BY: cosines[:, 1] * b_total,
            COL_BZ: cosines[:, 2] * b_total,
        }
    )
    config = PipelineConfig(model_terms="a")
    A = build_feature_matrix(base_df, config).to_numpy()
    c_true = np.array([1.0, -2.0, 0.5])
    delta_b = A @ c_true + rng.normal(0, 0.1, n_rows)
    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=n_rows)]
    return df, segments


def test_ridge_cv_selected_alpha_is_a_finite_positive_float() -> None:
    """use_cv=True with use_ridge=True yields a positive finite selected_alpha."""
    df, segments = _make_multicollinear_df_for_cv()
    config = PipelineConfig(model_terms="a", use_ridge=True, use_cv=True, cv_folds=5)
    result = calibrate(df, segments, config)
    assert result.selected_alpha is not None
    assert math.isfinite(result.selected_alpha)
    assert result.selected_alpha > 0.0


def test_ridge_cv_selected_alpha_is_positive() -> None:
    """When use_cv=True with use_ridge=True, selected_alpha must be positive."""
    df, segments = _make_multicollinear_df_for_cv()
    config = PipelineConfig(model_terms="a", use_ridge=True, use_cv=True, cv_folds=5)
    result = calibrate(df, segments, config)
    assert result.selected_alpha is not None
    # CV picks from logspace(-6, 2, 100); the fixed default 1e-3 may or may not be
    # chosen, but the alpha must come from the CV grid, not be trivially None.
    assert result.selected_alpha > 0.0


def test_lasso_cv_selected_alpha_is_finite_positive() -> None:
    df, segments = _make_multicollinear_df_for_cv()
    config = PipelineConfig(model_terms="a", use_lasso=True, use_cv=True, cv_folds=5)
    result = calibrate(df, segments, config)
    assert result.selected_alpha is not None
    assert math.isfinite(result.selected_alpha)
    assert result.selected_alpha > 0.0


def test_elastic_net_cv_selected_alpha_is_finite_positive() -> None:
    df, segments = _make_multicollinear_df_for_cv()
    config = PipelineConfig(
        model_terms="a", use_elastic_net=True, use_cv=True, cv_folds=5
    )
    result = calibrate(df, segments, config)
    assert result.selected_alpha is not None
    assert math.isfinite(result.selected_alpha)
    assert result.selected_alpha > 0.0


def test_cv_not_enabled_uses_config_alpha() -> None:
    """Without use_cv, selected_alpha must equal the config-specified ridge_alpha."""
    df, segments = _make_multicollinear_df_for_cv()
    config = PipelineConfig(
        model_terms="a", use_ridge=True, ridge_alpha=0.42, use_cv=False
    )
    result = calibrate(df, segments, config)
    assert result.selected_alpha == pytest.approx(0.42)  # pyright: ignore[reportUnknownMemberType]


def _make_ill_conditioned_df(
    n_rows: int = 200, seed: int = 7
) -> tuple[pl.DataFrame, list[Segment]]:
    """Return (df, segments) whose A-matrix has condition_number >> 1e6."""
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_rows, 3))
    raw[:, 1] = raw[:, 0] + rng.normal(0, 1e-7, n_rows)
    raw[:, 2] = raw[:, 0] + rng.normal(0, 1e-7, n_rows)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms
    b_total = 50_000.0
    base_df = pl.DataFrame(
        {
            COL_TIME: np.arange(n_rows, dtype=np.float64),
            COL_LAT: np.full(n_rows, 45.0),
            COL_LON: np.full(n_rows, -75.0),
            COL_ALT: np.full(n_rows, 300.0),
            COL_BTOTAL: np.full(n_rows, b_total),
            COL_BX: cosines[:, 0] * b_total,
            COL_BY: cosines[:, 1] * b_total,
            COL_BZ: cosines[:, 2] * b_total,
        }
    )
    config = PipelineConfig(model_terms="a")
    A = build_feature_matrix(base_df, config).to_numpy()
    c_true = np.array([1.0, -2.0, 0.5])
    delta_b = A @ c_true + rng.normal(0, 0.1, n_rows)
    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=n_rows)]
    return df, segments


def test_auto_regularize_with_cv_uses_cv_alpha() -> None:
    """auto_regularize=True + use_cv=True: CV-selected alpha when ill-conditioned."""
    df, segments = _make_ill_conditioned_df()
    config = PipelineConfig(
        model_terms="a", auto_regularize=True, use_cv=True, cv_folds=5
    )
    result = calibrate(df, segments, config)
    assert result.condition_number > config.condition_number_threshold
    assert result.selected_alpha is not None
    assert math.isfinite(result.selected_alpha)
    assert result.selected_alpha > 0.0


# ---------------------------------------------------------------------------
# auto_regularize tests
# ---------------------------------------------------------------------------


def test_auto_regularize_engages_ridge_when_ill_conditioned() -> None:
    """auto_regularize=True must set selected_alpha when condition_number is huge."""
    df, segments = _make_ill_conditioned_df()
    config = PipelineConfig(model_terms="a", auto_regularize=True)
    result = calibrate(df, segments, config)
    # Ill-conditioned → auto ridge should have been engaged.
    assert result.condition_number > config.condition_number_threshold
    assert result.selected_alpha is not None
    assert result.selected_alpha > 0.0


def test_auto_regularize_does_not_engage_when_well_conditioned() -> None:
    """auto_regularize=True must NOT apply regularization when well-conditioned."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A, n_rows=200)
    config = PipelineConfig(model_terms="a", auto_regularize=True)
    result = calibrate(df, segments, config)
    # Well-conditioned → OLS should have been used.
    assert result.condition_number <= config.condition_number_threshold
    assert result.selected_alpha is None


def test_auto_regularize_respects_explicit_method() -> None:
    """When use_lasso=True and auto_regularize=True, LASSO (not ridge) is used."""
    df, segments = _make_ill_conditioned_df()
    config = PipelineConfig(
        model_terms="a", auto_regularize=True, use_lasso=True, lasso_alpha=0.01
    )
    result = calibrate(df, segments, config)
    # Confirm the fixture is ill-conditioned so auto_regularize would have fired
    # without the explicit use_lasso flag.
    assert result.condition_number > config.condition_number_threshold
    # Explicit LASSO takes priority — selected_alpha should equal lasso_alpha.
    assert result.selected_alpha == pytest.approx(0.01)  # pyright: ignore[reportUnknownMemberType]
