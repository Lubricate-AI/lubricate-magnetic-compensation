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
