"""Unit tests for lmc.compensation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lmc.calibration import CalibrationResult
from lmc.columns import (
    COL_ALT,
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_LAT,
    COL_LON,
    COL_TIME,
    COL_TMI_COMPENSATED,
)
from lmc.compensation import compensate
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_CONFIG_A = PipelineConfig(model_terms="a")
_CONFIG_B = PipelineConfig(model_terms="b")
_CONFIG_C = PipelineConfig(model_terms="c")

_N_TERMS = {"a": 3, "b": 9, "c": 18}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n_rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Generate a valid magnetometer DataFrame with random (but normalised)
    B-vectors."""
    raw = rng.standard_normal((n_rows, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms

    b_total = 50_000.0
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


def _make_result(
    config: PipelineConfig,
    df: pl.DataFrame,
    coefficients: np.ndarray,
) -> CalibrationResult:
    """Build a CalibrationResult with the given coefficients for testing."""
    n_terms = _N_TERMS[config.model_terms]
    return CalibrationResult(
        coefficients=coefficients,
        residuals=np.zeros(len(df)),
        condition_number=1.0,
        n_terms=n_terms,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_output_has_tmi_compensated_column() -> None:
    rng = np.random.default_rng(0)
    df = _make_synthetic_df(50, rng)
    coefficients = np.ones(3)
    result = _make_result(_CONFIG_A, df, coefficients)
    out = compensate(df, result, _CONFIG_A)
    assert COL_TMI_COMPENSATED in out.columns


def test_output_row_count_preserved() -> None:
    rng = np.random.default_rng(1)
    df = _make_synthetic_df(60, rng)
    coefficients = np.ones(3)
    result = _make_result(_CONFIG_A, df, coefficients)
    out = compensate(df, result, _CONFIG_A)
    assert out.height == df.height


def test_compensated_values_correct() -> None:
    rng = np.random.default_rng(2)
    df = _make_synthetic_df(50, rng)
    config = _CONFIG_A
    coefficients = np.array([1.0, -2.0, 0.5])
    result = _make_result(config, df, coefficients)

    A = build_feature_matrix(df, config).to_numpy()
    interference = A @ coefficients
    expected = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64) - interference

    out = compensate(df, result, config)
    np.testing.assert_allclose(
        out[COL_TMI_COMPENSATED].to_numpy(), expected, atol=1e-10
    )


def test_round_trip_residual_near_zero() -> None:
    """Compensation recovers the constant earth field after removing interference.

    Set B_total = B_earth + interference (keeping direction cosines unchanged),
    then verify tmi_compensated recovers B_earth to floating-point precision.
    """
    rng = np.random.default_rng(3)
    n_rows = 60
    df_base = _make_synthetic_df(n_rows, rng)
    config = _CONFIG_B
    coefficients = np.array([1.0, -2.0, 0.5, 0.3, -0.1, 0.7, -0.4, 0.2, -0.8])

    # Compute interference using the base direction cosines.
    A = build_feature_matrix(df_base, config).to_numpy()
    interference = A @ coefficients

    # Build a survey df where the measured field = B_earth + interference.
    # Scale B_x/B_y/B_z proportionally so direction cosines stay identical,
    # preserving the feature matrix exactly.
    b_earth = np.asarray(df_base[COL_BTOTAL].to_numpy(), dtype=np.float64)
    b_total_survey = b_earth + interference  # always >> 0 (|interference| << b_earth)
    scale = b_total_survey / b_earth
    df_survey = df_base.with_columns(
        pl.Series(COL_BTOTAL, b_total_survey, dtype=pl.Float64),
        pl.Series(COL_BX, df_base[COL_BX].to_numpy() * scale, dtype=pl.Float64),
        pl.Series(COL_BY, df_base[COL_BY].to_numpy() * scale, dtype=pl.Float64),
        pl.Series(COL_BZ, df_base[COL_BZ].to_numpy() * scale, dtype=pl.Float64),
    )

    result = CalibrationResult(
        coefficients=coefficients,
        residuals=np.zeros(n_rows),
        condition_number=1.0,
        n_terms=9,
    )
    out = compensate(df_survey, result, config)
    # Compensated = (B_earth + interference) - interference = B_earth.
    np.testing.assert_allclose(out[COL_TMI_COMPENSATED].to_numpy(), b_earth, atol=1e-10)


def test_terms_mismatch_raises() -> None:
    rng = np.random.default_rng(4)
    df = _make_synthetic_df(50, rng)
    # result has 18 terms (model_terms="c") but we compensate with "a" (3 terms).
    result = CalibrationResult(
        coefficients=np.ones(18),
        residuals=np.zeros(50),
        condition_number=1.0,
        n_terms=18,
    )
    with pytest.raises(ValueError, match="18"):
        compensate(df, result, _CONFIG_A)


@pytest.mark.parametrize("model_terms", ["a", "b", "c"])
def test_works_for_all_model_terms(model_terms: str) -> None:
    rng = np.random.default_rng(5)
    n_rows = 50
    df = _make_synthetic_df(n_rows, rng)
    config = PipelineConfig(model_terms=model_terms)  # type: ignore[arg-type]
    n_terms = _N_TERMS[model_terms]
    result = CalibrationResult(
        coefficients=np.zeros(n_terms),
        residuals=np.zeros(n_rows),
        condition_number=1.0,
        n_terms=n_terms,
    )
    out = compensate(df, result, config)
    assert out.height == n_rows
    assert COL_TMI_COMPENSATED in out.columns
