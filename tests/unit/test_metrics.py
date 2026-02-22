"""Unit tests for lmc.metrics."""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from lmc.calibration import CalibrationResult, calibrate
from lmc.columns import (
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
from lmc.metrics import compute_fom_report
from lmc.segmentation import HeadingType, ManeuverType, Segment

# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------

_CONFIG_A = PipelineConfig(model_terms="a")

# ---------------------------------------------------------------------------
# Synthetic data helpers (copied from test_calibration.py)
# ---------------------------------------------------------------------------


def _make_synthetic_df(n_rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """Generate a valid magnetometer DataFrame with random normalised B-vectors."""
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
    """Build a (df, segments) pair for metrics tests."""
    rng = np.random.default_rng(seed)
    base_df = _make_synthetic_df(n_rows, rng)

    A = build_feature_matrix(base_df, config).to_numpy()
    delta_b = A @ c_true
    if noise_std > 0.0:
        delta_b = delta_b + rng.normal(0.0, noise_std, size=delta_b.shape)

    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=n_rows)]
    return df, segments


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_rms_before_after_exact() -> None:
    """Noise-free data: rms_after ≈ 0, rms_before > 0."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    report = compute_fom_report(df, segments, result)

    stats = report.per_maneuver["steady"]
    assert stats.rms_before > 0.0
    np.testing.assert_allclose(stats.rms_after, 0.0, atol=1e-8)


def test_improvement_ratio_greater_than_one() -> None:
    """Calibration with noise should still yield improvement_ratio > 1."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A, noise_std=1.0)
    result = calibrate(df, segments, _CONFIG_A)
    report = compute_fom_report(df, segments, result)

    assert report.improvement_ratio > 1.0


def test_improvement_ratio_very_large_on_exact_fit() -> None:
    """Noise-free OLS: residuals are near-zero → improvement_ratio is very large."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)
    report = compute_fom_report(df, segments, result)

    # Residuals are ~1e-15 due to floating-point precision, so ratio is huge but finite.
    assert report.improvement_ratio > 1e10


def test_per_maneuver_keys_match_segments() -> None:
    """per_maneuver keys should exactly match maneuver types in segments."""
    c_true = np.array([1.0, -2.0, 0.5])
    rng = np.random.default_rng(0)
    base_df = _make_synthetic_df(60, rng)
    A = build_feature_matrix(base_df, _CONFIG_A).to_numpy()
    delta_b = A @ c_true
    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))

    segments: list[Segment] = [
        Segment(maneuver="steady", heading="N", start_idx=0, end_idx=20),
        Segment(maneuver="pitch", heading="E", start_idx=20, end_idx=40),
        Segment(maneuver="roll", heading="S", start_idx=40, end_idx=60),
    ]
    result = calibrate(df, segments, _CONFIG_A)
    report = compute_fom_report(df, segments, result)

    expected_maneuvers: set[ManeuverType] = {"steady", "pitch", "roll"}
    assert set(report.per_maneuver.keys()) == expected_maneuvers


def test_per_heading_keys_match_segments() -> None:
    """per_heading keys should exactly match heading types in segments."""
    c_true = np.array([1.0, -2.0, 0.5])
    rng = np.random.default_rng(1)
    base_df = _make_synthetic_df(60, rng)
    A = build_feature_matrix(base_df, _CONFIG_A).to_numpy()
    delta_b = A @ c_true
    df = base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))

    segments: list[Segment] = [
        Segment(maneuver="steady", heading="N", start_idx=0, end_idx=20),
        Segment(maneuver="steady", heading="E", start_idx=20, end_idx=40),
        Segment(maneuver="steady", heading="W", start_idx=40, end_idx=60),
    ]
    result = calibrate(df, segments, _CONFIG_A)
    report = compute_fom_report(df, segments, result)

    expected_headings: set[HeadingType] = {"N", "E", "W"}
    assert set(report.per_heading.keys()) == expected_headings


def test_to_json_valid() -> None:
    """to_json() should produce valid JSON with expected top-level keys."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A, noise_std=0.5)
    result = calibrate(df, segments, _CONFIG_A)
    report = compute_fom_report(df, segments, result)

    parsed = json.loads(report.to_json())
    assert "per_maneuver" in parsed
    assert "per_heading" in parsed
    assert "improvement_ratio" in parsed


def test_missing_delta_b_raises() -> None:
    """compute_fom_report should raise ValueError when COL_DELTA_B is absent."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)

    df_no_delta = df.drop(COL_DELTA_B)
    with pytest.raises(ValueError, match=COL_DELTA_B):
        compute_fom_report(df_no_delta, segments, result)


def test_empty_segments_raises() -> None:
    """compute_fom_report should raise ValueError when segments list is empty."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, _ = _make_synthetic_data(c_true, _CONFIG_A)
    empty_result = CalibrationResult(
        coefficients=np.zeros(3),
        residuals=np.array([]),
        condition_number=1.0,
        n_terms=3,
    )
    with pytest.raises(ValueError, match="segments must be non-empty"):
        compute_fom_report(df, [], empty_result)


def test_residuals_length_mismatch_raises() -> None:
    """compute_fom_report should raise ValueError when residuals length mismatches."""
    c_true = np.array([1.0, -2.0, 0.5])
    df, segments = _make_synthetic_data(c_true, _CONFIG_A)
    result = calibrate(df, segments, _CONFIG_A)

    wrong_result = CalibrationResult(
        coefficients=result.coefficients,
        residuals=result.residuals[:-1],  # one element short
        condition_number=result.condition_number,
        n_terms=result.n_terms,
    )
    with pytest.raises(ValueError, match="does not match"):
        compute_fom_report(df, segments, wrong_result)
