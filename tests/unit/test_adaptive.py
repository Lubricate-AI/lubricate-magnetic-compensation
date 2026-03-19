"""Unit tests for lmc.adaptive."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lmc.adaptive import (
    AdaptiveCalibrationResult,
    _rolling_variance,  # pyright: ignore[reportPrivateUsage]
    calibrate_adaptive_maneuvers,
    compensate_adaptive,
)
from lmc.calibration import CalibrationResult, calibrate
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
from lmc.compensation import compensate
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import Segment

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_cal_result(n_terms: int) -> CalibrationResult:
    """Build a minimal CalibrationResult for use in tests."""
    return CalibrationResult(
        coefficients=np.ones(n_terms, dtype=np.float64),
        residuals=np.zeros(10, dtype=np.float64),
        condition_number=1.0,
        n_terms=n_terms,
    )


# ---------------------------------------------------------------------------
# AdaptiveCalibrationResult tests
# ---------------------------------------------------------------------------


def test_adaptive_result_stores_four_sub_results() -> None:
    r = _dummy_cal_result(3)
    result = AdaptiveCalibrationResult(pitch=r, roll=r, yaw=r, baseline=r, n_terms=3)
    assert result.pitch is r
    assert result.roll is r
    assert result.yaw is r
    assert result.baseline is r


def test_adaptive_result_stores_n_terms() -> None:
    r = _dummy_cal_result(9)
    result = AdaptiveCalibrationResult(pitch=r, roll=r, yaw=r, baseline=r, n_terms=9)
    assert result.n_terms == 9


def test_adaptive_result_is_frozen() -> None:
    r = _dummy_cal_result(3)
    result = AdaptiveCalibrationResult(pitch=r, roll=r, yaw=r, baseline=r, n_terms=3)
    with pytest.raises((AttributeError, TypeError)):
        result.n_terms = 9  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Synthetic data helpers (adapted from test_calibration.py)
# ---------------------------------------------------------------------------


def _make_base_df(n_rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """DataFrame with random unit B vectors but no delta_B."""
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
            COL_ALT: np.full(n_rows, 0.3),
            COL_BTOTAL: np.full(n_rows, b_total),
            COL_BX: bx,
            COL_BY: by,
            COL_BZ: bz,
        }
    )


def _make_adaptive_calibration_data(
    n_rows_each: int = 40,
    seed: int = 0,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build (df, segments) with one segment per maneuver type."""
    config = PipelineConfig(model_terms="a")
    rng = np.random.default_rng(seed)
    c_true = np.array([1.0, -2.0, 0.5])

    blocks: list[pl.DataFrame] = []
    segs: list[Segment] = []
    offset = 0
    for maneuver in ["steady", "pitch", "roll", "yaw"]:
        block = _make_base_df(n_rows_each, rng)
        A = build_feature_matrix(block, config).to_numpy()
        delta_b = A @ c_true + rng.normal(0, 0.01, n_rows_each)
        block = block.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))
        blocks.append(block)
        segs.append(
            Segment(
                maneuver=maneuver,  # type: ignore[arg-type]
                heading="N",  # type: ignore[arg-type]
                start_idx=offset,
                end_idx=offset + n_rows_each,
            )
        )
        offset += n_rows_each

    df: pl.DataFrame = pl.concat(blocks)
    return df, segs


# ---------------------------------------------------------------------------
# calibrate_adaptive_maneuvers tests
# ---------------------------------------------------------------------------


def test_adaptive_calibration_returns_correct_type() -> None:
    df, segments = _make_adaptive_calibration_data()
    config = PipelineConfig(model_terms="a")
    result = calibrate_adaptive_maneuvers(df, segments, config)
    assert isinstance(result, AdaptiveCalibrationResult)


def test_adaptive_calibration_n_terms_matches_config() -> None:
    df, segments = _make_adaptive_calibration_data()
    config = PipelineConfig(model_terms="a")
    result = calibrate_adaptive_maneuvers(df, segments, config)
    assert result.n_terms == 3
    assert result.pitch.n_terms == 3
    assert result.roll.n_terms == 3
    assert result.yaw.n_terms == 3
    assert result.baseline.n_terms == 3


def test_adaptive_calibration_raises_if_pitch_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_pitch = [s for s in segments if s.maneuver != "pitch"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="pitch"):
        calibrate_adaptive_maneuvers(df, non_pitch, config)


def test_adaptive_calibration_raises_if_roll_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_roll = [s for s in segments if s.maneuver != "roll"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="roll"):
        calibrate_adaptive_maneuvers(df, non_roll, config)


def test_adaptive_calibration_raises_if_yaw_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_yaw = [s for s in segments if s.maneuver != "yaw"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="yaw"):
        calibrate_adaptive_maneuvers(df, non_yaw, config)


def test_adaptive_calibration_raises_if_steady_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_steady = [s for s in segments if s.maneuver != "steady"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="steady"):
        calibrate_adaptive_maneuvers(df, non_steady, config)


# ---------------------------------------------------------------------------
# _rolling_variance tests
# ---------------------------------------------------------------------------


def test_rolling_variance_all_zeros_returns_zeros() -> None:
    arr = np.zeros(20, dtype=np.float64)
    result = _rolling_variance(arr, window=5)
    np.testing.assert_array_equal(result, np.zeros(20))


def test_rolling_variance_constant_returns_zeros() -> None:
    arr = np.full(20, 3.14, dtype=np.float64)
    result = _rolling_variance(arr, window=5)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_rolling_variance_output_shape_matches_input() -> None:
    arr = np.arange(30, dtype=np.float64)
    result = _rolling_variance(arr, window=10)
    assert result.shape == (30,)


def test_rolling_variance_single_element_window_is_zero() -> None:
    """Variance of a single value is 0."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _rolling_variance(arr, window=1)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_rolling_variance_known_values() -> None:
    """After window is full, variance matches numpy for a known window."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = _rolling_variance(arr, window=3)
    # Index 4: window is [3, 4, 5] → variance = np.var([3,4,5])
    expected_last = float(np.var(np.array([3.0, 4.0, 5.0])))
    np.testing.assert_allclose(result[4], expected_last, atol=1e-12)
    # Index 2: window is [1, 2, 3] → variance = np.var([1,2,3])
    expected_idx2 = float(np.var(np.array([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(result[2], expected_idx2, atol=1e-12)


# ---------------------------------------------------------------------------
# compensate_adaptive tests
# ---------------------------------------------------------------------------


def _make_full_adaptive_result(
    config: PipelineConfig,
    seed: int = 42,
) -> tuple[pl.DataFrame, AdaptiveCalibrationResult]:
    """Build a survey DataFrame + AdaptiveCalibrationResult from synthetic data."""
    df_cal, segments = _make_adaptive_calibration_data(n_rows_each=60, seed=seed)
    result = calibrate_adaptive_maneuvers(df_cal, segments, config)
    # Build a survey df (no delta_B required — compensation only needs B columns)
    rng = np.random.default_rng(seed + 1)
    survey = _make_base_df(50, rng)
    return survey, result


def test_compensate_adaptive_returns_tmi_compensated_column() -> None:
    config = PipelineConfig(model_terms="a")
    df, result = _make_full_adaptive_result(config)
    out = compensate_adaptive(df, result, config)
    assert COL_TMI_COMPENSATED in out.columns


def test_compensate_adaptive_row_count_preserved() -> None:
    config = PipelineConfig(model_terms="a")
    df, result = _make_full_adaptive_result(config)
    out = compensate_adaptive(df, result, config)
    assert len(out) == len(df)


def test_compensate_adaptive_tmi_values_finite() -> None:
    config = PipelineConfig(model_terms="a")
    df, result = _make_full_adaptive_result(config)
    out = compensate_adaptive(df, result, config)
    assert out[COL_TMI_COMPENSATED].is_finite().all()


def test_compensate_adaptive_terms_mismatch_raises() -> None:
    """Using model_terms='b' (9 terms) with an 'a' result (3 terms) must fail."""
    config_a = PipelineConfig(model_terms="a")
    config_b = PipelineConfig(model_terms="b")
    df, result_a = _make_full_adaptive_result(config_a)
    with pytest.raises(ValueError, match="9"):
        compensate_adaptive(df, result_a, config_b)


def test_compensate_adaptive_identical_coefs_matches_standard() -> None:
    """When all four coefficient sets are identical, adaptive == standard."""
    config = PipelineConfig(model_terms="a")
    rng = np.random.default_rng(99)
    c_true = np.array([1.0, -2.0, 0.5])

    # Build a df with delta_B so we can call calibrate()
    block = _make_base_df(80, rng)
    A = build_feature_matrix(block, config).to_numpy()
    delta_b = A @ c_true
    df_cal = block.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))

    from lmc.segmentation import Segment

    single_seg = [
        Segment(
            maneuver="steady",
            heading="N",  # type: ignore[arg-type]
            start_idx=0,
            end_idx=80,
        )
    ]
    single_result = calibrate(df_cal, single_seg, config)

    # Wrap the same result in all four slots
    adaptive_result = AdaptiveCalibrationResult(
        pitch=single_result,
        roll=single_result,
        yaw=single_result,
        baseline=single_result,
        n_terms=single_result.n_terms,
    )

    survey = _make_base_df(50, np.random.default_rng(7))
    out_standard = compensate(survey, single_result, config)
    out_adaptive = compensate_adaptive(survey, adaptive_result, config)

    np.testing.assert_allclose(
        out_adaptive[COL_TMI_COMPENSATED].to_numpy(),
        out_standard[COL_TMI_COMPENSATED].to_numpy(),
        atol=1e-10,
    )
