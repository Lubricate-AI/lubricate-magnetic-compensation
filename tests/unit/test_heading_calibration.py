"""Unit tests for lmc.heading_calibration."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

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
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.heading_calibration import calibrate_per_heading
from lmc.segmentation import Segment


def _make_df_with_delta_b(
    n_rows: int,
    rng: np.random.Generator,
    config: PipelineConfig,
) -> pl.DataFrame:
    raw = rng.standard_normal((n_rows, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms
    b_total = 50_000.0
    bx = cosines[:, 0] * b_total
    by = cosines[:, 1] * b_total
    bz = cosines[:, 2] * b_total
    df = pl.DataFrame(
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
    A: npt.NDArray[np.float64] = build_feature_matrix(df, config).to_numpy()
    c_true: npt.NDArray[np.float64] = rng.standard_normal(A.shape[1])  # type: ignore[assignment]
    delta_b: npt.NDArray[np.float64] = (A @ c_true).astype(np.float64)  # type: ignore[assignment]
    return df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))


def _make_multi_heading_data(
    config: PipelineConfig,
    n_per_heading: int = 60,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build a df + segments list with four distinct heading groups."""
    rng = np.random.default_rng(seed)
    headings = ["N", "E", "S", "W"]
    blocks: list[pl.DataFrame] = []
    segments: list[Segment] = []
    offset = 0
    for h in headings:
        block = _make_df_with_delta_b(n_per_heading, rng, config)
        blocks.append(block)
        segments.append(
            Segment(  # type: ignore[arg-type]
                maneuver="steady",
                heading=h,  # pyright: ignore[reportArgumentType]
                start_idx=offset,
                end_idx=offset + n_per_heading,
            )
        )
        offset += n_per_heading
    df = pl.concat(blocks)
    return df, segments


_CONFIG_A = PipelineConfig(model_terms="a")


def test_calibrate_per_heading_returns_result_for_each_heading() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    assert set(result.per_heading.keys()) == {"N", "E", "S", "W"}


def test_calibrate_per_heading_each_result_is_calibration_result() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    for cal in result.per_heading.values():
        assert isinstance(cal, CalibrationResult)


def test_calibrate_per_heading_vif_shape_matches_n_terms() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    for heading, vif_arr in result.per_heading_vif.items():
        expected_n_terms = result.per_heading[heading].n_terms
        assert vif_arr.shape == (expected_n_terms,), (
            f"Heading {heading}: VIF shape {vif_arr.shape} != ({expected_n_terms},)"
        )


def test_calibrate_per_heading_vif_all_positive() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    for heading, vif_arr in result.per_heading_vif.items():
        finite = vif_arr[np.isfinite(vif_arr)]
        assert np.all(finite >= 1.0), f"Heading {heading}: VIF must be >= 1"


def test_calibrate_per_heading_raises_on_empty_segments() -> None:
    df, _ = _make_multi_heading_data(_CONFIG_A)
    with pytest.raises(ValueError, match="segments must be non-empty"):
        calibrate_per_heading(df, [], _CONFIG_A)


def test_calibrate_per_heading_single_heading() -> None:
    """Single-heading data should produce a result with one key."""
    rng = np.random.default_rng(99)
    config = PipelineConfig(model_terms="a")
    df = _make_df_with_delta_b(60, rng, config)
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=60)]  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    result = calibrate_per_heading(df, segments, config)
    assert set(result.per_heading.keys()) == {"N"}
    assert set(result.per_heading_vif.keys()) == {"N"}


def test_calibrate_per_heading_multiple_segments_same_heading() -> None:
    """Two segments with the same heading should be pooled and return one key."""
    rng = np.random.default_rng(77)
    config = PipelineConfig(model_terms="a")
    df = _make_df_with_delta_b(120, rng, config)
    segments = [
        Segment(maneuver="steady", heading="N", start_idx=0, end_idx=60),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
        Segment(maneuver="pitch", heading="N", start_idx=60, end_idx=120),  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    ]
    result = calibrate_per_heading(df, segments, config)
    assert set(result.per_heading.keys()) == {"N"}
    assert set(result.per_heading_vif.keys()) == {"N"}
    # The model was trained on 120 rows pooled from both segments.
    assert result.per_heading["N"].residuals.shape[0] == 120
