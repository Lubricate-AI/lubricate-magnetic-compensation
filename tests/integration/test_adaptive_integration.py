"""Integration test for adaptive maneuver-based compensation."""

from __future__ import annotations

import numpy as np
import polars as pl

from lmc import (
    COL_BTOTAL,
    COL_TMI_COMPENSATED,
    AdaptiveCalibrationResult,
    PipelineConfig,
    calibrate_adaptive_maneuvers,
    compensate_adaptive,
    compute_interference,
    segment_fom,
    validate_dataframe,
)
from tests.integration.synthetic import make_fom_dataframe


def test_adaptive_compensation_produces_valid_output() -> None:
    """Adaptive pipeline produces finite compensated TMI on synthetic FOM data."""
    c_true = np.arange(1, 19, dtype=float) * 0.5  # 18 terms → model_terms="c"

    df_fom = make_fom_dataframe(c_true, noise_std=0.05)

    config = PipelineConfig(
        model_terms="c",
        earth_field_method="steady_mean",
        segment_label_col="segment",
    )

    df_fom = validate_dataframe(df_fom)
    segments = segment_fom(df_fom, config)

    # Build steady_mask for earth-field baseline
    mask_arr = np.zeros(df_fom.height, dtype=bool)
    for seg in segments:
        if seg.maneuver == "steady":
            mask_arr[seg.start_idx : seg.end_idx] = True
    steady_mask = pl.Series(mask_arr, dtype=pl.Boolean)

    delta_b = compute_interference(df_fom, config, steady_mask=steady_mask)
    df_fom = df_fom.with_columns(delta_b)

    # Calibrate
    adaptive_result = calibrate_adaptive_maneuvers(df_fom, segments, config)
    assert isinstance(adaptive_result, AdaptiveCalibrationResult)
    assert adaptive_result.n_terms == 18

    # Compensate on a single clean segment to avoid cross-segment derivative artefacts
    pitch_n_seg = next(
        seg for seg in segments if seg.maneuver == "pitch" and seg.heading == "N"
    )
    pitch_n = df_fom.slice(
        pitch_n_seg.start_idx, pitch_n_seg.end_idx - pitch_n_seg.start_idx
    )

    out = compensate_adaptive(pitch_n, adaptive_result, config)

    # Output validity checks
    assert COL_TMI_COMPENSATED in out.columns
    assert len(out) == len(pitch_n)
    assert out[COL_TMI_COMPENSATED].is_finite().all()

    # Compensation should reduce variance (signal is cleaner after removal)
    std_raw = out[COL_BTOTAL].std()
    std_comp = out[COL_TMI_COMPENSATED].std()
    assert isinstance(std_raw, float) and isinstance(std_comp, float)
    assert std_raw > std_comp, (
        f"Expected compensation to reduce std: raw={std_raw:.4f}, comp={std_comp:.4f}"
    )
