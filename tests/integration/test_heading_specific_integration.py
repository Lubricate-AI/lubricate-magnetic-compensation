"""Integration test: heading-specific calibration and compensation pipeline."""

from __future__ import annotations

import numpy as np

from lmc import (
    HeadingCalibrationResult,
    PipelineConfig,
    calibrate_per_heading,
    compensate_heading_specific,
    compute_fom_report,
    segment_fom,
)
from lmc.columns import COL_TMI_COMPENSATED
from lmc.earth_field import compute_interference
from tests.integration.synthetic import make_fom_dataframe


def _make_pipeline_data(model_terms: str = "b", seed: int = 42):
    """Return (df_with_delta_b, segments, config) for a 4-heading FOM flight."""
    rng = np.random.default_rng(seed)
    n_terms_map = {"a": 3, "b": 9, "c": 18}
    n_terms = n_terms_map[model_terms]
    c_true = rng.standard_normal(n_terms)

    config = PipelineConfig(
        model_terms=model_terms,
        segment_label_col="segment",
        use_heading_specific_calibration=True,
    )

    df = make_fom_dataframe(c_true, n_rows_per_block=60, noise_std=0.5, seed=seed)
    df = df.with_columns(compute_interference(df, config))

    segments = segment_fom(df, config)
    return df, segments, config, c_true


def test_calibrate_per_heading_returns_four_heading_keys() -> None:
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    assert isinstance(result, HeadingCalibrationResult)
    assert set(result.per_heading.keys()) == {"N", "E", "S", "W"}


def test_per_heading_vif_available_for_all_headings() -> None:
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    for heading in ("N", "E", "S", "W"):
        assert heading in result.per_heading_vif
        vif = result.per_heading_vif[heading]
        assert vif.shape[0] == result.per_heading[heading].n_terms


def test_compensate_heading_specific_produces_tmi_compensated() -> None:
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    out_df = compensate_heading_specific(df, result, config)
    assert COL_TMI_COMPENSATED in out_df.columns
    assert len(out_df) == len(df)


def test_heading_specific_improvement_ratio_exceeds_one() -> None:
    """After heading-specific compensation, variance should be reduced."""
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)

    north_segs = [s for s in segments if s.heading == "N"]
    north_result = result.per_heading["N"]

    report = compute_fom_report(df, north_segs, north_result)
    assert report.improvement_ratio > 1.0, (
        f"Expected improvement_ratio > 1.0, got {report.improvement_ratio}"
    )


def test_condition_number_available_per_heading() -> None:
    """Each per-heading CalibrationResult should carry a condition number."""
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    for heading, cal in result.per_heading.items():
        assert cal.condition_number > 0.0, f"Heading {heading}: bad condition number"
        assert np.isfinite(cal.condition_number), (
            f"Heading {heading}: inf condition number"
        )
