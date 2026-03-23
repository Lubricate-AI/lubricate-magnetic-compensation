"""End-to-end integration tests for the Tolles-Lawson pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import polars as pl

from lmc import (
    COL_BTOTAL,
    COL_TMI_COMPENSATED,
    REQUIRED_COLUMNS,
    PipelineConfig,
    calibrate,
    compensate,
    compute_fom_report,
    compute_interference,
    segment_fom,
    validate_dataframe,
)
from tests.integration.synthetic import make_fom_dataframe


def test_calibrate_compensate_full_pipeline() -> None:
    """Full pipeline: calibrate on synthetic FOM data, then compensate."""
    c_true = np.arange(1, 19, dtype=float) * 0.5  # 18 terms

    df_fom = make_fom_dataframe(c_true, noise_std=0.05)

    config = PipelineConfig(
        model_terms="c",
        earth_field_method="steady_mean",
        segment_label_col="segment",
    )

    df_fom = validate_dataframe(df_fom)
    segments = segment_fom(df_fom, config)

    # Build steady_mask from steady segments
    mask_arr = np.zeros(df_fom.height, dtype=bool)
    for seg in segments:
        if seg.maneuver == "steady":
            mask_arr[seg.start_idx : seg.end_idx] = True
    steady_mask = pl.Series(mask_arr, dtype=pl.Boolean)

    delta_b = compute_interference(df_fom, config, steady_mask=steady_mask)
    df_fom = df_fom.with_columns(delta_b)

    result = calibrate(df_fom, segments, config)
    report = compute_fom_report(df_fom, segments, result)

    assert report.improvement_ratio > 10.0

    # Verify compensation on a single smooth block (pitch_N).
    # The full FOM dataset has 90° heading jumps between groups; at those
    # boundaries np.gradient produces large cross-segment derivatives that
    # corrupt the eddy-current correction. Within a single block the feature
    # matrix is identical to what calibration used, so compensation is clean.
    pitch_n_seg = next(
        seg for seg in segments if seg.maneuver == "pitch" and seg.heading == "N"
    )
    pitch_n = df_fom.slice(
        pitch_n_seg.start_idx, pitch_n_seg.end_idx - pitch_n_seg.start_idx
    )
    df_compensated = compensate(pitch_n, result, config)
    std_raw = df_compensated[COL_BTOTAL].std()
    std_comp = df_compensated[COL_TMI_COMPENSATED].std()
    assert isinstance(std_raw, float) and isinstance(std_comp, float)
    assert std_raw / std_comp > 10.0


def test_cli_calibrate_compensate(tmp_path: Path) -> None:
    """CLI round-trip: calibrate then compensate via Typer CliRunner."""
    import json

    from typer.testing import CliRunner

    from lmc.cli.commands import app

    c_true = np.arange(1, 19, dtype=float) * 0.5

    # Write synthetic FOM CSV
    df_fom = make_fom_dataframe(c_true, noise_std=0.05)
    fom_csv = tmp_path / "fom.csv"
    df_fom.write_csv(fom_csv)

    runner = CliRunner()
    coef_json = tmp_path / "coefs.json"

    cal_result = runner.invoke(
        app,
        [
            "calibrate",
            str(fom_csv),
            "--output",
            str(coef_json),
            "--earth-field-method",
            "steady_mean",
            "--segment-label-col",
            "segment",
            "--model-terms",
            "c",
        ],
    )
    assert cal_result.exit_code == 0, cal_result.output
    assert coef_json.exists()

    # Verify singular_values is written to JSON output.
    coef_data: dict[str, object] = json.loads(coef_json.read_text())
    assert "singular_values" in coef_data, (
        "Expected 'singular_values' key in calibration JSON output"
    )
    sv = coef_data["singular_values"]
    assert isinstance(sv, list), f"Expected list, got {type(sv).__name__}"
    sv_list = cast(list[float], sv)
    n_terms = coef_data["n_terms"]
    assert isinstance(n_terms, int)
    assert len(sv_list) == n_terms, (
        f"Expected {n_terms} singular values, got {len(sv_list)}"
    )
    assert all(isinstance(v, float) for v in sv_list), (
        "Expected all singular_values entries to be floats"
    )

    # Write synthetic survey CSV (REQUIRED_COLUMNS only)
    survey_csv = tmp_path / "survey.csv"
    df_fom.select(list(REQUIRED_COLUMNS)).write_csv(survey_csv)

    out_csv = tmp_path / "out.csv"
    comp_result = runner.invoke(
        app,
        [
            "compensate",
            str(survey_csv),
            "--coefficients",
            str(coef_json),
            "--output",
            str(out_csv),
        ],
    )
    assert comp_result.exit_code == 0, comp_result.output
    assert out_csv.exists()
