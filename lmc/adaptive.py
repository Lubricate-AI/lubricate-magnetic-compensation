"""Adaptive maneuver-based compensation with per-maneuver coefficient blending."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from lmc.calibration import CalibrationResult, calibrate
from lmc.config import PipelineConfig
from lmc.segmentation import Segment


@dataclass(frozen=True)
class AdaptiveCalibrationResult:
    """Per-maneuver calibration results used for adaptive blending.

    Attributes
    ----------
    pitch:
        Coefficients fitted on pitch maneuver segments only.
    roll:
        Coefficients fitted on roll maneuver segments only.
    yaw:
        Coefficients fitted on yaw maneuver segments only.
    baseline:
        Coefficients fitted on steady (non-maneuver) segments.
    n_terms:
        Number of model terms (must match across all sub-results).
    """

    pitch: CalibrationResult
    roll: CalibrationResult
    yaw: CalibrationResult
    baseline: CalibrationResult
    n_terms: int


def calibrate_adaptive_maneuvers(
    df: pl.DataFrame,
    segments: list[Segment],
    config: PipelineConfig,
) -> AdaptiveCalibrationResult:
    """Fit separate Tolles-Lawson coefficients for each maneuver type.

    Parameters
    ----------
    df:
        Full calibration DataFrame including ``COL_DELTA_B``.
    segments:
        Labeled segments covering all four maneuver types:
        ``"steady"``, ``"pitch"``, ``"roll"``, ``"yaw"``.
    config:
        Pipeline configuration (same ``model_terms`` used for all fits).

    Returns
    -------
    AdaptiveCalibrationResult
        Four ``CalibrationResult`` objects, one per maneuver type.

    Raises
    ------
    ValueError
        If any of the four required maneuver types has no segments.
    """
    pitch_segs = [s for s in segments if s.maneuver == "pitch"]
    roll_segs = [s for s in segments if s.maneuver == "roll"]
    yaw_segs = [s for s in segments if s.maneuver == "yaw"]
    baseline_segs = [s for s in segments if s.maneuver == "steady"]

    for name, segs in [
        ("pitch", pitch_segs),
        ("roll", roll_segs),
        ("yaw", yaw_segs),
        ("steady", baseline_segs),
    ]:
        if not segs:
            raise ValueError(
                f"No '{name}' segments found. "
                f"calibrate_adaptive_maneuvers requires at least one segment "
                f"for each of: pitch, roll, yaw, steady."
            )

    pitch_result = calibrate(df, pitch_segs, config)
    roll_result = calibrate(df, roll_segs, config)
    yaw_result = calibrate(df, yaw_segs, config)
    baseline_result = calibrate(df, baseline_segs, config)

    return AdaptiveCalibrationResult(
        pitch=pitch_result,
        roll=roll_result,
        yaw=yaw_result,
        baseline=baseline_result,
        n_terms=pitch_result.n_terms,
    )
