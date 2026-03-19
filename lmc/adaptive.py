"""Adaptive maneuver-based compensation with per-maneuver coefficient blending."""

from __future__ import annotations

from dataclasses import dataclass

from lmc.calibration import CalibrationResult


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
