"""Adaptive maneuver-based compensation with per-maneuver coefficient blending."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np  # noqa: F401
import numpy.typing as npt  # noqa: F401
import polars as pl  # noqa: F401

from lmc.calibration import CalibrationResult, calibrate  # noqa: F401
from lmc.columns import (  # noqa: F401
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_TMI_COMPENSATED,
)
from lmc.config import PipelineConfig  # noqa: F401
from lmc.features import build_feature_matrix  # noqa: F401
from lmc.segmentation import Segment  # noqa: F401


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
