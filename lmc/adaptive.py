"""Adaptive maneuver-based compensation with per-maneuver coefficient blending."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.calibration import CalibrationResult, calibrate
from lmc.columns import (
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_TMI_COMPENSATED,
)
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
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


def _rolling_variance(
    arr: npt.NDArray[np.float64], window: int
) -> npt.NDArray[np.float64]:
    """Compute causal rolling population variance.

    For index ``i``, variance is computed over ``arr[max(0, i-window+1) : i+1]``.
    This means the first ``window-1`` samples use a shorter effective window
    (causal edge handling — no look-ahead, no padding).

    Parameters
    ----------
    arr:
        1-D input array.
    window:
        Maximum number of samples in the rolling window.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``arr`` with rolling variances.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(np.var(arr[max(0, i - window + 1) : i + 1]))
    return out


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
                "for each of: pitch, roll, yaw, steady."
            )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pitch_result = calibrate(df, pitch_segs, config)
        roll_result = calibrate(df, roll_segs, config)
        yaw_result = calibrate(df, yaw_segs, config)
        baseline_result = calibrate(df, baseline_segs, config)

    for name, result in [
        ("pitch", pitch_result),
        ("roll", roll_result),
        ("yaw", yaw_result),
        ("steady", baseline_result),
    ]:
        if result.condition_number > config.condition_number_threshold:
            warnings.warn(
                f"Adaptive calibration: '{name}' maneuver is ill-conditioned "
                f"(condition number {result.condition_number:.3e} exceeds threshold "
                f"{config.condition_number_threshold:.3e}). Coefficients may be "
                "unstable; compensate_adaptive() will suppress this maneuver's weight.",
                stacklevel=2,
            )

    return AdaptiveCalibrationResult(
        pitch=pitch_result,
        roll=roll_result,
        yaw=yaw_result,
        baseline=baseline_result,
        n_terms=pitch_result.n_terms,
    )


def compensate_adaptive(
    df: pl.DataFrame,
    result: AdaptiveCalibrationResult,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Apply compensation with maneuver-adaptive coefficient blending.

    Algorithm
    ---------
    1. Build feature matrix ``A`` from fluxgate columns (shape ``n × n_terms``).
    2. Compute direction cosines from raw fluxgate components.
    3. Compute rolling variance of each direction cosine to detect maneuver
       intensity (pitch ~ cos_x, roll ~ cos_y, yaw ~ cos_z).
    4. Normalise intensities + ``maneuver_baseline_weight`` to unit-sum weights.
    5. Blend four interference vectors: ``w_pitch*(A@c_pitch) + ...``
    6. Return ``df`` with column ``COL_TMI_COMPENSATED = B_total - interference``.

    Parameters
    ----------
    df:
        Survey DataFrame containing all required magnetometer columns.
    result:
        Adaptive calibration result from ``calibrate_adaptive_maneuvers()``.
    config:
        Pipeline configuration.  ``model_terms`` must match those used during
        calibration.  ``maneuver_detection_window`` and
        ``maneuver_baseline_weight`` control blending behaviour.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with added column ``COL_TMI_COMPENSATED``.

    Raises
    ------
    ValueError
        If feature matrix column count does not match ``result.n_terms``.
    """
    feature_matrix = build_feature_matrix(df, config)
    A = feature_matrix.to_numpy()

    if A.shape[1] != result.n_terms:
        raise ValueError(
            f"Feature matrix has {A.shape[1]} columns but AdaptiveCalibrationResult "
            f"has {result.n_terms} terms. Ensure the same model_terms are used "
            "for both calibration and compensation."
        )

    # --- Direction cosines for maneuver detection ---
    bx = np.asarray(df[COL_BX].to_numpy(), dtype=np.float64)
    by = np.asarray(df[COL_BY].to_numpy(), dtype=np.float64)
    bz = np.asarray(df[COL_BZ].to_numpy(), dtype=np.float64)
    b_flux_mag = np.sqrt(bx**2 + by**2 + bz**2)
    cos_x = bx / b_flux_mag
    cos_y = by / b_flux_mag
    cos_z = bz / b_flux_mag

    # --- Rolling variance → maneuver intensities ---
    window = config.maneuver_detection_window
    pitch_intensity = _rolling_variance(cos_x, window)
    roll_intensity = _rolling_variance(cos_y, window)
    yaw_intensity = _rolling_variance(cos_z, window)

    # --- Suppress weights for ill-conditioned maneuver types ---
    threshold = config.condition_number_threshold
    if result.pitch.condition_number > threshold:
        pitch_intensity = np.zeros_like(pitch_intensity)
    if result.roll.condition_number > threshold:
        roll_intensity = np.zeros_like(roll_intensity)
    if result.yaw.condition_number > threshold:
        yaw_intensity = np.zeros_like(yaw_intensity)

    # --- Normalise to blend weights (sum == 1 at every sample) ---
    baseline_w = config.maneuver_baseline_weight
    total = pitch_intensity + roll_intensity + yaw_intensity + baseline_w
    w_pitch = pitch_intensity / total
    w_roll = roll_intensity / total
    w_yaw = yaw_intensity / total
    w_baseline = baseline_w / total

    # --- Per-maneuver interference vectors (broadcast-efficient) ---
    interf_pitch = A @ result.pitch.coefficients
    interf_roll = A @ result.roll.coefficients
    interf_yaw = A @ result.yaw.coefficients
    interf_baseline = A @ result.baseline.coefficients

    interference = (
        w_pitch * interf_pitch
        + w_roll * interf_roll
        + w_yaw * interf_yaw
        + w_baseline * interf_baseline
    )

    b_total = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64)
    tmi_comp = b_total - interference

    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
