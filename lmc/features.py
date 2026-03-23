"""Tolles-Lawson A-matrix (design matrix) construction from magnetometer data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.columns import (
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_COS_X,
    COL_COS_X2,
    COL_COS_X_DCOS_X,
    COL_COS_X_DCOS_Y,
    COL_COS_X_DCOS_Z,
    COL_COS_XY,
    COL_COS_XZ,
    COL_COS_Y,
    COL_COS_Y2,
    COL_COS_Y_DCOS_X,
    COL_COS_Y_DCOS_Y,
    COL_COS_Y_DCOS_Z,
    COL_COS_YZ,
    COL_COS_Z,
    COL_COS_Z2,
    COL_COS_Z_DCOS_X,
    COL_COS_Z_DCOS_Y,
    COL_COS_Z_DCOS_Z,
    COL_DCOS_X,
    COL_DCOS_Y,
    COL_DCOS_Z,
    COL_PITCH_RATE,
    COL_ROLL_RATE,
    COL_TIME,
    COL_YAW_RATE,
)
from lmc.config import PipelineConfig
from lmc.validation import validate_dataframe


def _direction_cosines(
    df: pl.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute direction cosines from fluxgate magnetometer readings.

    Returns (cos_x, cos_y, cos_z) = (B_x/|B|, B_y/|B|, B_z/|B|).
    """
    b_x = np.asarray(df[COL_BX].to_numpy(), dtype=np.float64)
    b_y = np.asarray(df[COL_BY].to_numpy(), dtype=np.float64)
    b_z = np.asarray(df[COL_BZ].to_numpy(), dtype=np.float64)

    b_magnitude = np.sqrt(b_x**2 + b_y**2 + b_z**2)

    if np.any(b_magnitude <= 0.0):
        raise ValueError("Fluxgate magnitude must be strictly positive for all rows.")

    cos_x = b_x / b_magnitude
    cos_y = b_y / b_magnitude
    cos_z = b_z / b_magnitude

    return (
        np.asarray(cos_x, dtype=np.float64),
        np.asarray(cos_y, dtype=np.float64),
        np.asarray(cos_z, dtype=np.float64),
    )


def _cosine_derivatives(
    cos_x: npt.NDArray[np.float64],
    cos_y: npt.NDArray[np.float64],
    cos_z: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    *,
    causal: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute time derivatives of direction cosines.

    When ``causal=False`` (default), uses ``np.gradient`` (central differences)
    — each row may depend on its neighbours.

    When ``causal=True``, uses backward differences so that row *i* depends only
    on samples *i* and *i-1* (no look-ahead).  The first row is padded by
    replicating the first backward-difference value so the output length matches
    the input.  Use this path when ``config.use_cv=True`` to avoid train/test
    leakage at ``TimeSeriesSplit`` fold boundaries.

    Uses explicit time coordinates to handle non-uniform sampling.
    Returns (dcos_x/dt, dcos_y/dt, dcos_z/dt).
    """
    if causal:
        dt = np.diff(time)  # shape (n-1,)
        diffs_x = np.diff(cos_x) / dt
        diffs_y = np.diff(cos_y) / dt
        diffs_z = np.diff(cos_z) / dt
        # Row 0 has no prior sample, so pad with the first backward difference.
        # This mirrors np.gradient's own left-edge convention (forward diff at row 0)
        # and avoids NaNs, which simplifies downstream use (sklearn, etc.).
        return (
            np.asarray(np.concatenate([[diffs_x[0]], diffs_x]), dtype=np.float64),
            np.asarray(np.concatenate([[diffs_y[0]], diffs_y]), dtype=np.float64),
            np.asarray(np.concatenate([[diffs_z[0]], diffs_z]), dtype=np.float64),
        )

    dcos_x = np.asarray(np.gradient(cos_x, time), dtype=np.float64)
    dcos_y = np.asarray(np.gradient(cos_y, time), dtype=np.float64)
    dcos_z = np.asarray(np.gradient(cos_z, time), dtype=np.float64)
    return dcos_x, dcos_y, dcos_z


def _imu_rates(
    df: pl.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract IMU angular rate channels as direct substitutes for ḋcos signals.

    Returns (roll_rate, pitch_rate, yaw_rate) as float64 arrays.
    Raises ValueError if any of the three columns are absent.
    """
    missing = [
        col
        for col in (COL_ROLL_RATE, COL_PITCH_RATE, COL_YAW_RATE)
        if col not in df.columns
    ]
    if missing:
        raise ValueError(
            f"use_imu_rates=True but the following IMU columns are absent: {missing}."
        )
    return (
        np.asarray(df[COL_ROLL_RATE].to_numpy(), dtype=np.float64),
        np.asarray(df[COL_PITCH_RATE].to_numpy(), dtype=np.float64),
        np.asarray(df[COL_YAW_RATE].to_numpy(), dtype=np.float64),
    )


def build_feature_matrix(df: pl.DataFrame, config: PipelineConfig) -> pl.DataFrame:
    """Build the Tolles-Lawson A-matrix from magnetometer data.

    Parameters
    ----------
    df:
        Input DataFrame containing required magnetometer columns.
    config:
        Pipeline configuration controlling which term set to include.

    Returns
    -------
    pl.DataFrame
        Feature matrix with 3, 9, 18, or 21 columns depending on ``config.model_terms``.

    Raises
    ------
    TypeError
        If ``df`` is not a ``polars.DataFrame``.
    ValueError
        If DataFrame validation fails (missing columns, wrong dtypes, nulls,
        non-monotonic time), or if ``config.model_terms`` is ``"c"`` or ``"d"``
        and ``df`` has fewer than 2 rows, or if any ``B_total`` value is
        non-positive, or if ``config.use_imu_rates`` is ``True`` and any of
        the three IMU columns are absent.
    """
    validate_dataframe(df)

    if config.model_terms in ("c", "d") and len(df) < 2:
        raise ValueError(
            "At least 2 rows are required to compute eddy current terms"
            " (model_terms='c' or 'd')."
        )

    cos_x, cos_y, cos_z = _direction_cosines(df)

    data: dict[str, npt.NDArray[np.float64]] = {}

    # Permanent terms (always included)
    data[COL_COS_X] = cos_x
    data[COL_COS_Y] = cos_y
    data[COL_COS_Z] = cos_z

    if config.model_terms in ("b", "c", "d"):
        # Induced terms (6)
        data[COL_COS_X2] = cos_x * cos_x
        data[COL_COS_XY] = cos_x * cos_y
        data[COL_COS_XZ] = cos_x * cos_z
        data[COL_COS_Y2] = cos_y * cos_y
        data[COL_COS_YZ] = cos_y * cos_z
        data[COL_COS_Z2] = cos_z * cos_z

    if config.model_terms in ("c", "d"):
        if config.use_imu_rates:
            dcos_x, dcos_y, dcos_z = _imu_rates(df)
        else:
            time = np.asarray(df[COL_TIME].to_numpy(), dtype=np.float64)
            dcos_x, dcos_y, dcos_z = _cosine_derivatives(
                cos_x, cos_y, cos_z, time, causal=config.use_cv
            )

        # Eddy current terms (9)
        data[COL_COS_X_DCOS_X] = cos_x * dcos_x
        data[COL_COS_X_DCOS_Y] = cos_x * dcos_y
        data[COL_COS_X_DCOS_Z] = cos_x * dcos_z
        data[COL_COS_Y_DCOS_X] = cos_y * dcos_x
        data[COL_COS_Y_DCOS_Y] = cos_y * dcos_y
        data[COL_COS_Y_DCOS_Z] = cos_y * dcos_z
        data[COL_COS_Z_DCOS_X] = cos_z * dcos_x
        data[COL_COS_Z_DCOS_Y] = cos_z * dcos_y
        data[COL_COS_Z_DCOS_Z] = cos_z * dcos_z

        if config.model_terms == "d":
            # Rate derivative terms (3)
            data[COL_DCOS_X] = dcos_x
            data[COL_DCOS_Y] = dcos_y
            data[COL_DCOS_Z] = dcos_z

    return pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in data.items()})
