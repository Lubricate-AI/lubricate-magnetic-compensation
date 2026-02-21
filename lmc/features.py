"""Tolles-Lawson A-matrix (design matrix) construction from magnetometer data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.columns import (
    COL_BTOTAL,
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
    COL_TIME,
)
from lmc.config import PipelineConfig


def _direction_cosines(
    df: pl.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute direction cosines from fluxgate magnetometer readings.

    Returns (cos_x, cos_y, cos_z) = (B_x/|B|, B_y/|B|, B_z/|B|).
    """
    b_total = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64)
    b_x = np.asarray(df[COL_BX].to_numpy(), dtype=np.float64)
    b_y = np.asarray(df[COL_BY].to_numpy(), dtype=np.float64)
    b_z = np.asarray(df[COL_BZ].to_numpy(), dtype=np.float64)

    if np.any(b_total <= 0.0):
        raise ValueError("B_total must be strictly positive for all rows.")

    cos_x = b_x / b_total
    cos_y = b_y / b_total
    cos_z = b_z / b_total

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
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute time derivatives of direction cosines using np.gradient.

    Uses explicit time coordinates to handle non-uniform sampling.
    Returns (dcos_x/dt, dcos_y/dt, dcos_z/dt).
    """
    dcos_x = np.asarray(np.gradient(cos_x, time), dtype=np.float64)
    dcos_y = np.asarray(np.gradient(cos_y, time), dtype=np.float64)
    dcos_z = np.asarray(np.gradient(cos_z, time), dtype=np.float64)
    return dcos_x, dcos_y, dcos_z


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
        Feature matrix with 3, 9, or 18 columns depending on ``config.model_terms``.

    Raises
    ------
    ValueError
        If ``config.model_terms == "c"`` and ``df`` has fewer than 2 rows,
        or if any ``B_total`` value is non-positive.
    """
    if config.model_terms == "c" and len(df) < 2:
        raise ValueError(
            "At least 2 rows are required to compute eddy current terms"
            " (model_terms='c')."
        )

    cos_x, cos_y, cos_z = _direction_cosines(df)

    data: dict[str, npt.NDArray[np.float64]] = {}

    # Permanent terms (always included)
    data[COL_COS_X] = cos_x
    data[COL_COS_Y] = cos_y
    data[COL_COS_Z] = cos_z

    if config.model_terms in ("b", "c"):
        # Induced terms (6)
        data[COL_COS_X2] = cos_x * cos_x
        data[COL_COS_XY] = cos_x * cos_y
        data[COL_COS_XZ] = cos_x * cos_z
        data[COL_COS_Y2] = cos_y * cos_y
        data[COL_COS_YZ] = cos_y * cos_z
        data[COL_COS_Z2] = cos_z * cos_z

    if config.model_terms == "c":
        time = np.asarray(df[COL_TIME].to_numpy(), dtype=np.float64)
        dcos_x, dcos_y, dcos_z = _cosine_derivatives(cos_x, cos_y, cos_z, time)

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

    return pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float64) for k, v in data.items()})
