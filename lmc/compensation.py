"""Apply Tolles-Lawson compensation to survey magnetometer data."""

from __future__ import annotations

import numpy as np
import polars as pl

from lmc.calibration import CalibrationResult
from lmc.columns import COL_BTOTAL, COL_TMI_COMPENSATED
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix


def compensate(
    df: pl.DataFrame,
    result: CalibrationResult,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Subtract modelled aircraft interference from survey TMI.

    Parameters
    ----------
    df:
        Survey DataFrame containing all required magnetometer columns.
    result:
        Calibration result from a prior call to ``calibrate()``.
    config:
        Pipeline configuration controlling which term set to use when
        building the feature matrix.  Must match the configuration used
        during calibration (same ``model_terms``).

    Returns
    -------
    pl.DataFrame
        Input DataFrame with one additional column ``COL_TMI_COMPENSATED``
        containing ``B_total - A @ coefficients``.

    Raises
    ------
    ValueError
        If the number of columns in the feature matrix does not match
        ``result.n_terms``, which indicates a mismatch between the
        calibration and compensation configurations.
    """
    feature_matrix = build_feature_matrix(df, config)
    A = feature_matrix.to_numpy()

    if A.shape[1] != result.n_terms:
        raise ValueError(
            f"Feature matrix has {A.shape[1]} columns but CalibrationResult "
            f"has {result.n_terms} terms. Ensure the same model_terms are "
            "used for both calibration and compensation."
        )

    interference = A @ result.coefficients
    tmi_comp = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64) - interference

    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
