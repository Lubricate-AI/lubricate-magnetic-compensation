"""Apply Tolles-Lawson compensation to survey magnetometer data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from lmc.calibration import CalibrationResult
from lmc.columns import COL_BTOTAL, COL_HEADING, COL_TMI_COMPENSATED
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix

if TYPE_CHECKING:
    from lmc.heading_calibration import HeadingCalibrationResult


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


def compensate_heading_specific(
    df: pl.DataFrame,
    result: HeadingCalibrationResult,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Subtract heading-specific modelled interference from survey TMI.

    For each row of ``df``, selects the coefficients from the nearest
    calibrated heading bin and subtracts the modelled interference from
    ``COL_BTOTAL``.

    Parameters
    ----------
    df:
        Survey DataFrame.  Must contain ``COL_HEADING`` and all columns
        required by ``build_feature_matrix``.
    result:
        Result from ``calibrate_per_heading()``.
    config:
        Pipeline configuration.  Must use the same ``model_terms`` as
        calibration.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with one additional column ``COL_TMI_COMPENSATED``.

    Raises
    ------
    ValueError
        If ``COL_HEADING`` is absent from ``df``, or if the feature matrix
        column count does not match the ``n_terms`` of the calibrated heading
        models in ``result``.
    """
    from lmc.segmentation import HeadingType, assign_heading_bin, resolve_bin_centres

    if COL_HEADING not in df.columns:
        raise ValueError(
            f"Column '{COL_HEADING}' is required for heading-specific compensation "
            f"but was not found. Available columns: {df.columns}"
        )

    feature_matrix = build_feature_matrix(df, config)
    A = feature_matrix.to_numpy()

    n_terms = next(iter(result.per_heading.values())).n_terms
    if A.shape[1] != n_terms:
        raise ValueError(
            f"Feature matrix has {A.shape[1]} columns but HeadingCalibrationResult "
            f"n_terms={n_terms}. Ensure the same model_terms are used for both "
            "calibration and compensation."
        )

    headings = np.asarray(df[COL_HEADING].to_numpy(), dtype=np.float64)
    all_centres = resolve_bin_centres(config, headings)
    # Restrict to calibrated headings so every row routes to an available model.
    centres: dict[HeadingType, float] = {
        k: v for k, v in all_centres.items() if k in result.per_heading
    }

    # Route each row to the nearest calibrated heading.  Tolerance 180° ensures
    # every row is always assigned (nearest-neighbour, no gaps).
    interference = np.zeros(len(df), dtype=np.float64)
    for h_label, cal in result.per_heading.items():
        mask = np.array(
            [assign_heading_bin(h, centres, 180.0) == h_label for h in headings],
            dtype=bool,
        )
        if mask.any():
            interference[mask] = A[mask] @ cal.coefficients

    tmi_comp = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64) - interference

    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
