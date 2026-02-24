"""Earth field baseline computation using IGRF or steady-segment mean fallback."""

from __future__ import annotations

import datetime

import numpy as np
import polars as pl
import ppigrf

from lmc.columns import COL_ALT, COL_BTOTAL, COL_DELTA_B, COL_LAT, COL_LON
from lmc.config import PipelineConfig


def _igrf_baseline(df: pl.DataFrame, date: datetime.date) -> pl.Series:
    """Evaluate IGRF total field at each sample location.

    Parameters
    ----------
    df:
        Validated input DataFrame containing lat, lon, and alt columns.
    date:
        Date for IGRF model evaluation.

    Returns
    -------
    pl.Series
        Float64 Series of IGRF total field values (nT), one per sample row.
    """
    lon = df[COL_LON].to_numpy()
    lat = df[COL_LAT].to_numpy()
    alt_km = df[COL_ALT].to_numpy()

    # ppigrf expects a naive datetime (no tzinfo).
    dt = datetime.datetime(date.year, date.month, date.day)

    # ppigrf returns shape (N_dates, N_points); [0] selects the single date.
    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt_km, dt)  # type: ignore[no-untyped-call]
    b_igrf: np.ndarray = np.sqrt(Be[0] ** 2 + Bn[0] ** 2 + Bu[0] ** 2)

    return pl.Series(values=b_igrf, dtype=pl.Float64)


def _steady_mean_baseline(
    df: pl.DataFrame,
    steady_mask: pl.Series | None,
) -> pl.Series:
    """Return a constant baseline equal to mean B_total over steady segments.

    Parameters
    ----------
    df:
        Validated input DataFrame.
    steady_mask:
        Boolean Series selecting steady-maneuver rows. All rows are used
        when ``None``.

    Returns
    -------
    pl.Series
        Float64 Series where every element equals the mean B_total of the
        selected rows.

    Raises
    ------
    ValueError
        If ``steady_mask`` length does not match ``df.height``.
    ValueError
        If ``steady_mask`` selects no rows.
    """
    if steady_mask is not None and len(steady_mask) != df.height:
        raise ValueError(
            "steady_mask must have the same number of rows as df; "
            f"got {len(steady_mask)} vs {df.height}."
        )
    if steady_mask is not None:
        b_values = df[COL_BTOTAL].filter(steady_mask)
    else:
        b_values = df[COL_BTOTAL]
    mean_val = b_values.mean()
    if mean_val is None:
        raise ValueError("steady_mask selects no rows; cannot compute mean baseline.")
    return pl.Series(values=[mean_val] * df.height, dtype=pl.Float64)


def compute_interference(
    df: pl.DataFrame,
    config: PipelineConfig,
    *,
    steady_mask: pl.Series | None = None,
) -> pl.Series:
    """Compute δB = B_total_measured − earth_field_baseline per sample.

    Parameters
    ----------
    df:
        Validated input DataFrame.
    config:
        Pipeline configuration controlling which baseline path to use.
        Must have ``igrf_date`` set when ``earth_field_method`` is ``"igrf"``.
    steady_mask:
        Boolean Series selecting steady-maneuver rows for the
        ``"steady_mean"`` fallback path. All rows are used when ``None``.
        Ignored when ``config.earth_field_method`` is ``"igrf"``.

    Returns
    -------
    pl.Series
        Float64 Series named ``delta_B`` containing the per-sample
        aircraft interference signal (nT).
    """
    if config.earth_field_method == "igrf":
        if config.igrf_date is None:
            raise ValueError(  # pragma: no cover — enforced by PipelineConfig validator
                "igrf_date is required when earth_field_method is 'igrf'."
            )
        baseline = _igrf_baseline(df, config.igrf_date)
    else:
        baseline = _steady_mean_baseline(df, steady_mask)

    return (df[COL_BTOTAL] - baseline).alias(COL_DELTA_B)
