"""Unit tests for lmc.earth_field."""

from __future__ import annotations

import datetime
import math

import numpy as np
import polars as pl
import ppigrf
import pytest

from lmc.columns import (
    COL_ALT,
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_DELTA_B,
    COL_LAT,
    COL_LON,
    COL_TIME,
)
from lmc.config import PipelineConfig
from lmc.earth_field import compute_interference

# ---------------------------------------------------------------------------
# Reference location: Ottawa, Canada
# ---------------------------------------------------------------------------

_LAT = 45.4
_LON = -75.7
_ALT_KM = 0.07
_IGRF_DATE = datetime.date(2024, 1, 1)
_UNIX_T0 = 1_704_067_200.0  # 2024-01-01 00:00:00 UTC — used only for COL_TIME

_CONFIG_IGRF = PipelineConfig(earth_field_method="igrf", igrf_date=_IGRF_DATE)
_CONFIG_STEADY = PipelineConfig(earth_field_method="steady_mean")


def _make_df(n_rows: int = 5, b_total: float = 50_000.0) -> pl.DataFrame:
    """Return a minimal valid DataFrame with constant B_total."""
    return pl.DataFrame(
        {
            COL_TIME: [_UNIX_T0 + float(i) for i in range(n_rows)],
            COL_LAT: [_LAT] * n_rows,
            COL_LON: [_LON] * n_rows,
            COL_ALT: [_ALT_KM] * n_rows,
            COL_BTOTAL: [b_total] * n_rows,
            COL_BX: [100.0] * n_rows,
            COL_BY: [200.0] * n_rows,
            COL_BZ: [300.0] * n_rows,
        }
    )


def _igrf_total(lat: float, lon: float, alt_km: float, date: datetime.date) -> float:
    """Return the IGRF total field (nT) at a single point."""
    dt = datetime.datetime(date.year, date.month, date.day)
    Be, Bn, Bu = ppigrf.igrf(lon, lat, alt_km, dt)  # type: ignore[no-untyped-call]
    return float(np.sqrt(Be[0] ** 2 + Bn[0] ** 2 + Bu[0] ** 2))


# ---------------------------------------------------------------------------
# IGRF path — sanity checks
# ---------------------------------------------------------------------------


def test_igrf_total_field_in_physical_range() -> None:
    """IGRF total field at reference location should be within expected range."""
    b_igrf = _igrf_total(_LAT, _LON, _ALT_KM, _IGRF_DATE)
    # Earth's surface field ranges roughly 25 000 – 65 000 nT globally.
    assert 25_000.0 < b_igrf < 65_000.0


def test_compute_interference_igrf_near_zero() -> None:
    """δB ≈ 0 when B_total is set equal to the IGRF baseline."""
    b_igrf = _igrf_total(_LAT, _LON, _ALT_KM, _IGRF_DATE)
    df = _make_df(b_total=b_igrf)
    result = compute_interference(df, _CONFIG_IGRF)
    assert all(math.isclose(v, 0.0, abs_tol=1e-6) for v in result.to_list())


def test_compute_interference_igrf_nonzero_interference() -> None:
    """δB equals the known offset when B_total differs from IGRF by a constant."""
    b_igrf = _igrf_total(_LAT, _LON, _ALT_KM, _IGRF_DATE)
    offset = 500.0
    df = _make_df(b_total=b_igrf + offset)
    result = compute_interference(df, _CONFIG_IGRF)
    assert all(math.isclose(v, offset, rel_tol=1e-9) for v in result.to_list())


# ---------------------------------------------------------------------------
# Steady-mean fallback path
# ---------------------------------------------------------------------------


def test_compute_interference_steady_mean_no_mask() -> None:
    """Without a mask all rows contribute; δB = B_total − mean(B_total)."""
    b_values = [50_000.0, 51_000.0, 52_000.0, 53_000.0, 54_000.0]
    mean_b = sum(b_values) / len(b_values)
    df = pl.DataFrame(
        {
            COL_TIME: [_UNIX_T0 + float(i) for i in range(5)],
            COL_LAT: [_LAT] * 5,
            COL_LON: [_LON] * 5,
            COL_ALT: [_ALT_KM] * 5,
            COL_BTOTAL: b_values,
            COL_BX: [100.0] * 5,
            COL_BY: [200.0] * 5,
            COL_BZ: [300.0] * 5,
        }
    )
    result = compute_interference(df, _CONFIG_STEADY)
    expected = [b - mean_b for b in b_values]
    assert all(
        math.isclose(v, e, abs_tol=1e-9)
        for v, e in zip(result.to_list(), expected, strict=True)
    )


def test_compute_interference_steady_mean_with_mask() -> None:
    """With a mask only selected rows define the baseline constant."""
    b_values = [50_000.0, 51_000.0, 52_000.0, 53_000.0, 54_000.0]
    steady_mask = pl.Series([True, True, False, False, False])
    steady_mean = (50_000.0 + 51_000.0) / 2.0  # 50 500.0
    df = pl.DataFrame(
        {
            COL_TIME: [_UNIX_T0 + float(i) for i in range(5)],
            COL_LAT: [_LAT] * 5,
            COL_LON: [_LON] * 5,
            COL_ALT: [_ALT_KM] * 5,
            COL_BTOTAL: b_values,
            COL_BX: [100.0] * 5,
            COL_BY: [200.0] * 5,
            COL_BZ: [300.0] * 5,
        }
    )
    result = compute_interference(df, _CONFIG_STEADY, steady_mask=steady_mask)
    expected = [b - steady_mean for b in b_values]
    assert all(
        math.isclose(v, e, abs_tol=1e-9)
        for v, e in zip(result.to_list(), expected, strict=True)
    )


def test_compute_interference_steady_mean_constant_series() -> None:
    """The baseline itself is constant across all rows."""
    df = _make_df(b_total=50_000.0)
    result = compute_interference(df, _CONFIG_STEADY)
    # B_total is constant so δB = B_total − mean(B_total) = 0 everywhere.
    assert all(math.isclose(v, 0.0, abs_tol=1e-9) for v in result.to_list())


def test_steady_mean_empty_mask_raises() -> None:
    """An all-False mask selects no rows and should raise ValueError."""
    df = _make_df()
    empty_mask = pl.Series([False] * df.height)
    with pytest.raises(ValueError, match="no rows"):
        compute_interference(df, _CONFIG_STEADY, steady_mask=empty_mask)


# ---------------------------------------------------------------------------
# Output dtype and name
# ---------------------------------------------------------------------------


def test_result_dtype_is_float64() -> None:
    result = compute_interference(_make_df(), _CONFIG_IGRF)
    assert result.dtype == pl.Float64


def test_result_name_is_delta_b() -> None:
    result = compute_interference(_make_df(), _CONFIG_IGRF)
    assert result.name == COL_DELTA_B
