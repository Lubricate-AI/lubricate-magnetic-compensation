"""Unit tests for lmc.validation."""

import polars as pl
import pytest

from lmc.columns import (
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_TIME,
    REQUIRED_COLUMNS,
)
from lmc.validation import validate_dataframe


def _make_valid_df(n_rows: int = 5) -> pl.DataFrame:
    """Return a minimal valid DataFrame with ``n_rows`` rows."""
    return pl.DataFrame(
        {
            COL_TIME: [float(i) for i in range(n_rows)],
            "lat": [45.0 + 0.001 * i for i in range(n_rows)],
            "lon": [-75.0 + 0.001 * i for i in range(n_rows)],
            COL_BTOTAL: [50_000.0 + float(i) for i in range(n_rows)],
            COL_BX: [100.0 + float(i) for i in range(n_rows)],
            COL_BY: [200.0 + float(i) for i in range(n_rows)],
            COL_BZ: [300.0 + float(i) for i in range(n_rows)],
        }
    )


def test_validate_dataframe_returns_dataframe_unchanged() -> None:
    df = _make_valid_df()
    result = validate_dataframe(df)
    assert result is df


def test_validate_dataframe_accepts_extra_columns() -> None:
    df = _make_valid_df().with_columns(pl.lit(1.0).alias("extra_col"))
    result = validate_dataframe(df)
    assert "extra_col" in result.columns


def test_validate_dataframe_raises_type_error_for_non_dataframe() -> None:
    with pytest.raises(TypeError, match="polars.DataFrame"):
        validate_dataframe({"col": [1, 2, 3]})  # type: ignore[arg-type]


def test_validate_dataframe_raises_on_empty_dataframe() -> None:
    df = pl.DataFrame(schema=dict.fromkeys(REQUIRED_COLUMNS, pl.Float64))
    with pytest.raises(ValueError, match="must not be empty"):
        validate_dataframe(df)


def test_validate_dataframe_raises_on_missing_single_column() -> None:
    df = _make_valid_df().drop(COL_BTOTAL)
    with pytest.raises(ValueError, match=COL_BTOTAL):
        validate_dataframe(df)


def test_validate_dataframe_raises_on_multiple_missing_columns() -> None:
    df = _make_valid_df().drop([COL_BX, COL_BY, COL_BZ])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df)


def test_validate_dataframe_raises_on_integer_column() -> None:
    df = _make_valid_df().with_columns(pl.col(COL_BTOTAL).cast(pl.Int64))
    with pytest.raises(ValueError, match=COL_BTOTAL):
        validate_dataframe(df)


def test_validate_dataframe_raises_on_float32_column() -> None:
    df = _make_valid_df().with_columns(pl.col(COL_BX).cast(pl.Float32))
    with pytest.raises(ValueError, match=COL_BX):
        validate_dataframe(df)


def test_validate_dataframe_raises_on_null_in_required_column() -> None:
    df = _make_valid_df()
    nulled = df.with_columns(
        pl.Series(
            COL_BTOTAL,
            [50000.0, None, 50002.0, 50003.0, 50004.0],
            dtype=pl.Float64,
        )
    )
    with pytest.raises(ValueError, match="null values"):
        validate_dataframe(nulled)


def test_validate_dataframe_raises_on_non_monotonic_time() -> None:
    df = _make_valid_df()
    broken = df.with_columns(
        pl.Series(COL_TIME, [0.0, 2.0, 1.0, 3.0, 4.0], dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_dataframe(broken)


def test_validate_dataframe_raises_on_duplicate_timestamps() -> None:
    df = _make_valid_df()
    broken = df.with_columns(
        pl.Series(COL_TIME, [0.0, 1.0, 1.0, 2.0, 3.0], dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_dataframe(broken)
