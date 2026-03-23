"""Unit tests for lmc.features."""

from __future__ import annotations

import math

import polars as pl
import pytest

from lmc.columns import (
    COL_ALT,
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
    COL_DCOS_X,
    COL_DCOS_Y,
    COL_DCOS_Z,
    COL_LAT,
    COL_LON,
    COL_PITCH_RATE,
    COL_ROLL_RATE,
    COL_TIME,
    COL_YAW_RATE,
)
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix

_N_ROWS = 5
_BX = 3.0
_BY = 4.0
_BZ = 0.0
_COS_X = 0.6  # 3/5
_COS_Y = 0.8  # 4/5
_COS_Z = 0.0  # 0/5

_CONFIG_A = PipelineConfig(model_terms="a")
_CONFIG_B = PipelineConfig(model_terms="b")
_CONFIG_C = PipelineConfig(model_terms="c")
_CONFIG_D = PipelineConfig(model_terms="d")


def _make_df(
    bx: float = _BX,
    by: float = _BY,
    bz: float = _BZ,
    n: int = _N_ROWS,
) -> pl.DataFrame:
    """Return a valid DataFrame with constant B field across ``n`` rows."""
    btotal = 54000.0  # scalar magnetometer — intentionally != fluxgate magnitude
    return pl.DataFrame(
        {
            COL_TIME: [float(i) for i in range(n)],
            COL_LAT: [45.0] * n,
            COL_LON: [-75.0] * n,
            COL_ALT: [300.0] * n,
            COL_BTOTAL: [btotal] * n,
            COL_BX: [bx] * n,
            COL_BY: [by] * n,
            COL_BZ: [bz] * n,
        }
    )


def _make_linear_df() -> pl.DataFrame:
    """Return a DataFrame with linearly varying direction cosines.

    cos_x goes [0.6, 0.8, 1.0] across t = [0, 1, 2].
    cos_y goes [0.8, 0.6, 0.0].
    cos_z = 0.0 everywhere.
    Each row is a unit vector so normalisation is a no-op.
    B vectors are constructed so that fluxgate magnitude stays at 10.0 and
    direction cosines match the desired values at each row.
    """
    cos_x_vals = [0.6, 0.8, 1.0]
    cos_y_vals = [0.8, 0.6, 0.0]
    cos_z_vals = [0.0, 0.0, 0.0]
    fluxgate_mag = 10.0
    return pl.DataFrame(
        {
            COL_TIME: [0.0, 1.0, 2.0],
            COL_LAT: [45.0, 45.0, 45.0],
            COL_LON: [-75.0, -75.0, -75.0],
            COL_ALT: [300.0, 300.0, 300.0],
            COL_BTOTAL: [54000.0, 54000.0, 54000.0],
            COL_BX: [cx * fluxgate_mag for cx in cos_x_vals],
            COL_BY: [cy * fluxgate_mag for cy in cos_y_vals],
            COL_BZ: [cz * fluxgate_mag for cz in cos_z_vals],
        }
    )


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_terms_a_shape() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_A)
    assert result.shape == (_N_ROWS, 3)
    assert result.columns == [COL_COS_X, COL_COS_Y, COL_COS_Z]


def test_terms_b_shape() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_B)
    assert result.shape == (_N_ROWS, 9)
    assert result.columns == [
        COL_COS_X,
        COL_COS_Y,
        COL_COS_Z,
        COL_COS_X2,
        COL_COS_XY,
        COL_COS_XZ,
        COL_COS_Y2,
        COL_COS_YZ,
        COL_COS_Z2,
    ]


def test_terms_c_shape() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_C)
    assert result.shape == (_N_ROWS, 18)
    assert result.columns == [
        COL_COS_X,
        COL_COS_Y,
        COL_COS_Z,
        COL_COS_X2,
        COL_COS_XY,
        COL_COS_XZ,
        COL_COS_Y2,
        COL_COS_YZ,
        COL_COS_Z2,
        COL_COS_X_DCOS_X,
        COL_COS_X_DCOS_Y,
        COL_COS_X_DCOS_Z,
        COL_COS_Y_DCOS_X,
        COL_COS_Y_DCOS_Y,
        COL_COS_Y_DCOS_Z,
        COL_COS_Z_DCOS_X,
        COL_COS_Z_DCOS_Y,
        COL_COS_Z_DCOS_Z,
    ]


def test_terms_d_shape() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_D)
    assert result.shape == (_N_ROWS, 21)
    assert result.columns == [
        COL_COS_X,
        COL_COS_Y,
        COL_COS_Z,
        COL_COS_X2,
        COL_COS_XY,
        COL_COS_XZ,
        COL_COS_Y2,
        COL_COS_YZ,
        COL_COS_Z2,
        COL_COS_X_DCOS_X,
        COL_COS_X_DCOS_Y,
        COL_COS_X_DCOS_Z,
        COL_COS_Y_DCOS_X,
        COL_COS_Y_DCOS_Y,
        COL_COS_Y_DCOS_Z,
        COL_COS_Z_DCOS_X,
        COL_COS_Z_DCOS_Y,
        COL_COS_Z_DCOS_Z,
        COL_DCOS_X,
        COL_DCOS_Y,
        COL_DCOS_Z,
    ]


def test_row_count_matches_input() -> None:
    for n in (2, 5, 10):
        df = _make_df(n=n)
        result = build_feature_matrix(df, _CONFIG_C)
        assert result.height == n


# ---------------------------------------------------------------------------
# dtype test
# ---------------------------------------------------------------------------


def test_all_columns_float64() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_C)
    for col in result.columns:
        assert result[col].dtype == pl.Float64, f"Column {col!r} is not Float64"


# ---------------------------------------------------------------------------
# Value tests — permanent terms
# ---------------------------------------------------------------------------


def test_direction_cosines_values() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_A)
    assert all(math.isclose(v, _COS_X) for v in result[COL_COS_X].to_list())
    assert all(math.isclose(v, _COS_Y) for v in result[COL_COS_Y].to_list())
    assert all(
        math.isclose(v, _COS_Z, abs_tol=1e-12) for v in result[COL_COS_Z].to_list()
    )


def test_direction_cosines_unit_vector() -> None:
    """cos_x^2 + cos_y^2 + cos_z^2 must be approximately 1.0 for all rows."""
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_A)
    cos_x = result[COL_COS_X].to_numpy()
    cos_y = result[COL_COS_Y].to_numpy()
    cos_z = result[COL_COS_Z].to_numpy()
    norms = cos_x**2 + cos_y**2 + cos_z**2
    assert all(math.isclose(n, 1.0, rel_tol=1e-9) for n in norms)


# ---------------------------------------------------------------------------
# Value tests — induced terms
# ---------------------------------------------------------------------------


def test_induced_terms_values() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_B)

    expected = {
        COL_COS_X2: _COS_X * _COS_X,  # 0.36
        COL_COS_XY: _COS_X * _COS_Y,  # 0.48
        COL_COS_XZ: _COS_X * _COS_Z,  # 0.0
        COL_COS_Y2: _COS_Y * _COS_Y,  # 0.64
        COL_COS_YZ: _COS_Y * _COS_Z,  # 0.0
        COL_COS_Z2: _COS_Z * _COS_Z,  # 0.0
    }
    for col, expected_val in expected.items():
        assert all(
            math.isclose(v, expected_val, abs_tol=1e-12) for v in result[col].to_list()
        ), f"Mismatch in column {col!r}"


# ---------------------------------------------------------------------------
# Value tests — eddy terms
# ---------------------------------------------------------------------------


def test_eddy_terms_zero_constant_signal() -> None:
    """Constant B → all direction cosine derivatives = 0 → all eddy terms = 0."""
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_C)

    eddy_cols = [
        COL_COS_X_DCOS_X,
        COL_COS_X_DCOS_Y,
        COL_COS_X_DCOS_Z,
        COL_COS_Y_DCOS_X,
        COL_COS_Y_DCOS_Y,
        COL_COS_Y_DCOS_Z,
        COL_COS_Z_DCOS_X,
        COL_COS_Z_DCOS_Y,
        COL_COS_Z_DCOS_Z,
    ]
    for col in eddy_cols:
        assert all(
            math.isclose(v, 0.0, abs_tol=1e-12) for v in result[col].to_list()
        ), f"Expected zero for {col!r} with constant signal"


def test_eddy_terms_values_linear_signal() -> None:
    """Linearly varying cosines → correct derivative products at the middle row."""
    df = _make_linear_df()
    result = build_feature_matrix(df, _CONFIG_C)

    # cos_x = [0.6, 0.8, 1.0] over t=[0,1,2]
    # np.gradient central diff at index 1: (1.0 - 0.6) / (2*1) = 0.2
    # cos_y = [0.8, 0.6, 0.0] → dcos_y at index 1 = (0.0 - 0.8) / 2 = -0.4
    # cos_z = [0.0, 0.0, 0.0] → dcos_z = 0.0
    dcos_x_mid = 0.2
    dcos_y_mid = -0.4
    dcos_z_mid = 0.0
    cos_x_mid = 0.8
    cos_y_mid = 0.6

    mid = 1  # middle row index

    def _close(actual: float, expected: float) -> bool:
        return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)

    assert _close(float(result[COL_COS_X_DCOS_X][mid]), cos_x_mid * dcos_x_mid)
    assert _close(float(result[COL_COS_X_DCOS_Y][mid]), cos_x_mid * dcos_y_mid)
    assert _close(float(result[COL_COS_X_DCOS_Z][mid]), cos_x_mid * dcos_z_mid)
    assert _close(float(result[COL_COS_Y_DCOS_X][mid]), cos_y_mid * dcos_x_mid)
    assert _close(float(result[COL_COS_Y_DCOS_Y][mid]), cos_y_mid * dcos_y_mid)
    assert _close(float(result[COL_COS_Y_DCOS_Z][mid]), cos_y_mid * dcos_z_mid)
    assert _close(float(result[COL_COS_Z_DCOS_X][mid]), 0.0 * dcos_x_mid)
    assert _close(float(result[COL_COS_Z_DCOS_Y][mid]), 0.0 * dcos_y_mid)
    assert _close(float(result[COL_COS_Z_DCOS_Z][mid]), 0.0 * dcos_z_mid)


# ---------------------------------------------------------------------------
# Error condition tests
# ---------------------------------------------------------------------------


def test_raises_zero_fluxgate_magnitude() -> None:
    """All-zero fluxgate components -> zero magnitude -> ValueError."""
    df = pl.DataFrame(
        {
            COL_TIME: [0.0, 1.0],
            COL_LAT: [45.0, 45.0],
            COL_LON: [-75.0, -75.0],
            COL_ALT: [300.0, 300.0],
            COL_BTOTAL: [54000.0, 54000.0],
            COL_BX: [0.0, 0.0],
            COL_BY: [0.0, 0.0],
            COL_BZ: [0.0, 0.0],
        }
    )
    with pytest.raises(ValueError, match="strictly positive"):
        build_feature_matrix(df, _CONFIG_A)


def test_raises_type_error_for_non_dataframe() -> None:
    with pytest.raises(TypeError):
        build_feature_matrix({"not": "a dataframe"}, _CONFIG_A)  # type: ignore


def test_raises_on_missing_column() -> None:
    df = _make_df().drop(COL_BTOTAL)
    with pytest.raises(ValueError, match="Missing required columns"):
        build_feature_matrix(df, _CONFIG_A)


def test_raises_on_null_in_required_column() -> None:
    df = _make_df().with_columns(
        pl.Series(COL_BTOTAL, [None, 5.0, 5.0, 5.0, 5.0], dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="null values"):
        build_feature_matrix(df, _CONFIG_A)


def test_raises_on_non_monotonic_time() -> None:
    df = _make_df().with_columns(
        pl.Series(COL_TIME, [0.0, 2.0, 1.0, 3.0, 4.0], dtype=pl.Float64)
    )
    with pytest.raises(ValueError, match="monotonically increasing"):
        build_feature_matrix(df, _CONFIG_A)


def test_raises_single_row_terms_c() -> None:
    df = _make_df(n=1)
    with pytest.raises(ValueError, match="At least 2 rows"):
        build_feature_matrix(df, _CONFIG_C)


def test_no_raise_single_row_terms_a() -> None:
    df = _make_df(n=1)
    result = build_feature_matrix(df, _CONFIG_A)
    assert result.height == 1


def test_no_raise_single_row_terms_b() -> None:
    df = _make_df(n=1)
    result = build_feature_matrix(df, _CONFIG_B)
    assert result.height == 1


# ---------------------------------------------------------------------------
# Null check
# ---------------------------------------------------------------------------


def test_no_nulls_in_output() -> None:
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_C)
    null_counts = result.null_count()
    for col in result.columns:
        assert null_counts[col][0] == 0, f"Unexpected nulls in column {col!r}"


# ---------------------------------------------------------------------------
# IMU angular rate path tests
# ---------------------------------------------------------------------------

_ROLL_RATE = 0.1
_PITCH_RATE = 0.2
_YAW_RATE = 0.3

_CONFIG_C_IMU = PipelineConfig(model_terms="c", use_imu_rates=True)


def _make_imu_df(
    roll_rate: float = _ROLL_RATE,
    pitch_rate: float = _PITCH_RATE,
    yaw_rate: float = _YAW_RATE,
    n: int = _N_ROWS,
) -> pl.DataFrame:
    """Return a valid DataFrame with constant B field and constant IMU rate columns."""
    base = _make_df(n=n)
    return base.with_columns(
        pl.lit(roll_rate).cast(pl.Float64).alias(COL_ROLL_RATE),
        pl.lit(pitch_rate).cast(pl.Float64).alias(COL_PITCH_RATE),
        pl.lit(yaw_rate).cast(pl.Float64).alias(COL_YAW_RATE),
    )


def test_imu_path_shape() -> None:
    df = _make_imu_df()
    result = build_feature_matrix(df, _CONFIG_C_IMU)
    assert result.shape == (_N_ROWS, 18)
    assert result.columns == [
        COL_COS_X,
        COL_COS_Y,
        COL_COS_Z,
        COL_COS_X2,
        COL_COS_XY,
        COL_COS_XZ,
        COL_COS_Y2,
        COL_COS_YZ,
        COL_COS_Z2,
        COL_COS_X_DCOS_X,
        COL_COS_X_DCOS_Y,
        COL_COS_X_DCOS_Z,
        COL_COS_Y_DCOS_X,
        COL_COS_Y_DCOS_Y,
        COL_COS_Y_DCOS_Z,
        COL_COS_Z_DCOS_X,
        COL_COS_Z_DCOS_Y,
        COL_COS_Z_DCOS_Z,
    ]


def test_imu_path_eddy_values() -> None:
    """Constant B + constant IMU rates → eddy terms equal cos_i * rate_j exactly."""
    df = _make_imu_df()
    result = build_feature_matrix(df, _CONFIG_C_IMU)

    expected = {
        COL_COS_X_DCOS_X: _COS_X * _ROLL_RATE,
        COL_COS_X_DCOS_Y: _COS_X * _PITCH_RATE,
        COL_COS_X_DCOS_Z: _COS_X * _YAW_RATE,
        COL_COS_Y_DCOS_X: _COS_Y * _ROLL_RATE,
        COL_COS_Y_DCOS_Y: _COS_Y * _PITCH_RATE,
        COL_COS_Y_DCOS_Z: _COS_Y * _YAW_RATE,
        COL_COS_Z_DCOS_X: _COS_Z * _ROLL_RATE,
        COL_COS_Z_DCOS_Y: _COS_Z * _PITCH_RATE,
        COL_COS_Z_DCOS_Z: _COS_Z * _YAW_RATE,
    }
    for col, expected_val in expected.items():
        assert all(
            math.isclose(v, expected_val, abs_tol=1e-12) for v in result[col].to_list()
        ), f"Mismatch in column {col!r}"


# ---------------------------------------------------------------------------
# Causal (use_cv=True) derivative path
# ---------------------------------------------------------------------------

_CONFIG_C_CV = PipelineConfig(model_terms="c", use_cv=True, use_ridge=True)
_CONFIG_D_CV = PipelineConfig(model_terms="d", use_cv=True, use_ridge=True)


def test_causal_derivatives_no_leakage_at_boundary() -> None:
    """With use_cv=True, dcos at row i must not depend on row i+1.

    cos_x = [0.6, 0.6, 0.8, 0.9] (non-linear so central ≠ backward diffs).
    At row 2:
      backward diff = (0.8 - 0.6) / (2.0 - 1.0) = 0.2
      central  diff = (0.9 - 0.6) / (3.0 - 1.0) = 0.15
    The test asserts 0.2 — i.e. the causal (backward) result.
    """
    fluxgate_mag = 10.0
    cos_x_vals = [0.6, 0.6, 0.8, 0.9]
    cos_y_vals = [math.sqrt(1.0 - cx**2) for cx in cos_x_vals]
    df = pl.DataFrame(
        {
            COL_TIME: [0.0, 1.0, 2.0, 3.0],
            COL_LAT: [45.0] * 4,
            COL_LON: [-75.0] * 4,
            COL_ALT: [300.0] * 4,
            COL_BTOTAL: [54000.0] * 4,
            COL_BX: [cx * fluxgate_mag for cx in cos_x_vals],
            COL_BY: [cy * fluxgate_mag for cy in cos_y_vals],
            COL_BZ: [0.0] * 4,
        }
    )

    result = build_feature_matrix(df, _CONFIG_C_CV)

    # Extract dcos_x at row 2 by dividing the product col by cos_x
    actual_dcos_x_row2 = float(result[COL_COS_X_DCOS_X][2]) / cos_x_vals[2]
    # backward diff = 0.2; central diff would give 0.15
    expected_dcos_x_row2 = (cos_x_vals[2] - cos_x_vals[1]) / (2.0 - 1.0)
    assert math.isclose(actual_dcos_x_row2, expected_dcos_x_row2, rel_tol=1e-9), (
        f"Expected backward diff {expected_dcos_x_row2}, got {actual_dcos_x_row2}. "
        "Central diff would give 0.15 — use_cv=True is not using causal diffs."
    )


def test_causal_derivatives_first_row_replicated() -> None:
    """Row 0 gets the same derivative as row 1 (forward-padded).

    cos_x = [0.6, 0.6, 0.8, 0.9]:
      backward diff at row 1 = (0.6 - 0.6) / 1.0 = 0.0  → replicated to row 0
      np.gradient at row 0   = forward diff = 0.0
      np.gradient at row 1   = central diff = (0.8 - 0.6) / 2.0 = 0.1  ← differs!
    So dcos[0] == dcos[1] holds only for the causal path.
    """
    fluxgate_mag = 10.0
    cos_x_vals = [0.6, 0.6, 0.8, 0.9]
    cos_y_vals = [math.sqrt(1.0 - cx**2) for cx in cos_x_vals]
    df = pl.DataFrame(
        {
            COL_TIME: [0.0, 1.0, 2.0, 3.0],
            COL_LAT: [45.0] * 4,
            COL_LON: [-75.0] * 4,
            COL_ALT: [300.0] * 4,
            COL_BTOTAL: [54000.0] * 4,
            COL_BX: [cx * fluxgate_mag for cx in cos_x_vals],
            COL_BY: [cy * fluxgate_mag for cy in cos_y_vals],
            COL_BZ: [0.0] * 4,
        }
    )

    result = build_feature_matrix(df, _CONFIG_C_CV)

    dcos_x_row0 = float(result[COL_COS_X_DCOS_X][0]) / cos_x_vals[0]
    dcos_x_row1 = float(result[COL_COS_X_DCOS_X][1]) / cos_x_vals[1]
    assert math.isclose(dcos_x_row0, dcos_x_row1, abs_tol=1e-12), (
        f"Row 0 dcos_x ({dcos_x_row0}) should equal row 1 dcos_x ({dcos_x_row1}) "
        "when use_cv=True (row 0 padded from first backward diff)."
    )


def test_non_cv_path_unchanged() -> None:
    """use_cv=False still uses central differences (np.gradient) — no regression."""
    pass  # Covered by test_eddy_terms_values_linear_signal (central-diff path).


def test_imu_path_raises_when_columns_absent() -> None:
    df = _make_df()  # no IMU columns
    with pytest.raises(ValueError, match="IMU columns are absent"):
        build_feature_matrix(df, _CONFIG_C_IMU)


def test_imu_path_raises_partial_columns() -> None:
    """Only 1 of 3 IMU columns present → ValueError listing the missing ones."""
    df = _make_df().with_columns(
        pl.lit(_ROLL_RATE).cast(pl.Float64).alias(COL_ROLL_RATE)
    )
    with pytest.raises(ValueError, match="IMU columns are absent"):
        build_feature_matrix(df, _CONFIG_C_IMU)


def test_default_path_unchanged_with_imu_columns_present() -> None:
    """use_imu_rates=False (default) uses finite-difference path even
    when IMU columns exist."""
    df = _make_imu_df()
    result = build_feature_matrix(df, _CONFIG_C)  # default config, use_imu_rates=False

    # Constant B → all direction cosine derivatives = 0 → all eddy terms = 0
    eddy_cols = [
        COL_COS_X_DCOS_X,
        COL_COS_X_DCOS_Y,
        COL_COS_X_DCOS_Z,
        COL_COS_Y_DCOS_X,
        COL_COS_Y_DCOS_Y,
        COL_COS_Y_DCOS_Z,
        COL_COS_Z_DCOS_X,
        COL_COS_Z_DCOS_Y,
        COL_COS_Z_DCOS_Z,
    ]
    for col in eddy_cols:
        assert all(
            math.isclose(v, 0.0, abs_tol=1e-12) for v in result[col].to_list()
        ), f"Expected zero for {col!r} with constant signal and default (FD) path"
