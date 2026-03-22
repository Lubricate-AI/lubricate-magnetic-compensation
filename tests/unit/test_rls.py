"""Unit tests for lmc.rls — Recursive Least-Squares online updating."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lmc.calibration import CalibrationResult, calibrate
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
from lmc.features import build_feature_matrix
from lmc.rls import (
    RLSState,
    initialize_rls,
    rls_to_calibration_result,
    update_rls,
    update_rls_batch,
)
from lmc.segmentation import Segment


def _make_result(n_terms: int) -> CalibrationResult:
    """Minimal CalibrationResult with known coefficients."""
    return CalibrationResult(
        coefficients=np.arange(n_terms, dtype=np.float64),
        residuals=np.zeros(10, dtype=np.float64),
        condition_number=10.0,
        singular_values=np.ones(n_terms, dtype=np.float64),
        n_terms=n_terms,
    )


def test_initialize_rls_coefficient_shape() -> None:
    result = _make_result(9)
    state = initialize_rls(result)
    assert state.coefficients.shape == (9,)


def test_initialize_rls_covariance_shape() -> None:
    result = _make_result(9)
    state = initialize_rls(result)
    assert state.covariance.shape == (9, 9)


def test_initialize_rls_covariance_is_identity_scaled() -> None:
    result = _make_result(9)
    state = initialize_rls(result, initial_covariance_scale=100.0)
    expected = 100.0 * np.eye(9)
    np.testing.assert_array_equal(state.covariance, expected)


def test_initialize_rls_coefficients_match_result() -> None:
    result = _make_result(9)
    state = initialize_rls(result)
    np.testing.assert_array_equal(state.coefficients, result.coefficients)


def test_initialize_rls_n_terms() -> None:
    result = _make_result(3)
    state = initialize_rls(result)
    assert state.n_terms == 3


def test_initialize_rls_n_samples_zero() -> None:
    result = _make_result(9)
    state = initialize_rls(result)
    assert state.n_samples == 0


def test_initialize_rls_default_forgetting_factor_is_one() -> None:
    result = _make_result(9)
    state = initialize_rls(result)
    assert state.forgetting_factor == 1.0


def test_initialize_rls_custom_forgetting_factor() -> None:
    result = _make_result(9)
    state = initialize_rls(result, forgetting_factor=0.95)
    assert state.forgetting_factor == 0.95


def test_initialize_rls_rejects_zero_forgetting_factor() -> None:
    result = _make_result(9)
    with pytest.raises(ValueError, match="forgetting_factor"):
        initialize_rls(result, forgetting_factor=0.0)


def test_initialize_rls_rejects_forgetting_factor_above_one() -> None:
    result = _make_result(9)
    with pytest.raises(ValueError, match="forgetting_factor"):
        initialize_rls(result, forgetting_factor=1.01)


def test_initialize_rls_rejects_nonpositive_covariance_scale() -> None:
    result = _make_result(9)
    with pytest.raises(ValueError, match="initial_covariance_scale"):
        initialize_rls(result, initial_covariance_scale=0.0)


def _make_state(n_terms: int = 3, forgetting_factor: float = 1.0) -> RLSState:
    """Zero-coefficient state with identity covariance."""
    return RLSState(
        coefficients=np.zeros(n_terms, dtype=np.float64),
        covariance=np.eye(n_terms, dtype=np.float64),
        forgetting_factor=forgetting_factor,
        n_samples=0,
        n_terms=n_terms,
    )


def test_update_rls_returns_new_state() -> None:
    state = _make_state(3)
    a = np.array([1.0, 0.0, 0.0])
    new_state = update_rls(state, a, y=2.0)
    assert new_state is not state


def test_update_rls_does_not_mutate_original() -> None:
    state = _make_state(3)
    orig_coeffs = state.coefficients.copy()
    orig_cov = state.covariance.copy()
    update_rls(state, np.array([1.0, 0.0, 0.0]), y=2.0)
    np.testing.assert_array_equal(state.coefficients, orig_coeffs)
    np.testing.assert_array_equal(state.covariance, orig_cov)


def test_update_rls_increments_n_samples() -> None:
    state = _make_state(3)
    new_state = update_rls(state, np.array([1.0, 0.0, 0.0]), y=2.0)
    assert new_state.n_samples == 1
    new_state2 = update_rls(new_state, np.array([0.0, 1.0, 0.0]), y=1.0)
    assert new_state2.n_samples == 2


def test_update_rls_preserves_forgetting_factor() -> None:
    state = _make_state(3, forgetting_factor=0.95)
    new_state = update_rls(state, np.array([1.0, 0.0, 0.0]), y=2.0)
    assert new_state.forgetting_factor == 0.95


def test_update_rls_covariance_shrinks() -> None:
    """Processing a sample must reduce overall uncertainty (trace of P)."""
    state = _make_state(3)
    new_state = update_rls(state, np.array([1.0, 0.0, 0.0]), y=0.5)
    assert np.trace(new_state.covariance) < np.trace(state.covariance)


def test_update_rls_covariance_is_symmetric() -> None:
    state = _make_state(3)
    new_state = update_rls(state, np.array([0.5, 0.3, 0.8]), y=1.0)
    np.testing.assert_allclose(new_state.covariance, new_state.covariance.T, atol=1e-12)


def test_update_rls_single_term_exact() -> None:
    """Scalar case: one feature, one observation — verify by hand.

    θ=0, P=[[1]], a=[1], y=3, λ=1
    e = 3 - 0 = 3
    k = 1 / (1 + 1) = 0.5
    θ′ = 0 + 0.5 * 3 = 1.5
    P′ = (1 - 0.5 * 1) / 1 = 0.5
    """
    state = RLSState(
        coefficients=np.array([0.0]),
        covariance=np.array([[1.0]]),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=1,
    )
    new_state = update_rls(state, np.array([1.0]), y=3.0)
    np.testing.assert_allclose(new_state.coefficients, [1.5], atol=1e-12)
    np.testing.assert_allclose(new_state.covariance, [[0.5]], atol=1e-12)


def test_update_rls_forgetting_factor_less_than_one_inflates_covariance() -> None:
    """λ < 1 inflates P per update step; uncertainty grows faster."""
    state_no_forget = _make_state(3, forgetting_factor=1.0)
    state_forget = _make_state(3, forgetting_factor=0.9)
    a = np.array([0.1, 0.2, 0.3])
    y = 1.0
    ns_no = update_rls(state_no_forget, a, y)
    ns_with_forget = update_rls(state_forget, a, y)
    # With forgetting, P is divided by λ < 1 → larger P
    assert np.trace(ns_with_forget.covariance) > np.trace(ns_no.covariance)


# ---------------------------------------------------------------------------
# update_rls_batch tests
# ---------------------------------------------------------------------------

_CONFIG_A = PipelineConfig(model_terms="a")


def _make_rls_synthetic_df(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """Synthetic magnetometer DataFrame with COL_DELTA_B pre-populated.

    Uses ground-truth coefficients [1.0, -2.0, 0.5] for model_terms='a'.
    """
    rng = np.random.default_rng(seed)
    raw = rng.standard_normal((n_rows, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms
    b_total = 50_000.0
    bx = cosines[:, 0] * b_total
    by = cosines[:, 1] * b_total
    bz = cosines[:, 2] * b_total
    base_df = pl.DataFrame(
        {
            COL_TIME: np.arange(n_rows, dtype=np.float64),
            COL_LAT: np.full(n_rows, 45.0),
            COL_LON: np.full(n_rows, -75.0),
            COL_ALT: np.full(n_rows, 300.0),
            COL_BTOTAL: np.full(n_rows, b_total),
            COL_BX: bx,
            COL_BY: by,
            COL_BZ: bz,
        }
    )
    c_true = np.array([1.0, -2.0, 0.5])
    A = build_feature_matrix(base_df, _CONFIG_A).to_numpy()
    delta_b = A @ c_true
    return base_df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))


def test_update_rls_batch_increments_n_samples() -> None:
    df = _make_rls_synthetic_df(50)
    state = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=1e4 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=3,
    )
    new_state = update_rls_batch(state, df, _CONFIG_A)
    assert new_state.n_samples == 50


def test_update_rls_batch_matches_sequential() -> None:
    """update_rls_batch must produce identical result to sequential update_rls."""
    df = _make_rls_synthetic_df(20)
    A = build_feature_matrix(df, _CONFIG_A).to_numpy()
    dB = df[COL_DELTA_B].to_numpy()

    init_state = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=1e4 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=3,
    )

    # Sequential
    state_seq = init_state
    for i in range(len(df)):
        state_seq = update_rls(state_seq, A[i], dB[i])

    # Batch
    state_batch = update_rls_batch(init_state, df, _CONFIG_A)

    np.testing.assert_allclose(
        state_batch.coefficients, state_seq.coefficients, atol=1e-12
    )
    np.testing.assert_allclose(state_batch.covariance, state_seq.covariance, atol=1e-12)


def test_update_rls_batch_raises_if_no_delta_b() -> None:
    df = _make_rls_synthetic_df(10).drop(COL_DELTA_B)
    state = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=3,
    )
    with pytest.raises(ValueError, match=COL_DELTA_B):
        update_rls_batch(state, df, _CONFIG_A)


# ---------------------------------------------------------------------------
# OLS equivalence test
# ---------------------------------------------------------------------------


def test_rls_converges_to_ols_on_static_data() -> None:
    """RLS (λ=1) must match batch OLS after processing all training samples.

    This is the primary correctness guarantee for the RLS implementation.
    With no forgetting (λ=1) and sufficient samples, the RLS estimate
    is mathematically equivalent to the batch least-squares solution.
    """
    df = _make_rls_synthetic_df(n_rows=200, seed=99)
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=200)]

    # Batch OLS reference
    ols_result = calibrate(df, segments, _CONFIG_A)

    # RLS cold start: zero coefficients, very large initial uncertainty
    init_state = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=1e6 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=3,
    )
    rls_state = update_rls_batch(init_state, df, _CONFIG_A)

    # After processing all data, RLS should match OLS to high precision
    np.testing.assert_allclose(
        rls_state.coefficients, ols_result.coefficients, atol=1e-6
    )


# ---------------------------------------------------------------------------
# Forgetting factor adaptation test
# ---------------------------------------------------------------------------


def test_rls_forgetting_factor_adapts_to_coefficient_drift() -> None:
    """λ < 1 adapts to new coefficients faster than λ = 1.

    Two data segments with different ground-truth coefficients. After
    training on segment 1 and then updating on segment 2, the model
    with forgetting (λ=0.95) should produce lower coefficient error
    on segment 2 than the model without forgetting (λ=1.0).
    """
    n_seg = 150

    # Segment 1: ground-truth coefficients [1.0, -2.0, 0.5]
    df1 = _make_rls_synthetic_df(n_rows=n_seg, seed=10)

    # Segment 2: shifted ground-truth coefficients [3.0, 1.0, -1.5]
    c2 = np.array([3.0, 1.0, -1.5])
    rng = np.random.default_rng(20)
    raw = rng.standard_normal((n_seg, 3))
    cosines = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    b_total = 50_000.0
    base_df2 = pl.DataFrame(
        {
            COL_TIME: np.arange(n_seg, dtype=np.float64),
            COL_LAT: np.full(n_seg, 45.0),
            COL_LON: np.full(n_seg, -75.0),
            COL_ALT: np.full(n_seg, 300.0),
            COL_BTOTAL: np.full(n_seg, b_total),
            COL_BX: cosines[:, 0] * b_total,
            COL_BY: cosines[:, 1] * b_total,
            COL_BZ: cosines[:, 2] * b_total,
        }
    )
    A2 = build_feature_matrix(base_df2, _CONFIG_A).to_numpy()
    df2 = base_df2.with_columns(pl.Series(COL_DELTA_B, A2 @ c2, dtype=pl.Float64))

    init_no_forget = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=1e6 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=3,
    )
    init_forget = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=1e6 * np.eye(3, dtype=np.float64),
        forgetting_factor=0.95,
        n_samples=0,
        n_terms=3,
    )

    # Both train on segment 1
    state_no_forget = update_rls_batch(init_no_forget, df1, _CONFIG_A)
    state_forget = update_rls_batch(init_forget, df1, _CONFIG_A)

    # Both update on segment 2
    state_no_forget = update_rls_batch(state_no_forget, df2, _CONFIG_A)
    state_forget = update_rls_batch(state_forget, df2, _CONFIG_A)

    # Model with forgetting should be closer to c2
    error_no_forget = np.linalg.norm(state_no_forget.coefficients - c2)
    error_forget = np.linalg.norm(state_forget.coefficients - c2)
    assert error_forget < error_no_forget


# ---------------------------------------------------------------------------
# rls_to_calibration_result tests
# ---------------------------------------------------------------------------


def _make_converged_state() -> RLSState:
    """State with known coefficients [1.0, -2.0, 0.5] after processing 30 samples."""
    df = _make_rls_synthetic_df(30)
    init = RLSState(
        coefficients=np.zeros(3, dtype=np.float64),
        covariance=1e6 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=0,
        n_terms=3,
    )
    return update_rls_batch(init, df, _CONFIG_A)


def test_rls_to_calibration_result_returns_calibration_result() -> None:
    df = _make_rls_synthetic_df(30)
    state = _make_converged_state()
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert isinstance(result, CalibrationResult)


def test_rls_to_calibration_result_preserves_coefficients() -> None:
    df = _make_rls_synthetic_df(30)
    state = _make_converged_state()
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    np.testing.assert_array_equal(result.coefficients, state.coefficients)


def test_rls_to_calibration_result_n_terms_matches() -> None:
    df = _make_rls_synthetic_df(30)
    state = _make_converged_state()
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.n_terms == 3


def test_rls_to_calibration_result_residuals_shape() -> None:
    df = _make_rls_synthetic_df(30)
    state = _make_converged_state()
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.residuals.shape == (30,)


def test_rls_to_calibration_result_condition_number_is_positive() -> None:
    df = _make_rls_synthetic_df(30)
    state = _make_converged_state()
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.condition_number > 0.0


def test_rls_to_calibration_result_singular_values_descending() -> None:
    df = _make_rls_synthetic_df(30)
    state = _make_converged_state()
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.singular_values.shape == (3,)
    assert np.all(np.diff(result.singular_values) <= 0.0)  # descending


# ---------------------------------------------------------------------------
# Public API export test
# ---------------------------------------------------------------------------


def test_rls_symbols_exported_from_package() -> None:
    """All public RLS symbols must be importable from the top-level lmc package."""
    import lmc

    assert hasattr(lmc, "RLSState")
    assert hasattr(lmc, "initialize_rls")
    assert hasattr(lmc, "update_rls")
    assert hasattr(lmc, "update_rls_batch")
    assert hasattr(lmc, "rls_to_calibration_result")


def test_rlsstate_is_frozen() -> None:
    """RLSState must be immutable — field reassignment must raise an error."""
    state = _make_state(3)
    with pytest.raises(AttributeError):
        state.n_samples = 99  # type: ignore[misc]
