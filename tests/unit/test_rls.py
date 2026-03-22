"""Unit tests for lmc.rls — Recursive Least-Squares online updating."""

from __future__ import annotations

import numpy as np
import pytest

from lmc.calibration import CalibrationResult
from lmc.rls import RLSState, initialize_rls, update_rls


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
    np.testing.assert_allclose(
        new_state.covariance, new_state.covariance.T, atol=1e-12
    )


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
    ns_fo = update_rls(state_forget, a, y)
    # With forgetting, P is divided by λ < 1 → larger P
    assert np.trace(ns_fo.covariance) > np.trace(ns_no.covariance)
