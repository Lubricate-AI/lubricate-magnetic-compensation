"""Unit tests for lmc.rls — Recursive Least-Squares online updating."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lmc.calibration import CalibrationResult
from lmc.rls import RLSState, initialize_rls


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
