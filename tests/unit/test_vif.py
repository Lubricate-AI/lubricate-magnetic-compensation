"""Unit tests for lmc.vif."""

from __future__ import annotations

import numpy as np
import pytest

from lmc.vif import compute_vif


def test_identity_matrix_gives_vif_one() -> None:
    """Orthogonal columns have no multicollinearity — VIF == 1."""
    A = np.eye(4, dtype=np.float64)
    vif = compute_vif(A)
    assert vif.shape == (4,)
    np.testing.assert_allclose(vif, 1.0, atol=1e-10)


def test_independent_random_columns_have_low_vif() -> None:
    """Uncorrelated random columns should yield VIF close to 1."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((200, 4))
    vif = compute_vif(A)
    assert vif.shape == (4,)
    assert np.all(vif >= 1.0), "VIF must be >= 1 by definition"
    assert np.all(vif < 5.0), "Uncorrelated columns should have low VIF"


def test_perfectly_correlated_column_gives_inf_vif() -> None:
    """Duplicate column is perfectly predicted → R²=1 → VIF=inf."""
    rng = np.random.default_rng(1)
    base = rng.standard_normal(100)
    A = np.column_stack([base, base, rng.standard_normal(100)])
    vif = compute_vif(A)
    assert np.isinf(vif[0]) or np.isinf(vif[1]), (
        "Duplicate column should produce inf VIF"
    )


def test_vif_returns_float64_array() -> None:
    A = np.eye(3, dtype=np.float32)
    vif = compute_vif(A.astype(np.float64))
    assert vif.dtype == np.float64


def test_vif_raises_on_single_column() -> None:
    """Cannot compute R² when there are no other columns to regress on."""
    A = np.ones((10, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="at least 2 columns"):
        compute_vif(A)


def test_constant_column_gives_inf_vif() -> None:
    """A constant column has ss_tot == 0 — VIF should be inf."""
    rng = np.random.default_rng(2)
    # First column is constant, second and third are random.
    A = np.column_stack(
        [
            np.ones(50),
            rng.standard_normal(50),
            rng.standard_normal(50),
        ]
    )
    vif = compute_vif(A)
    assert np.isinf(vif[0]), "Constant column should have VIF == inf"


def test_vif_raises_on_1d_input() -> None:
    """A 1-D array should raise ValueError, not an opaque IndexError."""
    A = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2-D array"):
        compute_vif(A)
