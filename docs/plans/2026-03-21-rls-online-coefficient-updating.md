# RLS Online Coefficient Updating Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Recursive Least-Squares (RLS) for online/incremental Tolles-Lawson coefficient updating, enabling real-time adaptation to sensor drift and changing aircraft configuration.

**Architecture:** A new `lmc/rls.py` module owns `RLSState` (the stateful object) plus four public functions: `initialize_rls()`, `update_rls()`, `update_rls_batch()`, and `rls_to_calibration_result()`. The existing `calibrate()`, `build_feature_matrix()`, and `compensate()` functions are unchanged — RLS slots in as an optional path that produces a `CalibrationResult`-compatible output.

**Tech Stack:** Python 3.12+, NumPy (matrix ops, Kalman gain), Polars (DataFrame I/O), pytest (TDD). No new dependencies required.

---

## Background: Existing Pipeline + RLS Math

Before writing any code, read these files:

| File | Purpose |
|------|---------|
| `lmc/columns.py` | Column-name constants — need `COL_DELTA_B` |
| `lmc/config.py` | `PipelineConfig` — pass to `build_feature_matrix` |
| `lmc/calibration.py` | `CalibrationResult` frozen dataclass + `calibrate()` |
| `lmc/features.py` | `build_feature_matrix(df, config)` → polars DataFrame A-matrix |
| `lmc/segmentation.py` | `Segment` dataclass |
| `lmc/__init__.py` | Public API — add exports here last |
| `tests/unit/test_calibration.py` | `_make_synthetic_df` and `_make_synthetic_data` helpers — copy this pattern |

**RLS Algorithm (forgetting factor λ ∈ (0, 1]):**

Given state at time t−1: coefficients θ (p×1), covariance P (p×p), forgetting factor λ.
Given new sample: feature vector **a** (p×1), observation y (scalar).

```
e  = y - aᵀ θ                            # innovation (prediction error)
k  = P a / (λ + aᵀ P a)                  # Kalman gain (p×1)
θ′ = θ + k e                             # coefficient update
P′ = (P − k aᵀ P) / λ                   # covariance update
P′ = (P′ + P′ᵀ) / 2                      # symmetrize to prevent numerical drift
```

Key properties:
- **λ = 1.0**: Standard RLS — all data weighted equally; converges to OLS on static data
- **λ < 1.0**: Exponential forgetting — recent data weighted more; enables tracking non-stationary coefficients
- **O(p²) per sample** — no matrix inverse needed
- **Covariance P tracks uncertainty** — large diagonal = high uncertainty; shrinks as more data arrives

**Initialization:** Set `θ` from batch `CalibrationResult.coefficients`. Set `P = scale × I` where `scale` controls how much the RLS is willing to update away from the batch estimate. Large `scale` (e.g., 1e3) means high initial uncertainty → coefficients update aggressively from the first samples.

---

## Task 1: RLSState Dataclass + initialize_rls()

**Files:**
- Create: `lmc/rls.py`
- Create: `tests/unit/test_rls.py`

### Step 1: Write failing tests

Create `tests/unit/test_rls.py`:

```python
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
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_rls.py -v
```

Expected: `ModuleNotFoundError: No module named 'lmc.rls'`

### Step 3: Implement RLSState and initialize_rls

Create `lmc/rls.py`:

```python
"""Recursive Least-Squares (RLS) for online Tolles-Lawson coefficient updating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lmc.calibration import CalibrationResult
from lmc.columns import COL_DELTA_B
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix

import polars as pl


@dataclass
class RLSState:
    """Mutable state for Recursive Least-Squares online coefficient updating.

    Attributes
    ----------
    coefficients:
        Current coefficient estimate, shape ``(n_terms,)``.
    covariance:
        Current covariance matrix, shape ``(n_terms, n_terms)``.
        Diagonal entries approximate per-coefficient variance.
        Use ``np.diag(state.covariance)`` to get per-coefficient std devs.
    forgetting_factor:
        Exponential forgetting rate λ ∈ (0, 1]. λ=1 weights all history equally.
        λ<1 down-weights old samples; try λ=0.95–0.99 for slowly drifting systems.
    n_samples:
        Total number of individual samples processed since initialization.
    n_terms:
        Number of model coefficients (3, 9, 18, or 21 depending on model_terms).
    """

    coefficients: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    forgetting_factor: float
    n_samples: int
    n_terms: int


def initialize_rls(
    result: CalibrationResult,
    forgetting_factor: float = 1.0,
    *,
    initial_covariance_scale: float = 1.0,
) -> RLSState:
    """Create an RLSState from a batch CalibrationResult.

    Parameters
    ----------
    result:
        Batch calibration result to initialize from.  The ``coefficients``
        become the initial RLS estimate.
    forgetting_factor:
        Exponential forgetting rate λ ∈ (0, 1].  Defaults to 1.0 (no forgetting).
    initial_covariance_scale:
        Diagonal scale for the initial covariance P = scale × I.  Larger values
        allow the RLS to update aggressively from the first samples; smaller
        values anchor coefficients closer to the batch estimate.

    Returns
    -------
    RLSState
        Initialized state ready for incremental updates.

    Raises
    ------
    ValueError
        If ``forgetting_factor`` is not in (0, 1] or ``initial_covariance_scale``
        is not strictly positive.
    """
    if not (0.0 < forgetting_factor <= 1.0):
        raise ValueError(
            f"forgetting_factor must be in (0, 1]; got {forgetting_factor}."
        )
    if initial_covariance_scale <= 0.0:
        raise ValueError(
            f"initial_covariance_scale must be strictly positive; "
            f"got {initial_covariance_scale}."
        )

    n = result.n_terms
    return RLSState(
        coefficients=result.coefficients.copy(),
        covariance=initial_covariance_scale * np.eye(n, dtype=np.float64),
        forgetting_factor=forgetting_factor,
        n_samples=0,
        n_terms=n,
    )
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_rls.py -v -k "initialize"
```

Expected: All `initialize` tests PASS.

### Step 5: Commit

```bash
git add lmc/rls.py tests/unit/test_rls.py
git commit -m "feat: add RLSState dataclass and initialize_rls"
```

---

## Task 2: update_rls() Single-Sample Kalman Update

**Files:**
- Modify: `lmc/rls.py`
- Modify: `tests/unit/test_rls.py`

### Step 1: Write failing tests

Append to `tests/unit/test_rls.py`:

```python
from lmc.rls import update_rls


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
    """Scalar case: one feature, one observation — verify by hand."""
    # θ=0, P=[[1]], a=[1], y=3, λ=1
    # e = 3 - 0 = 3
    # k = 1 / (1 + 1) = 0.5
    # θ′ = 0 + 0.5 * 3 = 1.5
    # P′ = (1 - 0.5 * 1) / 1 = 0.5
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
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_rls.py -v -k "update_rls"
```

Expected: `ImportError` or `AttributeError` on `update_rls`.

### Step 3: Implement update_rls

Add to `lmc/rls.py` (after `initialize_rls`):

```python
def update_rls(
    state: RLSState,
    a: npt.NDArray[np.float64],
    y: float,
) -> RLSState:
    """Apply one RLS update step using the Kalman gain formulation.

    Parameters
    ----------
    state:
        Current RLS state.  Not mutated — a new state is returned.
    a:
        Feature vector for this sample, shape ``(n_terms,)``.
    y:
        Observed delta_B value for this sample (scalar).

    Returns
    -------
    RLSState
        Updated state with new coefficients and covariance.
    """
    lam = state.forgetting_factor
    theta = state.coefficients
    P = state.covariance

    # Innovation
    e = float(y) - float(a @ theta)

    # Kalman gain: k = P a / (λ + aᵀ P a)
    Pa = P @ a  # (p,)
    gain_denom = lam + float(a @ Pa)
    k = Pa / gain_denom  # (p,)

    # Coefficient update
    new_theta = theta + k * e

    # Covariance update: P′ = (P − k aᵀ P) / λ
    new_P = (P - np.outer(k, a @ P)) / lam
    # Symmetrize to prevent numerical drift
    new_P = (new_P + new_P.T) / 2.0

    return RLSState(
        coefficients=new_theta.astype(np.float64),
        covariance=new_P.astype(np.float64),
        forgetting_factor=lam,
        n_samples=state.n_samples + 1,
        n_terms=state.n_terms,
    )
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_rls.py -v -k "update_rls"
```

Expected: All `update_rls` tests PASS.

### Step 5: Commit

```bash
git add lmc/rls.py tests/unit/test_rls.py
git commit -m "feat: add update_rls single-sample Kalman gain update"
```

---

## Task 3: update_rls_batch() DataFrame-Based Bulk Update

**Files:**
- Modify: `lmc/rls.py`
- Modify: `tests/unit/test_rls.py`

### Step 1: Write failing tests

Append to `tests/unit/test_rls.py`:

```python
from lmc.rls import update_rls_batch
from lmc.calibration import calibrate
from lmc.columns import (
    COL_ALT, COL_BTOTAL, COL_BX, COL_BY, COL_BZ,
    COL_DELTA_B, COL_LAT, COL_LON, COL_TIME,
)
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import Segment


_CONFIG_A = PipelineConfig(model_terms="a")


def _make_rls_synthetic_df(n_rows: int, seed: int = 42) -> pl.DataFrame:
    """Synthetic magnetometer DataFrame with COL_DELTA_B pre-populated."""
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
    # Generate delta_B from known coefficients [1.0, -2.0, 0.5]
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
    np.testing.assert_allclose(
        state_batch.covariance, state_seq.covariance, atol=1e-12
    )


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
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_rls.py -v -k "batch"
```

Expected: `ImportError` on `update_rls_batch`.

### Step 3: Implement update_rls_batch

Add to `lmc/rls.py`:

```python
def update_rls_batch(
    state: RLSState,
    df: pl.DataFrame,
    config: PipelineConfig,
) -> RLSState:
    """Apply RLS updates for every row in a DataFrame segment.

    Builds the feature matrix from ``df`` via ``build_feature_matrix``, then
    iterates row-by-row calling :func:`update_rls`.  Equivalent to calling
    ``update_rls`` in a loop but more convenient for segment-based workflows.

    Parameters
    ----------
    state:
        Current RLS state.  Not mutated — a new state is returned.
    df:
        DataFrame containing all required magnetometer columns plus
        ``COL_DELTA_B``.
    config:
        Pipeline configuration used to build the feature matrix.

    Returns
    -------
    RLSState
        Updated state after processing all rows in ``df``.

    Raises
    ------
    ValueError
        If ``COL_DELTA_B`` is absent from ``df``.
    """
    if COL_DELTA_B not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_B}' is required for RLS updates but was not "
            f"found in the DataFrame. Available columns: {df.columns}"
        )

    A: npt.NDArray[np.float64] = build_feature_matrix(df, config).to_numpy()
    dB: npt.NDArray[np.float64] = df[COL_DELTA_B].to_numpy().astype(np.float64)

    current = state
    for i in range(A.shape[0]):
        current = update_rls(current, A[i], dB[i])
    return current
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_rls.py -v -k "batch"
```

Expected: All `batch` tests PASS.

### Step 5: Commit

```bash
git add lmc/rls.py tests/unit/test_rls.py
git commit -m "feat: add update_rls_batch for DataFrame-based bulk updates"
```

---

## Task 4: OLS Equivalence Test

**Files:**
- Modify: `tests/unit/test_rls.py`

This task adds the critical acceptance criterion: RLS with λ=1 must converge to the batch OLS solution when processing all training data.

### Step 1: Write the failing test

Append to `tests/unit/test_rls.py`:

```python
def test_rls_converges_to_ols_on_static_data() -> None:
    """RLS (λ=1) must match batch OLS after processing all training samples.

    This is the primary correctness guarantee for the RLS implementation.
    With no forgetting (λ=1) and sufficient samples, the RLS estimate
    is mathematically equivalent to the batch least-squares solution.
    """
    # Ground-truth coefficients for A-model (3 terms)
    c_true = np.array([1.0, -2.0, 0.5])
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
```

### Step 2: Run the test to verify it fails

```bash
uv run pytest tests/unit/test_rls.py::test_rls_converges_to_ols_on_static_data -v
```

Expected: FAIL — RLS hasn't been implemented yet (or coefficients don't match).

### Step 3: Run the test to verify it now passes

```bash
uv run pytest tests/unit/test_rls.py::test_rls_converges_to_ols_on_static_data -v
```

Expected: PASS (no code changes needed — the existing implementation should produce this result).

**If it fails:** Check the initial covariance scale. With `1e6 * I`, the first sample has very high influence. Increase to `1e8 * I` and/or increase `n_rows` to 500+.

### Step 4: Commit

```bash
git add tests/unit/test_rls.py
git commit -m "test: verify RLS convergence to OLS on static data"
```

---

## Task 5: Forgetting Factor Adaptation Test

**Files:**
- Modify: `tests/unit/test_rls.py`

### Step 1: Write the failing test

Append to `tests/unit/test_rls.py`:

```python
def test_rls_forgetting_factor_adapts_to_coefficient_drift() -> None:
    """λ < 1 adapts to new coefficients faster than λ = 1.

    Two data segments with different ground-truth coefficients. After
    training on segment 1 and then updating on segment 2, the model
    with forgetting (λ=0.95) should produce lower residuals on segment 2
    than the model without forgetting (λ=1.0).
    """
    n_seg = 150

    # Segment 1: coefficients [1.0, -2.0, 0.5]
    c1 = np.array([1.0, -2.0, 0.5])
    df1 = _make_rls_synthetic_df(n_rows=n_seg, seed=10)

    # Segment 2: shifted coefficients [3.0, 1.0, -1.5]
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
    df2 = base_df2.with_columns(
        pl.Series(COL_DELTA_B, A2 @ c2, dtype=pl.Float64)
    )

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

    # Residuals on segment 2 ground truth
    residuals_no_forget = np.abs(state_no_forget.coefficients - c2)
    residuals_forget = np.abs(state_forget.coefficients - c2)

    # Model with forgetting should be closer to c2
    assert np.linalg.norm(residuals_forget) < np.linalg.norm(residuals_no_forget)
```

### Step 2: Run test to verify it fails

```bash
uv run pytest tests/unit/test_rls.py::test_rls_forgetting_factor_adapts_to_coefficient_drift -v
```

Expected: FAIL — forgetting factor not yet implemented correctly, or test fails to import.

### Step 3: Run test to verify it passes

```bash
uv run pytest tests/unit/test_rls.py::test_rls_forgetting_factor_adapts_to_coefficient_drift -v
```

Expected: PASS (the existing `update_rls` implementation handles λ in the covariance update).

### Step 4: Commit

```bash
git add tests/unit/test_rls.py
git commit -m "test: verify forgetting factor enables adaptation to coefficient drift"
```

---

## Task 6: rls_to_calibration_result() for compensate() Compatibility

**Files:**
- Modify: `lmc/rls.py`
- Modify: `tests/unit/test_rls.py`

This enables the common workflow: call `rls_to_calibration_result(state, df, config)` then pass the result to `compensate()`.

### Step 1: Write failing tests

Append to `tests/unit/test_rls.py`:

```python
from lmc.rls import rls_to_calibration_result
from lmc.calibration import CalibrationResult


def test_rls_to_calibration_result_returns_calibration_result() -> None:
    df = _make_rls_synthetic_df(30)
    state = RLSState(
        coefficients=np.array([1.0, -2.0, 0.5]),
        covariance=0.01 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=30,
        n_terms=3,
    )
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert isinstance(result, CalibrationResult)


def test_rls_to_calibration_result_preserves_coefficients() -> None:
    df = _make_rls_synthetic_df(30)
    c = np.array([1.0, -2.0, 0.5])
    state = RLSState(
        coefficients=c.copy(),
        covariance=0.01 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=30,
        n_terms=3,
    )
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    np.testing.assert_array_equal(result.coefficients, c)


def test_rls_to_calibration_result_n_terms_matches() -> None:
    df = _make_rls_synthetic_df(30)
    state = RLSState(
        coefficients=np.array([1.0, -2.0, 0.5]),
        covariance=0.01 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=30,
        n_terms=3,
    )
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.n_terms == 3


def test_rls_to_calibration_result_residuals_shape() -> None:
    df = _make_rls_synthetic_df(30)
    state = RLSState(
        coefficients=np.array([1.0, -2.0, 0.5]),
        covariance=0.01 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=30,
        n_terms=3,
    )
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.residuals.shape == (30,)


def test_rls_to_calibration_result_condition_number_is_positive() -> None:
    df = _make_rls_synthetic_df(30)
    state = RLSState(
        coefficients=np.array([1.0, -2.0, 0.5]),
        covariance=0.01 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=30,
        n_terms=3,
    )
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.condition_number > 0.0


def test_rls_to_calibration_result_singular_values_descending() -> None:
    df = _make_rls_synthetic_df(30)
    state = RLSState(
        coefficients=np.array([1.0, -2.0, 0.5]),
        covariance=0.01 * np.eye(3, dtype=np.float64),
        forgetting_factor=1.0,
        n_samples=30,
        n_terms=3,
    )
    result = rls_to_calibration_result(state, df, _CONFIG_A)
    assert result.singular_values.shape == (3,)
    assert np.all(np.diff(result.singular_values) <= 0.0)  # descending
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_rls.py -v -k "calibration_result"
```

Expected: `ImportError` on `rls_to_calibration_result`.

### Step 3: Implement rls_to_calibration_result

Add to `lmc/rls.py`:

```python
def rls_to_calibration_result(
    state: RLSState,
    df: pl.DataFrame,
    config: PipelineConfig,
) -> CalibrationResult:
    """Convert an RLSState to a CalibrationResult for use with compensate().

    Computes residuals from ``df`` using the current RLS coefficients and
    derives diagnostic statistics (condition number, singular values) from
    the inverse of the covariance matrix P ≈ (AᵀA)⁻¹.

    Parameters
    ----------
    state:
        Current RLS state.
    df:
        DataFrame to compute residuals against.  Must contain
        ``COL_DELTA_B``.
    config:
        Pipeline configuration used to build the feature matrix.

    Returns
    -------
    CalibrationResult
        Drop-in replacement usable with :func:`lmc.compensate`.
    """
    A: npt.NDArray[np.float64] = build_feature_matrix(df, config).to_numpy()
    dB: npt.NDArray[np.float64] = df[COL_DELTA_B].to_numpy().astype(np.float64)
    residuals = np.asarray(A @ state.coefficients - dB, dtype=np.float64)

    # P ≈ (AᵀA)⁻¹, so eigenvalues of P⁻¹ ≈ singular_values(A)²
    # Singular values of A ≈ 1 / sqrt(eigenvalues(P))
    eigenvalues = np.linalg.eigvalsh(state.covariance)
    # Clamp to avoid sqrt of tiny negatives from numerical noise
    eigenvalues = np.maximum(eigenvalues, 0.0)
    approx_singular_values = np.sort(1.0 / np.sqrt(eigenvalues + 1e-30))[::-1]
    approx_singular_values = np.asarray(approx_singular_values, dtype=np.float64)

    condition_number = float(
        approx_singular_values[0] / (approx_singular_values[-1] + 1e-30)
    )

    return CalibrationResult(
        coefficients=state.coefficients.copy(),
        residuals=residuals,
        condition_number=condition_number,
        singular_values=approx_singular_values,
        n_terms=state.n_terms,
    )
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_rls.py -v -k "calibration_result"
```

Expected: All `calibration_result` tests PASS.

### Step 5: Commit

```bash
git add lmc/rls.py tests/unit/test_rls.py
git commit -m "feat: add rls_to_calibration_result for compensate() compatibility"
```

---

## Task 7: Export Public API

**Files:**
- Modify: `lmc/__init__.py`

### Step 1: Write a failing import test

Append to `tests/unit/test_rls.py`:

```python
def test_rls_symbols_exported_from_package() -> None:
    """All public RLS symbols must be importable from the top-level lmc package."""
    import lmc
    assert hasattr(lmc, "RLSState")
    assert hasattr(lmc, "initialize_rls")
    assert hasattr(lmc, "update_rls")
    assert hasattr(lmc, "update_rls_batch")
    assert hasattr(lmc, "rls_to_calibration_result")
```

### Step 2: Run test to verify it fails

```bash
uv run pytest tests/unit/test_rls.py::test_rls_symbols_exported_from_package -v
```

Expected: FAIL — `AttributeError: module 'lmc' has no attribute 'RLSState'`

### Step 3: Add exports to `lmc/__init__.py`

Add the import at the top of the existing imports block in `lmc/__init__.py`:

```python
from lmc.rls import (
    RLSState,
    initialize_rls,
    update_rls,
    update_rls_batch,
    rls_to_calibration_result,
)
```

And add to `__all__`:

```python
    "RLSState",
    "initialize_rls",
    "update_rls",
    "update_rls_batch",
    "rls_to_calibration_result",
```

### Step 4: Run test to verify it passes

```bash
uv run pytest tests/unit/test_rls.py::test_rls_symbols_exported_from_package -v
```

Expected: PASS.

### Step 5: Commit

```bash
git add lmc/__init__.py tests/unit/test_rls.py
git commit -m "feat: export RLS public API from lmc package"
```

---

## Task 8: Full Lint + Test Pass

**Files:** None (verification only)

### Step 1: Run the full test suite

```bash
uv run pytest tests/ -v
```

Expected: All tests PASS (no regressions in existing calibration, adaptive, compensation, etc.).

### Step 2: Run lint

```bash
make lint
```

Expected: No ruff, typos, or pyright errors.

**Common pyright issues to fix:**
- `npt.NDArray[np.float64]` — ensure numpy.typing is imported
- `float(a @ Pa)` — explicit cast needed for pyright's strict mode
- Return type annotations on all public functions

### Step 3: Fix any lint issues

Apply `make format` first:

```bash
make format
```

Then address remaining pyright/ruff errors manually.

### Step 4: Commit any lint fixes

```bash
git add lmc/rls.py
git commit -m "chore: fix lint issues in rls module"
```

### Step 5: Final check

```bash
make lint && make test
```

Expected: Both pass with zero errors.

---

## Summary

| Task | New File | Key Function | Test File |
|------|----------|-------------|-----------|
| 1 | `lmc/rls.py` | `RLSState`, `initialize_rls` | `tests/unit/test_rls.py` |
| 2 | — | `update_rls` | same |
| 3 | — | `update_rls_batch` | same |
| 4 | — | (equivalence test) | same |
| 5 | — | (forgetting test) | same |
| 6 | — | `rls_to_calibration_result` | same |
| 7 | — | (exports) | `lmc/__init__.py` + same |
| 8 | — | (lint/test) | — |

**Acceptance Criteria Coverage:**
- ✅ RLS with Kalman gain approach → Tasks 1–3
- ✅ `update_rls` API (equivalent to `update_coefficients`) → Task 2
- ✅ Covariance matrix tracking → `RLSState.covariance` (Task 1)
- ✅ Forgetting factor lambda → `initialize_rls(forgetting_factor=...)` (Task 1, 5)
- ✅ Initialize from batch calibration → `initialize_rls(result)` (Task 1)
- ✅ Unit tests: equivalence with batch on static data → Task 4
- ✅ O(p²) memory (P matrix) and O(p²) time per update (matrix-vector multiply) → Tasks 2–3
- ✅ Convergence analysis → Task 5 (forgetting factor test shows adaptation)
