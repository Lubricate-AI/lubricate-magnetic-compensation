# Normalize Alpha Convention Between Ridge and Sklearn Regularizers

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `lasso_alpha` and `elastic_net_alpha` follow the same unnormalized convention as `ridge_alpha`, so a given alpha value produces comparable regularization strength regardless of dataset size or which method is chosen.

**Architecture:** sklearn's `Lasso` and `ElasticNet` minimize `(1/(2*n)) * ||Aw-dB||² + alpha * ||w||`, so their alpha scales inversely with `n_samples`. Ridge uses the augmented-matrix approach (`||Aw-dB||² + alpha*||w||²`) which is sample-count-agnostic. The fix (Option A from Issue #66): multiply the user-facing alpha by `n_samples` before passing it to sklearn, and divide CV-returned alphas back by `n_samples`. No config changes needed—only `calibration.py` and field description updates.

**Tech Stack:** Python, numpy, sklearn (`Lasso`, `ElasticNet`, `LassoCV`, `ElasticNetCV`), polars, pydantic, pytest

---

### Task 1: Write failing tests for non-CV LASSO alpha scaling

**Files:**
- Modify: `tests/unit/test_calibration.py` (append new tests after `test_elastic_net_l1_ratio_1_behaves_like_lasso`)

**Context:** For model_terms="a", features are direction cosines in `[-1, 1]`. With c_true=[1.0, -2.0, 0.5], `||A[:,j]||²/n ≈ 1/3`. Current code passes `lasso_alpha=1e-3` directly to sklearn; the fix must pass `1e-3 * n_samples`. The recovery test with `lasso_alpha=1e-3, n=80` will break because `alpha_sk=0.08` introduces ~0.24 bias per coefficient (>> atol=0.1).

**Step 1: Write two failing tests**

Add to `tests/unit/test_calibration.py` after `test_elastic_net_l1_ratio_1_behaves_like_lasso`:

```python
# ---------------------------------------------------------------------------
# Alpha convention tests: unnormalized (ridge) convention
# ---------------------------------------------------------------------------


def test_lasso_uses_n_samples_scaled_alpha_internally() -> None:
    """Non-CV LASSO should pass lasso_alpha * n_samples to sklearn Lasso.

    sklearn's Lasso normalizes by n_samples internally, so to maintain
    the unnormalized (ridge) alpha convention, we scale up before passing.
    """
    c_true = np.array([1.0, -2.0, 0.5])
    n_rows = 80
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config, n_rows=n_rows)

    result = calibrate(df, segments, config)

    # Recompute manually with the expected scaled alpha
    A = build_feature_matrix(
        df.slice(0, n_rows), config
    ).to_numpy()
    dB = df[COL_DELTA_B].to_numpy().astype(np.float64)
    from sklearn.linear_model import Lasso as _Lasso

    expected = _Lasso(alpha=1e-3 * n_rows, fit_intercept=False, max_iter=10_000)
    expected.fit(A, dB)
    np.testing.assert_allclose(result.coefficients, expected.coef_, atol=1e-10)


def test_lasso_selected_alpha_is_user_convention_not_scaled() -> None:
    """CalibrationResult.selected_alpha should be the user-facing alpha, not scaled."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=2e-4)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    # selected_alpha must reflect what the user passed, not the sklearn-scaled value
    assert result.selected_alpha == pytest.approx(2e-4)  # pyright: ignore[reportUnknownMemberType]
```

**Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/unit/test_calibration.py::test_lasso_uses_n_samples_scaled_alpha_internally tests/unit/test_calibration.py::test_lasso_selected_alpha_is_user_convention_not_scaled -v
```

Expected: first test FAILS (coefficients don't match scaled-alpha result), second test PASSES (already stores config value).

---

### Task 2: Write failing test for CV LASSO alpha convention

**Files:**
- Modify: `tests/unit/test_calibration.py`

**Step 1: Write failing test**

Append after the tests added in Task 1:

```python
def test_lasso_cv_selected_alpha_in_unnormalized_convention() -> None:
    """CV LASSO selected_alpha must be in unnormalized (ridge) convention.

    LassoCV returns alpha in sklearn's convention (normalized by n_samples).
    We must divide by n_samples to bring it into the same convention as ridge.
    """
    df, segments = _make_multicollinear_df_for_cv()
    n_rows = len(df)
    config = PipelineConfig(model_terms="a", use_lasso=True, use_cv=True, cv_folds=5)
    result = calibrate(df, segments, config)

    # Reproduce what LassoCV returns in sklearn convention
    A = build_feature_matrix(df, config).to_numpy()
    dB = df[COL_DELTA_B].to_numpy().astype(np.float64)
    from sklearn.linear_model import LassoCV as _LassoCV
    from sklearn.model_selection import TimeSeriesSplit as _TSS

    cv = _TSS(n_splits=5)
    model_cv = _LassoCV(cv=cv, fit_intercept=False, max_iter=10_000)
    model_cv.fit(A, dB)

    # After fix: selected_alpha = model_cv.alpha_ / n_rows
    expected_user_alpha = float(model_cv.alpha_) / n_rows
    assert result.selected_alpha == pytest.approx(expected_user_alpha, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]
```

**Step 2: Run to confirm it fails**

```bash
uv run pytest tests/unit/test_calibration.py::test_lasso_cv_selected_alpha_in_unnormalized_convention -v
```

Expected: FAILS (current code stores `model_lcv.alpha_` directly without dividing by n_rows).

---

### Task 3: Fix LASSO alpha scaling in `calibrate()`

**Files:**
- Modify: `lmc/calibration.py:181-195`

**Step 1: Apply the fix**

Replace the non-CV LASSO block (currently lines 189-194):

```python
# BEFORE:
            model_l = Lasso(
                alpha=config.lasso_alpha, fit_intercept=False, max_iter=10_000
            )
            model_l.fit(A, dB)  # pyright: ignore[reportUnknownMemberType]
            coefficients = np.asarray(model_l.coef_, dtype=np.float64)
            selected_alpha = config.lasso_alpha
```

```python
# AFTER:
            # Alpha convention: user alpha follows the unnormalized ridge convention
            # (||Aw-dB||² + alpha*||w||). sklearn normalizes by n_samples, so we
            # multiply by n_samples to compensate. selected_alpha stores the
            # user-facing value unchanged.
            n_samples = A.shape[0]
            model_l = Lasso(
                alpha=config.lasso_alpha * n_samples,
                fit_intercept=False,
                max_iter=10_000,
            )
            model_l.fit(A, dB)  # pyright: ignore[reportUnknownMemberType]
            coefficients = np.asarray(model_l.coef_, dtype=np.float64)
            selected_alpha = config.lasso_alpha
```

Replace the CV LASSO selected_alpha line (currently line 187):

```python
# BEFORE:
            selected_alpha = float(model_lcv.alpha_)
```

```python
# AFTER:
            # Convert from sklearn's n_samples-normalized convention back to
            # the unnormalized user convention (same as ridge_alpha).
            selected_alpha = float(model_lcv.alpha_) / A.shape[0]
```

**Step 2: Run Task 1 + 2 tests to verify they pass**

```bash
uv run pytest tests/unit/test_calibration.py::test_lasso_uses_n_samples_scaled_alpha_internally tests/unit/test_calibration.py::test_lasso_selected_alpha_is_user_convention_not_scaled tests/unit/test_calibration.py::test_lasso_cv_selected_alpha_in_unnormalized_convention -v
```

Expected: all 3 PASS.

**Step 3: Run the full LASSO test group to find regressions**

```bash
uv run pytest tests/unit/test_calibration.py -k "lasso" -v
```

Expected: `test_lasso_recovers_reasonable` FAILS (alpha semantics changed; `alpha_sk=0.08` now, was `0.001` before—bias exceeds `atol=0.1`).

---

### Task 4: Update `test_lasso_recovers_reasonable` for new alpha convention

**Files:**
- Modify: `tests/unit/test_calibration.py:324-332`

**Context:** With the new convention, `lasso_alpha=1e-3` with n=80 passes `alpha_sk=0.08` to sklearn. For direction-cosine features (scale ~1/3), this introduces ~0.24 bias on a coefficient of 0.5—exceeding `atol=0.1`. The fix: use `lasso_alpha=1e-5` so `alpha_sk=8e-4`, which is weak enough for near-exact recovery.

**Step 1: Update the test**

```python
# BEFORE:
def test_lasso_recovers_reasonable() -> None:
    """LASSO introduces bias but should return plausible, finite coefficients."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-3)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)
```

```python
# AFTER:
def test_lasso_recovers_reasonable() -> None:
    """LASSO introduces bias but should return plausible, finite coefficients.

    Uses a small lasso_alpha (unnormalized convention) so the scaled sklearn
    alpha remains weak and near-exact recovery is expected.
    """
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(model_terms="a", use_lasso=True, lasso_alpha=1e-5)
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)
```

**Step 2: Run lasso tests**

```bash
uv run pytest tests/unit/test_calibration.py -k "lasso" -v
```

Expected: all PASS.

**Step 3: Commit LASSO changes**

```bash
git add lmc/calibration.py tests/unit/test_calibration.py
git commit -m "fix: normalize lasso alpha to unnormalized (ridge) convention

Multiply lasso_alpha by n_samples before passing to sklearn Lasso so the
user-facing alpha is sample-count-agnostic, matching ridge_alpha semantics.
For CV, divide the returned alpha_ back by n_samples."
```

---

### Task 5: Write failing tests for non-CV ElasticNet alpha scaling

**Files:**
- Modify: `tests/unit/test_calibration.py`

**Step 1: Write failing tests**

Append after the LASSO alpha convention tests (after Task 2's tests):

```python
def test_elastic_net_uses_n_samples_scaled_alpha_internally() -> None:
    """Non-CV ElasticNet should pass elastic_net_alpha * n_samples to sklearn.

    sklearn's ElasticNet normalizes by n_samples; scaling up matches ridge convention.
    """
    c_true = np.array([1.0, -2.0, 0.5])
    n_rows = 80
    config = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=1e-3,
        elastic_net_l1_ratio=0.5,
    )
    df, segments = _make_synthetic_data(c_true, config, n_rows=n_rows)

    result = calibrate(df, segments, config)

    A = build_feature_matrix(df.slice(0, n_rows), config).to_numpy()
    dB = df[COL_DELTA_B].to_numpy().astype(np.float64)
    from sklearn.linear_model import ElasticNet as _ElasticNet

    expected = _ElasticNet(
        alpha=1e-3 * n_rows,
        l1_ratio=0.5,
        fit_intercept=False,
        max_iter=10_000,
    )
    expected.fit(A, dB)
    np.testing.assert_allclose(result.coefficients, expected.coef_, atol=1e-10)


def test_elastic_net_cv_selected_alpha_in_unnormalized_convention() -> None:
    """CV ElasticNet selected_alpha must be in unnormalized (ridge) convention."""
    df, segments = _make_multicollinear_df_for_cv()
    n_rows = len(df)
    config = PipelineConfig(
        model_terms="a", use_elastic_net=True, use_cv=True, cv_folds=5
    )
    result = calibrate(df, segments, config)

    A = build_feature_matrix(df, config).to_numpy()
    dB = df[COL_DELTA_B].to_numpy().astype(np.float64)
    from sklearn.linear_model import ElasticNetCV as _ENCV
    from sklearn.model_selection import TimeSeriesSplit as _TSS

    cv = _TSS(n_splits=5)
    model_cv = _ENCV(
        l1_ratio=config.elastic_net_l1_ratio,
        cv=cv,
        fit_intercept=False,
        max_iter=10_000,
    )
    model_cv.fit(A, dB)

    expected_user_alpha = float(model_cv.alpha_) / n_rows
    assert result.selected_alpha == pytest.approx(expected_user_alpha, rel=1e-5)  # pyright: ignore[reportUnknownMemberType]
```

**Step 2: Run to confirm they fail**

```bash
uv run pytest tests/unit/test_calibration.py::test_elastic_net_uses_n_samples_scaled_alpha_internally tests/unit/test_calibration.py::test_elastic_net_cv_selected_alpha_in_unnormalized_convention -v
```

Expected: both FAIL.

---

### Task 6: Fix ElasticNet alpha scaling in `calibrate()`

**Files:**
- Modify: `lmc/calibration.py:208-218`

**Step 1: Apply the fix**

Replace the non-CV ElasticNet block (currently lines 209-217):

```python
# BEFORE:
            model_e = ElasticNet(
                alpha=config.elastic_net_alpha,
                l1_ratio=config.elastic_net_l1_ratio,
                fit_intercept=False,
                max_iter=10_000,
            )
            model_e.fit(A, dB)  # pyright: ignore[reportUnknownMemberType]
            coefficients = np.asarray(model_e.coef_, dtype=np.float64)
            selected_alpha = config.elastic_net_alpha
```

```python
# AFTER:
            # Alpha convention: same unnormalized (ridge) convention.
            # Multiply by n_samples to compensate for sklearn's normalization.
            n_samples = A.shape[0]
            model_e = ElasticNet(
                alpha=config.elastic_net_alpha * n_samples,
                l1_ratio=config.elastic_net_l1_ratio,
                fit_intercept=False,
                max_iter=10_000,
            )
            model_e.fit(A, dB)  # pyright: ignore[reportUnknownMemberType]
            coefficients = np.asarray(model_e.coef_, dtype=np.float64)
            selected_alpha = config.elastic_net_alpha
```

Replace the CV ElasticNet selected_alpha line (currently line 207):

```python
# BEFORE:
            selected_alpha = float(model_ecv.alpha_)
```

```python
# AFTER:
            # Convert sklearn convention → user convention (divide by n_samples).
            selected_alpha = float(model_ecv.alpha_) / A.shape[0]
```

**Step 2: Run Task 5 tests to verify**

```bash
uv run pytest tests/unit/test_calibration.py::test_elastic_net_uses_n_samples_scaled_alpha_internally tests/unit/test_calibration.py::test_elastic_net_cv_selected_alpha_in_unnormalized_convention -v
```

Expected: both PASS.

**Step 3: Run full ElasticNet test group to find regressions**

```bash
uv run pytest tests/unit/test_calibration.py -k "elastic" -v
```

Expected: `test_elastic_net_recovers_reasonable` FAILS (same reason as lasso recovery test).

---

### Task 7: Update `test_elastic_net_recovers_reasonable` for new alpha convention

**Files:**
- Modify: `tests/unit/test_calibration.py:357-370`

**Step 1: Update the test**

```python
# BEFORE:
def test_elastic_net_recovers_reasonable() -> None:
    """ElasticNet should return plausible, finite coefficients."""
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=1e-3,
        elastic_net_l1_ratio=0.5,
    )
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)
```

```python
# AFTER:
def test_elastic_net_recovers_reasonable() -> None:
    """ElasticNet should return plausible, finite coefficients.

    Uses a small elastic_net_alpha (unnormalized convention) so the scaled
    sklearn alpha remains weak and near-exact recovery is expected.
    """
    c_true = np.array([1.0, -2.0, 0.5])
    config = PipelineConfig(
        model_terms="a",
        use_elastic_net=True,
        elastic_net_alpha=1e-5,
        elastic_net_l1_ratio=0.5,
    )
    df, segments = _make_synthetic_data(c_true, config)
    result = calibrate(df, segments, config)
    assert result.coefficients.shape == (3,)
    assert np.all(np.isfinite(result.coefficients))
    np.testing.assert_allclose(result.coefficients, c_true, atol=0.1)
```

**Step 2: Run elastic net tests**

```bash
uv run pytest tests/unit/test_calibration.py -k "elastic" -v
```

Expected: all PASS.

**Step 3: Commit ElasticNet changes**

```bash
git add lmc/calibration.py tests/unit/test_calibration.py
git commit -m "fix: normalize elastic_net_alpha to unnormalized (ridge) convention

Multiply elastic_net_alpha by n_samples before passing to sklearn ElasticNet,
matching the convention established for lasso_alpha. For CV, divide returned
alpha_ by n_samples to store in user-facing unnormalized convention."
```

---

### Task 8: Update field descriptions and add decision comment

**Files:**
- Modify: `lmc/config.py:70-85`
- Modify: `lmc/calibration.py` (add comment block before lasso branch)

**Step 1: Update `lasso_alpha` field description**

```python
# BEFORE:
    lasso_alpha: float = Field(
        default=1e-3,
        ge=0.0,
        description="LASSO regularisation strength. Ignored when use_lasso is False.",
    )
```

```python
# AFTER:
    lasso_alpha: float = Field(
        default=1e-3,
        ge=0.0,
        description=(
            "LASSO regularisation strength in the unnormalized convention: "
            "the user-visible alpha corresponds to the objective "
            "||Aw - dB||² + alpha * ||w||₁, matching ridge_alpha semantics. "
            "Internally, calibrate() passes alpha * n_samples to sklearn's Lasso "
            "to compensate for sklearn's (1/2n)-normalized loss. "
            "Ignored when use_lasso is False."
        ),
    )
```

**Step 2: Update `elastic_net_alpha` field description**

```python
# BEFORE:
    elastic_net_alpha: float = Field(
        default=1e-3,
        ge=0.0,
        description=(
            "ElasticNet regularisation strength. Ignored when use_elastic_net is False."
        ),
    )
```

```python
# AFTER:
    elastic_net_alpha: float = Field(
        default=1e-3,
        ge=0.0,
        description=(
            "ElasticNet regularisation strength in the unnormalized convention: "
            "the user-visible alpha corresponds to the objective "
            "||Aw - dB||² + alpha * (l1_ratio*||w||₁ + (1-l1_ratio)*||w||²/2), "
            "matching ridge_alpha semantics. "
            "Internally, calibrate() passes alpha * n_samples to sklearn's ElasticNet "
            "to compensate for sklearn's (1/2n)-normalized loss. "
            "Ignored when use_elastic_net is False."
        ),
    )
```

**Step 3: Verify linting passes**

```bash
uv run make lint
```

Expected: no errors.

**Step 4: Commit documentation**

```bash
git add lmc/config.py
git commit -m "docs: document unnormalized alpha convention for lasso and elastic net fields"
```

---

### Task 9: Run full test suite and verify

**Step 1: Run all tests**

```bash
uv run make test
```

Expected: all tests pass with no regressions.

**Step 2: Run linting**

```bash
uv run make lint
```

Expected: clean.

**Step 3: If all passes, final commit (if any uncommitted changes remain)**

```bash
git status
```

Only commit if there are unstaged changes not yet committed.
