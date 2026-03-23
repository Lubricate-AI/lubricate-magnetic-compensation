# Causal Derivative Features for CV Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `np.gradient` (central differences) with backward differences in `_cosine_derivatives()` when `use_cv=True`, eliminating train/test data leakage at CV fold boundaries.

**Architecture:** Add a `causal: bool` parameter to `_cosine_derivatives()`. When `causal=True`, compute `dcos[i] = (cos[i] - cos[i-1]) / (t[i] - t[i-1])` for `i >= 1` and replicate the first difference for row 0 (forward-pad). `build_feature_matrix()` passes `causal=config.use_cv`. The non-causal (central difference) path is unchanged for `use_cv=False`.

**Tech Stack:** Python 3.12, NumPy (`np.diff`), Polars, pytest

---

### Task 1: Write a failing test for causal backward differences

**Files:**
- Modify: `tests/unit/test_features.py`

**Step 1: Write the failing test**

Add these tests after the existing eddy-term value tests (around line 320 in the file):

```python
# ---------------------------------------------------------------------------
# Causal (use_cv=True) derivative path
# ---------------------------------------------------------------------------

_CONFIG_C_CV = PipelineConfig(model_terms="c", use_cv=True, use_ridge=True)
_CONFIG_D_CV = PipelineConfig(model_terms="d", use_cv=True, use_ridge=True)


def test_causal_derivatives_no_leakage_at_boundary() -> None:
    """With use_cv=True, dcos at row i must not depend on row i+1.

    Build a 4-row df where cos values differ at each row.
    Compute the feature matrix. The derivative at row 2 (boundary) must
    equal the backward diff (cos[2]-cos[1])/(t[2]-t[1]), NOT the central
    diff that would pull in cos[3].
    """
    # cos_x = [0.6, 0.7, 0.8, 0.9], linear, so all diffs equal 0.1/dt=0.1
    # cos_y = [0.8, 0.714..., 0.6, 0.436...] — unit vectors with the above cos_x
    # Use a simple setup: constant direction with cos_x changing linearly
    fluxgate_mag = 10.0
    # cos_x at rows 0..3: 0.6, 0.7, 0.8, 0.9 => cos_y = sqrt(1 - cos_x^2), cos_z=0
    cos_x_vals = [0.6, 0.7, 0.8, 0.9]
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

    # Backward diff at row 2: (cos_x[2] - cos_x[1]) / (t[2] - t[1]) = 0.1
    expected_dcos_x_row2 = (cos_x_vals[2] - cos_x_vals[1]) / (2.0 - 1.0)
    # Central diff at row 2 would be: (cos_x[3] - cos_x[1]) / (3.0 - 1.0) = 0.15
    # If central diff is used, this test fails. If backward diff, it passes.
    actual_dcos_x_row2 = float(result[COL_COS_X_DCOS_X][2]) / cos_x_vals[2]
    assert math.isclose(actual_dcos_x_row2, expected_dcos_x_row2, rel_tol=1e-9), (
        f"Expected backward diff {expected_dcos_x_row2}, got {actual_dcos_x_row2}. "
        "Central diff would give 0.15 — likely use_cv=True is not using causal diffs."
    )


def test_causal_derivatives_first_row_replicated() -> None:
    """Row 0 gets the same derivative as row 1 (forward-padded)."""
    fluxgate_mag = 10.0
    cos_x_vals = [0.6, 0.7, 0.8, 0.9]
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

    # Row 0 should equal row 1 (replicated forward-pad)
    dcos_x_row0 = float(result[COL_COS_X_DCOS_X][0]) / cos_x_vals[0]
    dcos_x_row1 = float(result[COL_COS_X_DCOS_X][1]) / cos_x_vals[1]
    # Both are 0.1 / 1.0 = 0.1 (backward diff for rows 1..3 all equal 0.1)
    assert math.isclose(dcos_x_row0 / cos_x_vals[0] * cos_x_vals[0],
                        dcos_x_row1 / cos_x_vals[1] * cos_x_vals[1],
                        rel_tol=1e-9), (
        "Row 0 derivative should be replicated from the first backward diff (row 1)."
    )


def test_non_cv_path_unchanged() -> None:
    """use_cv=False still uses central differences (np.gradient)."""
    df = _make_linear_df()

    result_cv_off = build_feature_matrix(df, _CONFIG_C)   # central diffs
    result_cv_on  = build_feature_matrix(df, _CONFIG_C_CV)  # backward diffs

    # For the middle row (index 1) of a 3-row df:
    #   central diff: (1.0 - 0.6) / (2*1) = 0.2
    #   backward diff: (0.8 - 0.6) / (1.0 - 0.0) = 0.2
    # They happen to be equal here because spacing is uniform and linear.
    # BUT first row differs:
    #   central diff at row 0 = forward diff = (cos[1]-cos[0])/(t[1]-t[0]) = 0.2
    #   backward-padded row 0 = backward diff at row 1 = (cos[1]-cos[0])/(t[1]-t[0]) = 0.2
    # Also last row differs (use a 4-row df to see the difference):
    pass  # The existing test_eddy_terms_values_linear_signal covers the central-diff path.
```

> Note: The `test_non_cv_path_unchanged` is intentionally a pass — the existing
> `test_eddy_terms_values_linear_signal` already covers the central-diff (non-CV) path.
> The two new substantive tests are `test_causal_derivatives_no_leakage_at_boundary`
> and `test_causal_derivatives_first_row_replicated`.

**Step 2: Run tests to verify they fail**

```bash
cd /Users/joshuapoirier/conductor/workspaces/lubricate-magnetic-compensation/bozeman
uv run pytest tests/unit/test_features.py::test_causal_derivatives_no_leakage_at_boundary tests/unit/test_features.py::test_causal_derivatives_first_row_replicated -v
```

Expected: Both tests FAIL (because `_cosine_derivatives` still uses `np.gradient` regardless of `use_cv`).

---

### Task 2: Add `causal` parameter to `_cosine_derivatives()`

**Files:**
- Modify: `lmc/features.py:70-84`

**Step 1: Write the minimal implementation**

Replace the `_cosine_derivatives` function body:

```python
def _cosine_derivatives(
    cos_x: npt.NDArray[np.float64],
    cos_y: npt.NDArray[np.float64],
    cos_z: npt.NDArray[np.float64],
    time: npt.NDArray[np.float64],
    *,
    causal: bool = False,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute time derivatives of direction cosines.

    When ``causal=False`` (default), uses ``np.gradient`` (central differences)
    — each row may depend on its neighbours.

    When ``causal=True``, uses backward differences so that row *i* depends only
    on samples *i* and *i-1* (no look-ahead).  The first row is padded by
    replicating the first backward-difference value so the output length matches
    the input.  Use this path when ``config.use_cv=True`` to avoid train/test
    leakage at ``TimeSeriesSplit`` fold boundaries.

    Uses explicit time coordinates to handle non-uniform sampling.
    Returns (dcos_x/dt, dcos_y/dt, dcos_z/dt).
    """
    if causal:
        dt = np.diff(time)                    # shape (n-1,)
        def _backward(cos: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            diffs = np.diff(cos) / dt         # shape (n-1,)
            return np.asarray(
                np.concatenate([[diffs[0]], diffs]), dtype=np.float64
            )
        return _backward(cos_x), _backward(cos_y), _backward(cos_z)

    dcos_x = np.asarray(np.gradient(cos_x, time), dtype=np.float64)
    dcos_y = np.asarray(np.gradient(cos_y, time), dtype=np.float64)
    dcos_z = np.asarray(np.gradient(cos_z, time), dtype=np.float64)
    return dcos_x, dcos_y, dcos_z
```

**Step 2: Update the call site in `build_feature_matrix` (line ~168)**

Change:
```python
dcos_x, dcos_y, dcos_z = _cosine_derivatives(cos_x, cos_y, cos_z, time)
```
To:
```python
dcos_x, dcos_y, dcos_z = _cosine_derivatives(
    cos_x, cos_y, cos_z, time, causal=config.use_cv
)
```

**Step 3: Run new tests to verify they pass**

```bash
uv run pytest tests/unit/test_features.py::test_causal_derivatives_no_leakage_at_boundary tests/unit/test_features.py::test_causal_derivatives_first_row_replicated -v
```

Expected: Both PASS.

**Step 4: Run the full test suite to verify no regressions**

```bash
uv run pytest tests/unit/test_features.py -v
```

Expected: All existing tests still PASS (the central-diff path is unchanged for `use_cv=False`).

**Step 5: Commit**

```bash
git add lmc/features.py tests/unit/test_features.py
git commit -m "fix: use causal backward differences for derivatives when use_cv=True

Prevents train/test leakage at TimeSeriesSplit fold boundaries for C/D
model terms. np.gradient (central diffs) is retained for use_cv=False.

Closes #72"
```

---

### Task 3: Run linting and full test suite

**Files:**
- No changes — verification only.

**Step 1: Run linting**

```bash
uv run make lint
```

Expected: No errors. If ruff complains about the nested `_backward` function, refactor to a module-level helper or inline it. If typos flags something, fix spelling.

**Step 2: Run full test suite**

```bash
uv run pytest -v
```

Expected: All tests pass.

**Step 3: Commit any lint fixes (if needed)**

```bash
git add -p
git commit -m "fix: resolve lint issues in causal derivative implementation"
```

---

### Task 4: Create PR

**Step 1: Push the branch**

```bash
git push -u origin 72-fix-causal-derivative-features-for-time-series-cv-with-cd-model-terms
```

**Step 2: Create PR**

```bash
gh pr create \
  --repo Lubricate-AI/lubricate-magnetic-compensation \
  --title "fix: causal derivative features for time-series CV with C/D model terms" \
  --body "$(cat <<'EOF'
## INFO

**What:** Replace `np.gradient` (central differences) with backward differences in `_cosine_derivatives()` when `config.use_cv=True`.

**Key changes:**
- `lmc/features.py`: `_cosine_derivatives()` gains a `causal: bool = False` keyword parameter. When `True`, computes backward differences (`np.diff` / `np.diff(time)`) and forward-pads row 0 by replicating the first difference. The `np.gradient` path is unchanged for `causal=False`.
- `build_feature_matrix()` passes `causal=config.use_cv` to `_cosine_derivatives()`.
- `tests/unit/test_features.py`: Two new tests verify (a) no look-ahead at fold boundaries and (b) correct row-0 padding.

**Breaking changes:** None. The default `causal=False` preserves existing behaviour for all non-CV usage.

**Testing:** `make test` passes. New tests cover the causal path directly.

**TODOs:** None.

## REFERENCES

Closes #72
PR #71 (feat: cross-validation and automatic regularization method selection)
EOF
)"
```
