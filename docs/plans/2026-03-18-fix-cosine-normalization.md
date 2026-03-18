# Fix Direction Cosines Normalization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix direction cosines to normalize by fluxgate vector magnitude instead of scalar magnetometer total field (issue #46).

**Architecture:** Single function fix in `_direction_cosines()` plus test updates to use realistic sensor values that prevent the bug from being masked.

**Tech Stack:** Python, NumPy, Polars, pytest

---

### Task 1: Update test fixtures to use realistic COL_BTOTAL

**Files:**
- Modify: `tests/unit/test_features.py:58-102`

**Step 1: Update `_make_df` and `_make_linear_df` to use realistic COL_BTOTAL**

Change `_make_df()` so `COL_BTOTAL` is a realistic scalar magnetometer value (54000.0) that differs from the fluxgate magnitude (5.0). This ensures any code that accidentally divides by `COL_BTOTAL` will produce visibly wrong results.

In `_make_df`, replace the `btotal = math.sqrt(...)` line with a hardcoded realistic value:

```python
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
```

In `_make_linear_df`, change `btotal = 10.0` to `btotal = 54000.0` and compute B vectors from the fluxgate magnitude (10.0) instead:

```python
def _make_linear_df() -> pl.DataFrame:
    """Return a DataFrame with linearly varying direction cosines.

    cos_x goes [0.6, 0.7, 0.8] across t = [0, 1, 2].
    B vectors are constructed so that fluxgate magnitude stays at 10.0 and
    direction cosines match the desired values at each row.
    """
    cos_x_vals = [0.6, 0.7, 0.8]
    cos_y_vals = [0.8, 0.7, 0.6]
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
```

**Step 2: Run tests to confirm they now fail (proving the bug)**

Run: `make test`
Expected: Multiple failures in `test_features.py` — direction cosine values will be ~10,800x too small because the code still divides by `COL_BTOTAL` (54000) instead of fluxgate magnitude (5.0).

**Step 3: Commit the test fixture changes**

```bash
git add tests/unit/test_features.py
git commit -m "test: use realistic COL_BTOTAL in feature test fixtures

Fixtures now use COL_BTOTAL=54000 (scalar magnetometer) which differs
from the fluxgate magnitude, exposing the normalization bug (#46)."
```

---

### Task 2: Fix `_direction_cosines` to normalize by fluxgate magnitude

**Files:**
- Modify: `lmc/features.py:9-10,41-64`

**Step 1: Fix the implementation**

In `lmc/features.py`, remove `COL_BTOTAL` from the import list (line 10) and update `_direction_cosines`:

```python
def _direction_cosines(
    df: pl.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute direction cosines from fluxgate magnetometer readings.

    Returns (cos_x, cos_y, cos_z) = (B_x/|B|, B_y/|B|, B_z/|B|).
    """
    b_x = np.asarray(df[COL_BX].to_numpy(), dtype=np.float64)
    b_y = np.asarray(df[COL_BY].to_numpy(), dtype=np.float64)
    b_z = np.asarray(df[COL_BZ].to_numpy(), dtype=np.float64)

    b_magnitude = np.sqrt(b_x**2 + b_y**2 + b_z**2)

    if np.any(b_magnitude <= 0.0):
        raise ValueError("Fluxgate magnitude must be strictly positive for all rows.")

    cos_x = b_x / b_magnitude
    cos_y = b_y / b_magnitude
    cos_z = b_z / b_magnitude

    return (
        np.asarray(cos_x, dtype=np.float64),
        np.asarray(cos_y, dtype=np.float64),
        np.asarray(cos_z, dtype=np.float64),
    )
```

**Step 2: Run tests to confirm they pass**

Run: `make test`
Expected: All existing tests pass — direction cosines are now correctly normalized.

**Step 3: Commit the fix**

```bash
git add lmc/features.py
git commit -m "fix: normalize direction cosines by fluxgate magnitude, not B_total

Direction cosines were divided by COL_BTOTAL (scalar magnetometer,
~54,000 nT) instead of the fluxgate vector magnitude (~50 nT),
making them ~1000x too small and producing a near-degenerate
feature matrix.

Closes #46"
```

---

### Task 3: Add regression test and update error-condition tests

**Files:**
- Modify: `tests/unit/test_features.py:280-311`

**Step 1: Add unit-vector regression test**

Add a new test after `test_direction_cosines_values` that asserts the unit-vector property:

```python
def test_direction_cosines_unit_vector() -> None:
    """cos_x^2 + cos_y^2 + cos_z^2 must be approximately 1.0 for all rows."""
    df = _make_df()
    result = build_feature_matrix(df, _CONFIG_A)
    cos_x = result[COL_COS_X].to_numpy()
    cos_y = result[COL_COS_Y].to_numpy()
    cos_z = result[COL_COS_Z].to_numpy()
    norms = cos_x**2 + cos_y**2 + cos_z**2
    assert all(math.isclose(n, 1.0, rel_tol=1e-9) for n in norms)
```

**Step 2: Run to verify it passes**

Run: `pytest tests/unit/test_features.py::test_direction_cosines_unit_vector -v`
Expected: PASS

**Step 3: Replace `test_raises_zero_b_total` and `test_raises_negative_b_total`**

These tested `COL_BTOTAL <= 0` which is no longer checked in `_direction_cosines`. Replace both with a single test for zero fluxgate magnitude:

```python
def test_raises_zero_fluxgate_magnitude() -> None:
    """All-zero fluxgate components → zero magnitude → ValueError."""
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
```

Delete `test_raises_zero_b_total` and `test_raises_negative_b_total`.

**Step 4: Run full test suite**

Run: `make test`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add tests/unit/test_features.py
git commit -m "test: add unit-vector assertion and zero-fluxgate error test

Adds test_direction_cosines_unit_vector to guard against normalization
regressions. Replaces B_total positivity tests with a fluxgate
magnitude test matching the updated guard in _direction_cosines."
```

---

### Task 4: Lint and final verification

**Step 1: Run linter**

Run: `make lint`
Expected: Clean — no warnings or errors.

**Step 2: Run full test suite one more time**

Run: `make test`
Expected: All tests pass.

**Step 3: Fix any issues found, then commit if needed**
