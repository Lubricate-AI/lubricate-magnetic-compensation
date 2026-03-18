# Fix: Direction cosines normalization uses wrong denominator

**Issue:** #46
**Date:** 2026-03-18

## Problem

`_direction_cosines()` in `lmc/features.py` normalizes fluxgate components by `COL_BTOTAL` (scalar magnetometer, ~54,000 nT) instead of the fluxgate vector magnitude `sqrt(B_x^2 + B_y^2 + B_z^2)` (~50 nT). This makes direction cosines ~1000x too small, producing a near-degenerate feature matrix.

## Approach

Minimal fix in `_direction_cosines` plus test updates to prevent regression.

## Changes

### `lmc/features.py` — `_direction_cosines()`

- Remove `COL_BTOTAL` read from this function.
- Compute `b_magnitude = np.sqrt(b_x**2 + b_y**2 + b_z**2)`.
- Guard: raise `ValueError` if any `b_magnitude <= 0.0`.
- Normalize: `cos_i = b_i / b_magnitude`.

### `tests/unit/test_features.py`

- Change `_make_df()` to use a realistic `COL_BTOTAL` (e.g., 54000.0) distinct from fluxgate magnitude, so the bug can't be masked.
- Similarly update `_make_linear_df()`.
- Add `test_direction_cosines_unit_vector` asserting `cos_x^2 + cos_y^2 + cos_z^2 ~ 1.0`.
- Replace `test_raises_zero_b_total` / `test_raises_negative_b_total` with a test for zero fluxgate magnitude.

### Unchanged

- `COL_BTOTAL` remains a required column used elsewhere in the pipeline.
- All other functions in `features.py` are unchanged.
