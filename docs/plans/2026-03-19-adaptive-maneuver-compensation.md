# Adaptive Maneuver-Based Compensation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fit separate Tolles-Lawson coefficients per maneuver type (pitch/roll/yaw/baseline) and blend them at compensation time using rolling-variance-based weights derived from fluxgate direction cosines.

**Architecture:** A new `lmc/adaptive.py` module owns the `AdaptiveCalibrationResult` dataclass plus two public functions (`calibrate_adaptive_maneuvers`, `compensate_adaptive`). These call the existing `calibrate()` and `build_feature_matrix()` helpers rather than duplicating logic. Three new optional fields are added to `PipelineConfig`.

**Tech Stack:** Python 3.12+, NumPy (vectorised rolling variance), Polars (DataFrame I/O), Pydantic (config validation), pytest (TDD).

---

## Background: How the Existing Pipeline Works

Before writing any code, read these files top-to-bottom:

| File | Purpose |
|------|---------|
| `lmc/columns.py` | All column-name constants (`COL_BTOTAL`, `COL_BX` … `COL_TMI_COMPENSATED`) |
| `lmc/config.py` | `PipelineConfig` Pydantic model — **add new fields here** |
| `lmc/calibration.py` | `CalibrationResult` frozen dataclass + `calibrate()` |
| `lmc/compensation.py` | `compensate()` — see the pattern we replicate for adaptive |
| `lmc/features.py` | `build_feature_matrix(df, config)` — returns polars DataFrame with A-matrix |
| `lmc/segmentation.py` | `Segment` dataclass, `ManeuverType = Literal["steady","pitch","roll","yaw"]` |
| `lmc/__init__.py` | Public API — add exports here last |
| `tests/unit/test_calibration.py` | See `_make_synthetic_df` helper — copy this pattern for tests |
| `tests/integration/synthetic.py` | `make_fom_dataframe()` — used by integration tests |
| `tests/integration/test_integration_simple.py` | Full pipeline test — model the integration test on this |

Key invariants to preserve:
- `compensate_adaptive` must output `COL_TMI_COMPENSATED = "tmi_compensated"` (same column as `compensate`).
- Direction cosines are computed as `B_x / ‖B‖_flux` where `‖B‖_flux = sqrt(B_x² + B_y² + B_z²)` — **not** divided by `B_total` (fixed in commit 77a619c).
- Feature matrix first 3 columns are always `[cos_x, cos_y, cos_z]` for all `model_terms`.

---

## Task 1: Add Three New Fields to PipelineConfig

**Files:**
- Modify: `lmc/config.py`
- Modify: `tests/unit/test_config.py`

### Step 1: Write three failing tests

Add to `tests/unit/test_config.py`:

```python
def test_compensation_strategy_default_is_standard() -> None:
    config = PipelineConfig()
    assert config.compensation_strategy == "standard"


def test_compensation_strategy_accepts_adaptive_maneuver() -> None:
    config = PipelineConfig(compensation_strategy="adaptive_maneuver")
    assert config.compensation_strategy == "adaptive_maneuver"


def test_maneuver_detection_window_default() -> None:
    config = PipelineConfig()
    assert config.maneuver_detection_window == 50


def test_maneuver_baseline_weight_default() -> None:
    config = PipelineConfig()
    assert config.maneuver_baseline_weight == 0.1


def test_maneuver_detection_window_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(maneuver_detection_window=0)


def test_maneuver_baseline_weight_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(maneuver_baseline_weight=0.0)
```

Make sure `from pydantic import ValidationError` is imported at the top of the test file.

### Step 2: Run tests to verify they fail

```
pytest tests/unit/test_config.py -v -k "compensation_strategy or maneuver_detection or maneuver_baseline"
```

Expected: FAIL — `PipelineConfig` does not have these attributes yet.

### Step 3: Add the three fields to `lmc/config.py`

Inside the `PipelineConfig` class body, after the `use_imu_rates` field:

```python
compensation_strategy: Literal["standard", "adaptive_maneuver"] = Field(
    default="standard",
    description=(
        "Compensation strategy: 'standard' uses a single coefficient set; "
        "'adaptive_maneuver' blends per-maneuver coefficients based on "
        "detected maneuver intensity."
    ),
)
maneuver_detection_window: int = Field(
    default=50,
    gt=0,
    description=(
        "Rolling window size [samples] for computing direction-cosine variance "
        "used to detect maneuver intensity. "
        "At 10 Hz, 50 samples ≈ 5 seconds."
    ),
)
maneuver_baseline_weight: float = Field(
    default=0.1,
    gt=0.0,
    description=(
        "Constant additive weight for baseline coefficients during blending. "
        "Keeps a small baseline contribution even when a strong maneuver is "
        "detected, preventing zero-weight extrapolation."
    ),
)
```

Also add `"adaptive_maneuver"` to the `Literal` — Pydantic will enforce valid values.

### Step 4: Run tests to verify they pass

```
pytest tests/unit/test_config.py -v -k "compensation_strategy or maneuver_detection or maneuver_baseline"
```

Expected: PASS (6 tests).

### Step 5: Run full lint + test suite to check for regressions

```
make lint && make test
```

Expected: all pass.

### Step 6: Commit

```bash
git add lmc/config.py tests/unit/test_config.py
git commit -m "feat: add adaptive maneuver config fields to PipelineConfig"
```

---

## Task 2: Create `AdaptiveCalibrationResult` Dataclass

**Files:**
- Create: `lmc/adaptive.py`
- Create: `tests/unit/test_adaptive.py`

### Step 1: Write failing tests

Create `tests/unit/test_adaptive.py`:

```python
"""Unit tests for lmc.adaptive."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import pytest

from lmc.adaptive import AdaptiveCalibrationResult
from lmc.calibration import CalibrationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_cal_result(n_terms: int) -> CalibrationResult:
    """Build a minimal CalibrationResult for use in tests."""
    return CalibrationResult(
        coefficients=np.ones(n_terms, dtype=np.float64),
        residuals=np.zeros(10, dtype=np.float64),
        condition_number=1.0,
        n_terms=n_terms,
    )


# ---------------------------------------------------------------------------
# AdaptiveCalibrationResult tests
# ---------------------------------------------------------------------------

def test_adaptive_result_stores_four_sub_results() -> None:
    r = _dummy_cal_result(3)
    result = AdaptiveCalibrationResult(pitch=r, roll=r, yaw=r, baseline=r, n_terms=3)
    assert result.pitch is r
    assert result.roll is r
    assert result.yaw is r
    assert result.baseline is r


def test_adaptive_result_stores_n_terms() -> None:
    r = _dummy_cal_result(9)
    result = AdaptiveCalibrationResult(pitch=r, roll=r, yaw=r, baseline=r, n_terms=9)
    assert result.n_terms == 9


def test_adaptive_result_is_frozen() -> None:
    r = _dummy_cal_result(3)
    result = AdaptiveCalibrationResult(pitch=r, roll=r, yaw=r, baseline=r, n_terms=3)
    with pytest.raises((AttributeError, TypeError)):
        result.n_terms = 9  # type: ignore[misc]
```

### Step 2: Run tests to verify they fail

```
pytest tests/unit/test_adaptive.py -v
```

Expected: FAIL — `lmc.adaptive` module does not exist.

### Step 3: Create `lmc/adaptive.py` with just the dataclass

```python
"""Adaptive maneuver-based compensation with per-maneuver coefficient blending."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.calibration import CalibrationResult, calibrate
from lmc.columns import COL_BTOTAL, COL_BX, COL_BY, COL_BZ, COL_TMI_COMPENSATED
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import Segment


@dataclass(frozen=True)
class AdaptiveCalibrationResult:
    """Per-maneuver calibration results used for adaptive blending.

    Attributes
    ----------
    pitch:
        Coefficients fitted on pitch maneuver segments only.
    roll:
        Coefficients fitted on roll maneuver segments only.
    yaw:
        Coefficients fitted on yaw maneuver segments only.
    baseline:
        Coefficients fitted on steady (non-maneuver) segments.
    n_terms:
        Number of model terms (must match across all sub-results).
    """

    pitch: CalibrationResult
    roll: CalibrationResult
    yaw: CalibrationResult
    baseline: CalibrationResult
    n_terms: int
```

### Step 4: Run tests to verify they pass

```
pytest tests/unit/test_adaptive.py::test_adaptive_result_stores_four_sub_results tests/unit/test_adaptive.py::test_adaptive_result_stores_n_terms tests/unit/test_adaptive.py::test_adaptive_result_is_frozen -v
```

Expected: PASS (3 tests).

### Step 5: Commit

```bash
git add lmc/adaptive.py tests/unit/test_adaptive.py
git commit -m "feat: add AdaptiveCalibrationResult dataclass"
```

---

## Task 3: Implement `calibrate_adaptive_maneuvers`

**Files:**
- Modify: `lmc/adaptive.py`
- Modify: `tests/unit/test_adaptive.py`

### Step 1: Write failing tests

Append to `tests/unit/test_adaptive.py`:

```python
from lmc.adaptive import calibrate_adaptive_maneuvers
from lmc.columns import (
    COL_ALT, COL_BTOTAL, COL_BX, COL_BY, COL_BZ,
    COL_DELTA_B, COL_LAT, COL_LON, COL_TIME,
)
from lmc.features import build_feature_matrix


# ---------------------------------------------------------------------------
# Synthetic data helpers (adapted from test_calibration.py)
# ---------------------------------------------------------------------------

def _make_base_df(n_rows: int, rng: np.random.Generator) -> pl.DataFrame:
    """DataFrame with random unit B vectors but no delta_B."""
    raw = rng.standard_normal((n_rows, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms
    b_total = 50_000.0
    bx = cosines[:, 0] * b_total
    by = cosines[:, 1] * b_total
    bz = cosines[:, 2] * b_total
    return pl.DataFrame({
        COL_TIME: np.arange(n_rows, dtype=np.float64),
        COL_LAT: np.full(n_rows, 45.0),
        COL_LON: np.full(n_rows, -75.0),
        COL_ALT: np.full(n_rows, 0.3),
        COL_BTOTAL: np.full(n_rows, b_total),
        COL_BX: bx,
        COL_BY: by,
        COL_BZ: bz,
    })


def _make_adaptive_calibration_data(
    n_rows_each: int = 40,
    seed: int = 0,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build (df, segments) with one segment per maneuver type."""
    from lmc.segmentation import Segment

    config = PipelineConfig(model_terms="a")
    rng = np.random.default_rng(seed)
    c_true = np.array([1.0, -2.0, 0.5])

    blocks = []
    segs = []
    offset = 0
    for maneuver in ["steady", "pitch", "roll", "yaw"]:
        block = _make_base_df(n_rows_each, rng)
        A = build_feature_matrix(block, config).to_numpy()
        delta_b = A @ c_true + rng.normal(0, 0.01, n_rows_each)
        block = block.with_columns(
            pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64)
        )
        blocks.append(block)
        segs.append(Segment(maneuver=maneuver, heading="N",  # type: ignore[arg-type]
                            start_idx=offset, end_idx=offset + n_rows_each))
        offset += n_rows_each

    df = pl.concat(blocks)
    return df, segs


# ---------------------------------------------------------------------------
# calibrate_adaptive_maneuvers tests
# ---------------------------------------------------------------------------

def test_adaptive_calibration_returns_correct_type() -> None:
    df, segments = _make_adaptive_calibration_data()
    config = PipelineConfig(model_terms="a")
    result = calibrate_adaptive_maneuvers(df, segments, config)
    assert isinstance(result, AdaptiveCalibrationResult)


def test_adaptive_calibration_n_terms_matches_config() -> None:
    df, segments = _make_adaptive_calibration_data()
    config = PipelineConfig(model_terms="a")
    result = calibrate_adaptive_maneuvers(df, segments, config)
    assert result.n_terms == 3
    assert result.pitch.n_terms == 3
    assert result.roll.n_terms == 3
    assert result.yaw.n_terms == 3
    assert result.baseline.n_terms == 3


def test_adaptive_calibration_raises_if_pitch_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_pitch = [s for s in segments if s.maneuver != "pitch"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="pitch"):
        calibrate_adaptive_maneuvers(df, non_pitch, config)


def test_adaptive_calibration_raises_if_roll_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_roll = [s for s in segments if s.maneuver != "roll"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="roll"):
        calibrate_adaptive_maneuvers(df, non_roll, config)


def test_adaptive_calibration_raises_if_yaw_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_yaw = [s for s in segments if s.maneuver != "yaw"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="yaw"):
        calibrate_adaptive_maneuvers(df, non_yaw, config)


def test_adaptive_calibration_raises_if_steady_segments_missing() -> None:
    df, segments = _make_adaptive_calibration_data()
    non_steady = [s for s in segments if s.maneuver != "steady"]
    config = PipelineConfig(model_terms="a")
    with pytest.raises(ValueError, match="steady"):
        calibrate_adaptive_maneuvers(df, non_steady, config)
```

### Step 2: Run tests to verify they fail

```
pytest tests/unit/test_adaptive.py -v -k "calibrate_adaptive"
```

Expected: FAIL — `calibrate_adaptive_maneuvers` not yet defined.

### Step 3: Implement `calibrate_adaptive_maneuvers` in `lmc/adaptive.py`

Add after the `AdaptiveCalibrationResult` dataclass:

```python
def calibrate_adaptive_maneuvers(
    df: pl.DataFrame,
    segments: list[Segment],
    config: PipelineConfig,
) -> AdaptiveCalibrationResult:
    """Fit separate Tolles-Lawson coefficients for each maneuver type.

    Parameters
    ----------
    df:
        Full calibration DataFrame including ``COL_DELTA_B``.
    segments:
        Labeled segments covering all four maneuver types:
        ``"steady"``, ``"pitch"``, ``"roll"``, ``"yaw"``.
    config:
        Pipeline configuration (same ``model_terms`` used for all fits).

    Returns
    -------
    AdaptiveCalibrationResult
        Four ``CalibrationResult`` objects, one per maneuver type.

    Raises
    ------
    ValueError
        If any of the four required maneuver types has no segments.
    """
    pitch_segs = [s for s in segments if s.maneuver == "pitch"]
    roll_segs = [s for s in segments if s.maneuver == "roll"]
    yaw_segs = [s for s in segments if s.maneuver == "yaw"]
    baseline_segs = [s for s in segments if s.maneuver == "steady"]

    for name, segs in [
        ("pitch", pitch_segs),
        ("roll", roll_segs),
        ("yaw", yaw_segs),
        ("steady", baseline_segs),
    ]:
        if not segs:
            raise ValueError(
                f"No '{name}' segments found. "
                f"calibrate_adaptive_maneuvers requires at least one segment "
                f"for each of: pitch, roll, yaw, steady."
            )

    pitch_result = calibrate(df, pitch_segs, config)
    roll_result = calibrate(df, roll_segs, config)
    yaw_result = calibrate(df, yaw_segs, config)
    baseline_result = calibrate(df, baseline_segs, config)

    return AdaptiveCalibrationResult(
        pitch=pitch_result,
        roll=roll_result,
        yaw=yaw_result,
        baseline=baseline_result,
        n_terms=pitch_result.n_terms,
    )
```

### Step 4: Run tests to verify they pass

```
pytest tests/unit/test_adaptive.py -v -k "calibrate_adaptive"
```

Expected: PASS (6 tests).

### Step 5: Commit

```bash
git add lmc/adaptive.py tests/unit/test_adaptive.py
git commit -m "feat: implement calibrate_adaptive_maneuvers"
```

---

## Task 4: Implement `_rolling_variance` Helper

**Files:**
- Modify: `lmc/adaptive.py`
- Modify: `tests/unit/test_adaptive.py`

### Step 1: Write failing tests

Append to `tests/unit/test_adaptive.py`:

```python
from lmc.adaptive import _rolling_variance


def test_rolling_variance_all_zeros_returns_zeros() -> None:
    arr = np.zeros(20, dtype=np.float64)
    result = _rolling_variance(arr, window=5)
    np.testing.assert_array_equal(result, np.zeros(20))


def test_rolling_variance_constant_returns_zeros() -> None:
    arr = np.full(20, 3.14, dtype=np.float64)
    result = _rolling_variance(arr, window=5)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_rolling_variance_output_shape_matches_input() -> None:
    arr = np.arange(30, dtype=np.float64)
    result = _rolling_variance(arr, window=10)
    assert result.shape == (30,)


def test_rolling_variance_single_element_window_is_zero() -> None:
    """Variance of a single value is 0."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = _rolling_variance(arr, window=1)
    np.testing.assert_allclose(result, 0.0, atol=1e-12)


def test_rolling_variance_known_values() -> None:
    """After window is full, variance matches numpy for a known window."""
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    result = _rolling_variance(arr, window=3)
    # Index 4: window is [3, 4, 5] → variance = np.var([3,4,5])
    expected_last = float(np.var(np.array([3.0, 4.0, 5.0])))
    np.testing.assert_allclose(result[4], expected_last, atol=1e-12)
    # Index 2: window is [1, 2, 3] → variance = np.var([1,2,3])
    expected_idx2 = float(np.var(np.array([1.0, 2.0, 3.0])))
    np.testing.assert_allclose(result[2], expected_idx2, atol=1e-12)
```

### Step 2: Run tests to verify they fail

```
pytest tests/unit/test_adaptive.py -v -k "rolling_variance"
```

Expected: FAIL — `_rolling_variance` not yet defined.

### Step 3: Implement `_rolling_variance` in `lmc/adaptive.py`

Add before `calibrate_adaptive_maneuvers`:

```python
def _rolling_variance(arr: npt.NDArray[np.float64], window: int) -> npt.NDArray[np.float64]:
    """Compute causal rolling population variance.

    For index ``i``, variance is computed over ``arr[max(0, i-window+1) : i+1]``.
    This means the first ``window-1`` samples use a shorter effective window
    (causal edge handling — no look-ahead, no padding).

    Parameters
    ----------
    arr:
        1-D input array.
    window:
        Maximum number of samples in the rolling window.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``arr`` with rolling variances.
    """
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = float(np.var(arr[max(0, i - window + 1) : i + 1]))
    return out
```

### Step 4: Run tests to verify they pass

```
pytest tests/unit/test_adaptive.py -v -k "rolling_variance"
```

Expected: PASS (5 tests).

### Step 5: Commit

```bash
git add lmc/adaptive.py tests/unit/test_adaptive.py
git commit -m "feat: add _rolling_variance helper for maneuver intensity detection"
```

---

## Task 5: Implement `compensate_adaptive`

**Files:**
- Modify: `lmc/adaptive.py`
- Modify: `tests/unit/test_adaptive.py`

### Step 1: Write failing tests

Append to `tests/unit/test_adaptive.py`:

```python
from lmc.adaptive import compensate_adaptive
from lmc.calibration import calibrate
from lmc.compensation import compensate
from lmc.columns import COL_TMI_COMPENSATED


def _make_full_adaptive_result(
    c_true: npt.NDArray[np.float64],
    config: PipelineConfig,
    seed: int = 42,
) -> tuple[pl.DataFrame, AdaptiveCalibrationResult]:
    """Build a survey DataFrame + AdaptiveCalibrationResult from synthetic data."""
    df_cal, segments = _make_adaptive_calibration_data(n_rows_each=60, seed=seed)
    result = calibrate_adaptive_maneuvers(df_cal, segments, config)
    # Build a survey df (no delta_B required — compensation only needs B columns)
    rng = np.random.default_rng(seed + 1)
    survey = _make_base_df(50, rng)
    return survey, result


def test_compensate_adaptive_returns_tmi_compensated_column() -> None:
    config = PipelineConfig(model_terms="a")
    df, result = _make_full_adaptive_result(np.array([1.0, -2.0, 0.5]), config)
    out = compensate_adaptive(df, result, config)
    assert COL_TMI_COMPENSATED in out.columns


def test_compensate_adaptive_row_count_preserved() -> None:
    config = PipelineConfig(model_terms="a")
    df, result = _make_full_adaptive_result(np.array([1.0, -2.0, 0.5]), config)
    out = compensate_adaptive(df, result, config)
    assert len(out) == len(df)


def test_compensate_adaptive_tmi_values_finite() -> None:
    config = PipelineConfig(model_terms="a")
    df, result = _make_full_adaptive_result(np.array([1.0, -2.0, 0.5]), config)
    out = compensate_adaptive(df, result, config)
    assert out[COL_TMI_COMPENSATED].is_finite().all()


def test_compensate_adaptive_terms_mismatch_raises() -> None:
    """Using model_terms='b' (9 terms) with an 'a' result (3 terms) must fail."""
    config_a = PipelineConfig(model_terms="a")
    config_b = PipelineConfig(model_terms="b")
    df, result_a = _make_full_adaptive_result(np.array([1.0, -2.0, 0.5]), config_a)
    with pytest.raises(ValueError, match="9"):
        compensate_adaptive(df, result_a, config_b)


def test_compensate_adaptive_identical_coefs_matches_standard() -> None:
    """When all four coefficient sets are identical, adaptive == standard."""
    config = PipelineConfig(model_terms="a")
    rng = np.random.default_rng(99)
    c_true = np.array([1.0, -2.0, 0.5])

    # Build a df with delta_B so we can call calibrate()
    block = _make_base_df(80, rng)
    A = build_feature_matrix(block, config).to_numpy()
    delta_b = A @ c_true
    df_cal = block.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))

    from lmc.segmentation import Segment
    single_seg = [Segment(maneuver="steady", heading="N",  # type: ignore[arg-type]
                          start_idx=0, end_idx=80)]
    single_result = calibrate(df_cal, single_seg, config)

    # Wrap the same result in all four slots
    adaptive_result = AdaptiveCalibrationResult(
        pitch=single_result,
        roll=single_result,
        yaw=single_result,
        baseline=single_result,
        n_terms=single_result.n_terms,
    )

    survey = _make_base_df(50, np.random.default_rng(7))
    out_standard = compensate(survey, single_result, config)
    out_adaptive = compensate_adaptive(survey, adaptive_result, config)

    np.testing.assert_allclose(
        out_adaptive[COL_TMI_COMPENSATED].to_numpy(),
        out_standard[COL_TMI_COMPENSATED].to_numpy(),
        atol=1e-10,
    )
```

### Step 2: Run tests to verify they fail

```
pytest tests/unit/test_adaptive.py -v -k "compensate_adaptive"
```

Expected: FAIL — `compensate_adaptive` not yet defined.

### Step 3: Implement `compensate_adaptive` in `lmc/adaptive.py`

Add after `calibrate_adaptive_maneuvers`:

```python
def compensate_adaptive(
    df: pl.DataFrame,
    result: AdaptiveCalibrationResult,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Apply compensation with maneuver-adaptive coefficient blending.

    Algorithm
    ---------
    1. Build feature matrix ``A`` from fluxgate columns (shape ``n × n_terms``).
    2. Compute direction cosines from raw fluxgate components.
    3. Compute rolling variance of each direction cosine to detect maneuver
       intensity (pitch ~ cos_x, roll ~ cos_y, yaw ~ cos_z).
    4. Normalise intensities + ``maneuver_baseline_weight`` to unit-sum weights.
    5. Blend four interference vectors: ``w_pitch*(A@c_pitch) + ...``
    6. Return ``df`` with column ``COL_TMI_COMPENSATED = B_total - interference``.

    Parameters
    ----------
    df:
        Survey DataFrame containing all required magnetometer columns.
    result:
        Adaptive calibration result from ``calibrate_adaptive_maneuvers()``.
    config:
        Pipeline configuration.  ``model_terms`` must match those used during
        calibration.  ``maneuver_detection_window`` and
        ``maneuver_baseline_weight`` control blending behaviour.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with added column ``COL_TMI_COMPENSATED``.

    Raises
    ------
    ValueError
        If feature matrix column count does not match ``result.n_terms``.
    """
    feature_matrix = build_feature_matrix(df, config)
    A = feature_matrix.to_numpy()

    if A.shape[1] != result.n_terms:
        raise ValueError(
            f"Feature matrix has {A.shape[1]} columns but AdaptiveCalibrationResult "
            f"has {result.n_terms} terms. Ensure the same model_terms are used "
            "for both calibration and compensation."
        )

    # --- Direction cosines for maneuver detection ---
    bx = np.asarray(df[COL_BX].to_numpy(), dtype=np.float64)
    by = np.asarray(df[COL_BY].to_numpy(), dtype=np.float64)
    bz = np.asarray(df[COL_BZ].to_numpy(), dtype=np.float64)
    b_flux_mag = np.sqrt(bx**2 + by**2 + bz**2)
    cos_x = bx / b_flux_mag
    cos_y = by / b_flux_mag
    cos_z = bz / b_flux_mag

    # --- Rolling variance → maneuver intensities ---
    window = config.maneuver_detection_window
    pitch_intensity = _rolling_variance(cos_x, window)
    roll_intensity = _rolling_variance(cos_y, window)
    yaw_intensity = _rolling_variance(cos_z, window)

    # --- Normalise to blend weights (sum == 1 at every sample) ---
    baseline_w = config.maneuver_baseline_weight
    total = pitch_intensity + roll_intensity + yaw_intensity + baseline_w
    w_pitch = pitch_intensity / total
    w_roll = roll_intensity / total
    w_yaw = yaw_intensity / total
    w_baseline = baseline_w / total

    # --- Per-maneuver interference vectors (broadcast-efficient) ---
    interf_pitch = A @ result.pitch.coefficients
    interf_roll = A @ result.roll.coefficients
    interf_yaw = A @ result.yaw.coefficients
    interf_baseline = A @ result.baseline.coefficients

    interference = (
        w_pitch * interf_pitch
        + w_roll * interf_roll
        + w_yaw * interf_yaw
        + w_baseline * interf_baseline
    )

    b_total = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64)
    tmi_comp = b_total - interference

    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
```

### Step 4: Run tests to verify they pass

```
pytest tests/unit/test_adaptive.py -v -k "compensate_adaptive"
```

Expected: PASS (5 tests).

### Step 5: Run full test suite

```
make test
```

Expected: all pass.

### Step 6: Commit

```bash
git add lmc/adaptive.py tests/unit/test_adaptive.py
git commit -m "feat: implement compensate_adaptive with rolling variance blending"
```

---

## Task 6: Export New Symbols from `lmc/__init__.py`

**Files:**
- Modify: `lmc/__init__.py`
- Modify: `tests/unit/test_adaptive.py`

### Step 1: Write failing test

Append to `tests/unit/test_adaptive.py`:

```python
def test_public_api_exports_adaptive_symbols() -> None:
    """All three new symbols must be importable directly from lmc."""
    import lmc
    assert hasattr(lmc, "AdaptiveCalibrationResult")
    assert hasattr(lmc, "calibrate_adaptive_maneuvers")
    assert hasattr(lmc, "compensate_adaptive")
```

### Step 2: Run test to verify it fails

```
pytest tests/unit/test_adaptive.py::test_public_api_exports_adaptive_symbols -v
```

Expected: FAIL.

### Step 3: Add exports to `lmc/__init__.py`

Open `lmc/__init__.py`. Add an import near the existing `calibrate`/`compensate` imports:

```python
from lmc.adaptive import (
    AdaptiveCalibrationResult,
    calibrate_adaptive_maneuvers,
    compensate_adaptive,
)
```

Also add the three names to the `__all__` list if one exists in that file.

### Step 4: Run test to verify it passes

```
pytest tests/unit/test_adaptive.py::test_public_api_exports_adaptive_symbols -v
```

Expected: PASS.

### Step 5: Run full lint + test suite

```
make lint && make test
```

Expected: all pass.

### Step 6: Commit

```bash
git add lmc/__init__.py tests/unit/test_adaptive.py
git commit -m "feat: export adaptive compensation API from lmc"
```

---

## Task 7: Integration Test

**Files:**
- Create: `tests/integration/test_adaptive_integration.py`

This test verifies the end-to-end pipeline on synthetic FOM data, mirroring `test_integration_simple.py`.

### Step 1: Write the failing test

Create `tests/integration/test_adaptive_integration.py`:

```python
"""Integration test for adaptive maneuver-based compensation."""

from __future__ import annotations

import numpy as np
import polars as pl

from lmc import (
    COL_BTOTAL,
    COL_TMI_COMPENSATED,
    AdaptiveCalibrationResult,
    PipelineConfig,
    calibrate_adaptive_maneuvers,
    compensate_adaptive,
    compute_interference,
    segment_fom,
    validate_dataframe,
)
from tests.integration.synthetic import make_fom_dataframe


def test_adaptive_compensation_produces_valid_output() -> None:
    """Adaptive pipeline produces finite compensated TMI on synthetic FOM data."""
    c_true = np.arange(1, 19, dtype=float) * 0.5  # 18 terms → model_terms="c"

    df_fom = make_fom_dataframe(c_true, noise_std=0.05)

    config = PipelineConfig(
        model_terms="c",
        earth_field_method="steady_mean",
        segment_label_col="segment",
    )

    df_fom = validate_dataframe(df_fom)
    segments = segment_fom(df_fom, config)

    # Build steady_mask for earth-field baseline
    mask_arr = np.zeros(df_fom.height, dtype=bool)
    for seg in segments:
        if seg.maneuver == "steady":
            mask_arr[seg.start_idx : seg.end_idx] = True
    steady_mask = pl.Series(mask_arr, dtype=pl.Boolean)

    delta_b = compute_interference(df_fom, config, steady_mask=steady_mask)
    df_fom = df_fom.with_columns(delta_b)

    # Calibrate
    adaptive_result = calibrate_adaptive_maneuvers(df_fom, segments, config)
    assert isinstance(adaptive_result, AdaptiveCalibrationResult)
    assert adaptive_result.n_terms == 18

    # Compensate on a single clean segment to avoid cross-segment derivative artefacts
    pitch_n_seg = next(
        seg for seg in segments if seg.maneuver == "pitch" and seg.heading == "N"
    )
    pitch_n = df_fom.slice(
        pitch_n_seg.start_idx, pitch_n_seg.end_idx - pitch_n_seg.start_idx
    )

    out = compensate_adaptive(pitch_n, adaptive_result, config)

    # Output validity checks
    assert COL_TMI_COMPENSATED in out.columns
    assert len(out) == len(pitch_n)
    assert out[COL_TMI_COMPENSATED].is_finite().all()

    # Compensation should reduce variance (signal is cleaner after removal)
    std_raw = float(out[COL_BTOTAL].std())
    std_comp = float(out[COL_TMI_COMPENSATED].std())
    assert std_raw > std_comp, (
        f"Expected compensation to reduce std: raw={std_raw:.4f}, comp={std_comp:.4f}"
    )
```

### Step 2: Run test to verify it fails

```
pytest tests/integration/test_adaptive_integration.py -v
```

Expected: FAIL — `calibrate_adaptive_maneuvers` / `compensate_adaptive` not exported yet (or already done in Task 6).

### Step 3: No implementation needed — just run

All implementation was done in Tasks 2–6. This task only adds the test.

### Step 4: Run test to verify it passes

```
pytest tests/integration/test_adaptive_integration.py -v
```

Expected: PASS.

### Step 5: Run full test suite + lint

```
make lint && make test
```

Expected: all pass.

### Step 6: Commit

```bash
git add tests/integration/test_adaptive_integration.py
git commit -m "test: add integration test for adaptive maneuver compensation"
```

---

## Final Checklist

Before opening a PR, verify:

- [ ] `make lint` passes (ruff, typos, yamllint, pyright)
- [ ] `make test` passes (all unit + integration tests)
- [ ] `lmc/__init__.py` exports `AdaptiveCalibrationResult`, `calibrate_adaptive_maneuvers`, `compensate_adaptive`
- [ ] No manual version bump in `pyproject.toml` or `lmc/__init__.py`
- [ ] PR title references issue #50
- [ ] PR body follows `.github/PULL_REQUEST_TEMPLATE.md`

---

## Pitfalls to Avoid

| Pitfall | Why | Fix |
|---------|-----|-----|
| `B_total` as fluxgate magnitude denominator | Incorrect — fixed in commit 77a619c | Use `sqrt(B_x²+B_y²+B_z²)` |
| Importing `ManeuverType` directly | It lives in `lmc.segmentation`, not `lmc` | `from lmc.segmentation import Segment` |
| Zero division in blend weights | Total can't be zero because `baseline_weight > 0` | Field validator `gt=0.0` already prevents this |
| Cross-segment derivative artefacts in integration test | `np.gradient` at block boundaries sees a 90° heading jump | Test on a single clean segment (e.g. `pitch_N`) not the full FOM DataFrame |
| Adding `pip install` commands | Project uses `uv` exclusively | Use `uv add` or `make install` |
