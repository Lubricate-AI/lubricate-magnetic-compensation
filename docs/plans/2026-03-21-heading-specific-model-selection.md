# Heading-Specific Model Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fit separate Tolles-Lawson models per heading bin and select the appropriate model at compensation time, reducing heading-dependent multicollinearity.

**Architecture:** A new `lmc/vif.py` module computes per-column Variance Inflation Factors. A new `lmc/heading_calibration.py` module owns `HeadingCalibrationResult` and `calibrate_per_heading()`, calling the existing `calibrate()` helper per heading group. A new `compensate_heading_specific()` is added to `lmc/compensation.py` to route each survey row to the nearest calibrated heading model. One new boolean field is added to `PipelineConfig`.

**Tech Stack:** Python 3.12+, NumPy (VIF via lstsq, vectorised heading routing), Polars (DataFrame I/O), Pydantic (config), scikit-learn (via existing `calibrate()`), pytest (TDD).

---

## Background: How the Existing Pipeline Works

Read these files top-to-bottom before writing any code:

| File | Purpose |
|------|---------|
| `lmc/columns.py` | All column-name constants |
| `lmc/config.py` | `PipelineConfig` Pydantic model — add new field here |
| `lmc/calibration.py` | `CalibrationResult` dataclass + `calibrate()` — **call this, don't replicate it** |
| `lmc/compensation.py` | `compensate()` — replicate its structure for the new heading-specific variant |
| `lmc/features.py` | `build_feature_matrix(df, config)` |
| `lmc/segmentation.py` | `Segment`, `HeadingType`, `_resolve_bin_centres`, `_assign_heading_bin` — **import the private helpers directly within the package** |
| `lmc/__init__.py` | Public API — add exports here last |
| `tests/unit/test_calibration.py` | `_make_synthetic_df` + `_make_synthetic_data` helpers — copy for tests |
| `tests/integration/synthetic.py` | `make_fom_dataframe()` — use in integration test |

Key invariants:
- `compensate_heading_specific` must output `COL_TMI_COMPENSATED = "tmi_compensated"` (same column name as `compensate`).
- The `_resolve_bin_centres` and `_assign_heading_bin` helpers in `segmentation.py` are private but importable within the package — use them directly.
- Nearest-heading selection (no tolerance check) is used during survey compensation: every row is routed to the closest calibrated heading, even if the aircraft wasn't flying that heading during calibration. This is intentional — survey legs may not exactly match FOM headings.

---

## Task 1: Add `use_heading_specific_calibration` to `PipelineConfig`

**Files:**
- Modify: `lmc/config.py`
- Modify: `tests/unit/test_config.py`

### Step 1: Write the failing test

Add to `tests/unit/test_config.py`:

```python
def test_use_heading_specific_calibration_defaults_false() -> None:
    cfg = PipelineConfig()
    assert cfg.use_heading_specific_calibration is False


def test_use_heading_specific_calibration_can_be_enabled() -> None:
    cfg = PipelineConfig(use_heading_specific_calibration=True)
    assert cfg.use_heading_specific_calibration is True
```

### Step 2: Run test to verify it fails

```bash
uv run pytest tests/unit/test_config.py::test_use_heading_specific_calibration_defaults_false -v
```

Expected: `FAILED` with `AttributeError` or similar.

### Step 3: Add the field to `PipelineConfig`

Add after the `maneuver_baseline_weight` field in `lmc/config.py`:

```python
use_heading_specific_calibration: bool = Field(
    default=False,
    description=(
        "Fit separate Tolles-Lawson models per heading bin to reduce "
        "heading-dependent multicollinearity. When True, use "
        "calibrate_per_heading() and compensate_heading_specific() "
        "instead of calibrate() and compensate()."
    ),
)
```

### Step 4: Run test to verify it passes

```bash
uv run pytest tests/unit/test_config.py::test_use_heading_specific_calibration_defaults_false tests/unit/test_config.py::test_use_heading_specific_calibration_can_be_enabled -v
```

Expected: both `PASSED`.

### Step 5: Run full test suite to confirm no regressions

```bash
uv run pytest tests/unit/test_config.py -v
```

Expected: all tests pass.

### Step 6: Commit

```bash
git add lmc/config.py tests/unit/test_config.py
git commit -m "feat: add use_heading_specific_calibration config field"
```

---

## Task 2: Implement VIF Computation in `lmc/vif.py`

VIF for column *i* = 1 / (1 − R²_i), where R²_i is computed by regressing column *i* on all other columns. High VIF (> 10) signals multicollinearity.

**Files:**
- Create: `lmc/vif.py`
- Create: `tests/unit/test_vif.py`

### Step 1: Write failing tests

Create `tests/unit/test_vif.py`:

```python
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
    assert np.isinf(vif[0]) or np.isinf(vif[1]), "Duplicate column should produce inf VIF"


def test_vif_returns_float64_array() -> None:
    A = np.eye(3, dtype=np.float32)
    vif = compute_vif(A.astype(np.float64))
    assert vif.dtype == np.float64


def test_vif_raises_on_single_column() -> None:
    """Cannot compute R² when there are no other columns to regress on."""
    A = np.ones((10, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="at least 2 columns"):
        compute_vif(A)
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_vif.py -v
```

Expected: `FAILED` with `ModuleNotFoundError`.

### Step 3: Implement `lmc/vif.py`

```python
"""Variance Inflation Factor (VIF) computation for multicollinearity diagnostics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_vif(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the Variance Inflation Factor for each column of a design matrix.

    VIF_i = 1 / (1 - R²_i), where R²_i is the coefficient of determination
    from regressing column *i* on all other columns.  VIF > 10 is commonly
    used as a threshold for problematic multicollinearity.

    Parameters
    ----------
    A:
        Design matrix of shape ``(n_samples, n_terms)``.  Must have at least
        2 columns.

    Returns
    -------
    npt.NDArray[np.float64]
        VIF values, shape ``(n_terms,)``.  Returns ``inf`` for columns that
        are perfectly predicted by the others (R² == 1).

    Raises
    ------
    ValueError
        If ``A`` has fewer than 2 columns.
    """
    n_terms = A.shape[1]
    if n_terms < 2:
        raise ValueError(
            f"compute_vif requires at least 2 columns; got {n_terms}."
        )

    vif = np.empty(n_terms, dtype=np.float64)

    for i in range(n_terms):
        y = A[:, i]
        X = np.delete(A, i, axis=1)

        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coef

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        if ss_tot == 0.0:
            # Constant column — undefined VIF, treat as inf.
            vif[i] = float("inf")
        else:
            r2 = 1.0 - ss_res / ss_tot
            vif[i] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")

    return vif
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_vif.py -v
```

Expected: all 5 tests `PASSED`.

### Step 5: Commit

```bash
git add lmc/vif.py tests/unit/test_vif.py
git commit -m "feat: add VIF computation for multicollinearity diagnostics"
```

---

## Task 3: Implement `HeadingCalibrationResult` and `calibrate_per_heading()`

**Files:**
- Create: `lmc/heading_calibration.py`
- Create: `tests/unit/test_heading_calibration.py`

### Step 1: Write failing tests

Create `tests/unit/test_heading_calibration.py`:

```python
"""Unit tests for lmc.heading_calibration."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lmc.calibration import CalibrationResult
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
from lmc.heading_calibration import HeadingCalibrationResult, calibrate_per_heading
from lmc.segmentation import Segment


# ---------------------------------------------------------------------------
# Shared synthetic data helper (same pattern as test_calibration.py)
# ---------------------------------------------------------------------------


def _make_df_with_delta_b(
    n_rows: int,
    rng: np.random.Generator,
    config: PipelineConfig,
) -> pl.DataFrame:
    raw = rng.standard_normal((n_rows, 3))
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    cosines = raw / norms
    b_total = 50_000.0
    bx = cosines[:, 0] * b_total
    by = cosines[:, 1] * b_total
    bz = cosines[:, 2] * b_total
    df = pl.DataFrame(
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
    A = build_feature_matrix(df, config).to_numpy()
    c_true = rng.standard_normal(A.shape[1])
    delta_b = (A @ c_true).astype(np.float64)
    return df.with_columns(pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64))


def _make_multi_heading_data(
    config: PipelineConfig,
    n_per_heading: int = 60,
    seed: int = 42,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build a df + segments list with four distinct heading groups."""
    rng = np.random.default_rng(seed)
    headings = ["N", "E", "S", "W"]
    blocks: list[pl.DataFrame] = []
    segments: list[Segment] = []
    offset = 0
    for h in headings:
        block = _make_df_with_delta_b(n_per_heading, rng, config)
        blocks.append(block)
        segments.append(
            Segment(maneuver="steady", heading=h, start_idx=offset, end_idx=offset + n_per_heading)  # type: ignore[arg-type]
        )
        offset += n_per_heading
    df = pl.concat(blocks)
    return df, segments


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_CONFIG_A = PipelineConfig(model_terms="a")


def test_calibrate_per_heading_returns_result_for_each_heading() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    assert set(result.per_heading.keys()) == {"N", "E", "S", "W"}


def test_calibrate_per_heading_each_result_is_calibration_result() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    for cal in result.per_heading.values():
        assert isinstance(cal, CalibrationResult)


def test_calibrate_per_heading_vif_shape_matches_n_terms() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    for heading, vif_arr in result.per_heading_vif.items():
        expected_n_terms = result.per_heading[heading].n_terms
        assert vif_arr.shape == (expected_n_terms,), (
            f"Heading {heading}: VIF shape {vif_arr.shape} != ({expected_n_terms},)"
        )


def test_calibrate_per_heading_vif_all_positive() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    for heading, vif_arr in result.per_heading_vif.items():
        finite = vif_arr[np.isfinite(vif_arr)]
        assert np.all(finite >= 1.0), f"Heading {heading}: VIF must be >= 1"


def test_calibrate_per_heading_raises_on_empty_segments() -> None:
    df, _ = _make_multi_heading_data(_CONFIG_A)
    with pytest.raises(ValueError, match="segments must be non-empty"):
        calibrate_per_heading(df, [], _CONFIG_A)


def test_calibrate_per_heading_single_heading() -> None:
    """Single-heading data should produce a result with one key."""
    rng = np.random.default_rng(99)
    config = PipelineConfig(model_terms="a")
    df = _make_df_with_delta_b(60, rng, config)
    segments = [Segment(maneuver="steady", heading="N", start_idx=0, end_idx=60)]  # type: ignore[arg-type]
    result = calibrate_per_heading(df, segments, config)
    assert set(result.per_heading.keys()) == {"N"}
    assert set(result.per_heading_vif.keys()) == {"N"}
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_heading_calibration.py -v
```

Expected: `FAILED` with `ModuleNotFoundError`.

### Step 3: Implement `lmc/heading_calibration.py`

```python
"""Heading-specific Tolles-Lawson calibration for multicollinearity reduction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.calibration import CalibrationResult, calibrate
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import HeadingType, Segment
from lmc.vif import compute_vif


@dataclass(frozen=True)
class HeadingCalibrationResult:
    """Result of fitting separate Tolles-Lawson models per heading bin.

    Attributes
    ----------
    per_heading:
        Calibration result keyed by heading label (``"N"``, ``"E"``, etc.).
        Only headings present in the supplied segments are populated.
    per_heading_vif:
        Variance Inflation Factors for the design matrix of each heading,
        shape ``(n_terms,)`` per heading.  High VIF (> 10) signals
        multicollinearity for that heading.
    """

    per_heading: dict[HeadingType, CalibrationResult]
    per_heading_vif: dict[HeadingType, npt.NDArray[np.float64]]


def calibrate_per_heading(
    df: pl.DataFrame,
    segments: list[Segment],
    config: PipelineConfig,
) -> HeadingCalibrationResult:
    """Fit a separate Tolles-Lawson model for each heading bin.

    Groups the supplied segments by their ``heading`` label and calls
    the existing ``calibrate()`` function for each group independently.
    Also computes Variance Inflation Factors for the design matrix of
    each heading.

    Parameters
    ----------
    df:
        Full calibration DataFrame containing all required columns including
        ``COL_DELTA_B``.
    segments:
        Non-empty list of labeled flight segments.  Segments with the same
        ``heading`` are pooled together.
    config:
        Pipeline configuration.

    Returns
    -------
    HeadingCalibrationResult
        Per-heading calibration results and VIF diagnostics.

    Raises
    ------
    ValueError
        If ``segments`` is empty or if ``calibrate()`` raises for any heading.
    """
    if not segments:
        raise ValueError("segments must be non-empty; cannot calibrate with no data.")

    # Group segments by heading.
    heading_segments: dict[HeadingType, list[Segment]] = defaultdict(list)
    for seg in segments:
        heading_segments[seg.heading].append(seg)

    per_heading: dict[HeadingType, CalibrationResult] = {}
    per_heading_vif: dict[HeadingType, npt.NDArray[np.float64]] = {}

    for heading, segs in heading_segments.items():
        # Fit per-heading model using existing calibrate().
        per_heading[heading] = calibrate(df, segs, config)

        # Build stacked A matrix for VIF computation.
        a_blocks: list[npt.NDArray[np.float64]] = []
        for seg in segs:
            segment_df = df.slice(seg.start_idx, seg.end_idx - seg.start_idx)
            a_seg = build_feature_matrix(segment_df, config).to_numpy()
            a_blocks.append(a_seg)
        A: npt.NDArray[np.float64] = np.vstack(a_blocks)

        per_heading_vif[heading] = compute_vif(A)

    return HeadingCalibrationResult(
        per_heading=per_heading,
        per_heading_vif=per_heading_vif,
    )
```

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_heading_calibration.py -v
```

Expected: all 6 tests `PASSED`.

### Step 5: Run linting

```bash
uv run ruff check lmc/heading_calibration.py lmc/vif.py
```

Expected: no errors.

### Step 6: Commit

```bash
git add lmc/vif.py lmc/heading_calibration.py tests/unit/test_heading_calibration.py
git commit -m "feat: add per-heading calibration with VIF diagnostics"
```

---

## Task 4: Implement `compensate_heading_specific()`

**Files:**
- Modify: `lmc/compensation.py`
- Modify: `tests/unit/test_compensation.py`

### Step 1: Write failing tests

Add to `tests/unit/test_compensation.py`:

```python
from lmc.columns import COL_HEADING
from lmc.heading_calibration import HeadingCalibrationResult, calibrate_per_heading
from lmc.compensation import compensate_heading_specific
from lmc.segmentation import Segment


def _make_heading_specific_data(
    config: PipelineConfig,
    n_per_heading: int = 60,
    seed: int = 7,
) -> tuple[pl.DataFrame, list[Segment]]:
    """Build (df, segments) with four heading groups, each with a HEADING column."""
    rng = np.random.default_rng(seed)
    headings_map = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    blocks: list[pl.DataFrame] = []
    segments: list[Segment] = []
    offset = 0
    for h_label, h_deg in headings_map.items():
        raw = rng.standard_normal((n_per_heading, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        cosines = raw / norms
        b_total = 50_000.0
        df_block = pl.DataFrame(
            {
                COL_TIME: np.arange(offset, offset + n_per_heading, dtype=np.float64),
                COL_LAT: np.full(n_per_heading, 45.0),
                COL_LON: np.full(n_per_heading, -75.0),
                COL_ALT: np.full(n_per_heading, 300.0),
                COL_BTOTAL: np.full(n_per_heading, b_total),
                COL_BX: cosines[:, 0] * b_total,
                COL_BY: cosines[:, 1] * b_total,
                COL_BZ: cosines[:, 2] * b_total,
                COL_HEADING: np.full(n_per_heading, h_deg),
            }
        )
        A = build_feature_matrix(df_block, config).to_numpy()
        c_true = rng.standard_normal(A.shape[1])
        delta_b = (A @ c_true).astype(np.float64)
        df_block = df_block.with_columns(
            pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64)
        )
        blocks.append(df_block)
        segments.append(
            Segment(
                maneuver="steady",
                heading=h_label,  # type: ignore[arg-type]
                start_idx=offset,
                end_idx=offset + n_per_heading,
            )
        )
        offset += n_per_heading
    df = pl.concat(blocks)
    return df, segments


def test_compensate_heading_specific_adds_tmi_compensated_column() -> None:
    config = PipelineConfig(model_terms="a")
    df, segments = _make_heading_specific_data(config)
    cal_result = calibrate_per_heading(df, segments, config)
    result_df = compensate_heading_specific(df, cal_result, config)
    assert COL_TMI_COMPENSATED in result_df.columns


def test_compensate_heading_specific_output_length_matches_input() -> None:
    config = PipelineConfig(model_terms="a")
    df, segments = _make_heading_specific_data(config)
    cal_result = calibrate_per_heading(df, segments, config)
    result_df = compensate_heading_specific(df, cal_result, config)
    assert len(result_df) == len(df)


def test_compensate_heading_specific_raises_on_n_terms_mismatch() -> None:
    config_a = PipelineConfig(model_terms="a")
    config_b = PipelineConfig(model_terms="b")
    df, segments = _make_heading_specific_data(config_a)
    # Calibrate with model_a but try to compensate with model_b config.
    cal_result = calibrate_per_heading(df, segments, config_a)
    with pytest.raises(ValueError, match="n_terms"):
        compensate_heading_specific(df, cal_result, config_b)
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/unit/test_compensation.py::test_compensate_heading_specific_adds_tmi_compensated_column -v
```

Expected: `FAILED` with `ImportError`.

### Step 3: Implement `compensate_heading_specific()` in `lmc/compensation.py`

Add at the bottom of `lmc/compensation.py` (after the existing imports, add new imports and the function):

```python
from lmc.columns import COL_HEADING
from lmc.heading_calibration import HeadingCalibrationResult
from lmc.segmentation import _assign_heading_bin, _resolve_bin_centres


def compensate_heading_specific(
    df: pl.DataFrame,
    result: HeadingCalibrationResult,
    config: PipelineConfig,
) -> pl.DataFrame:
    """Subtract heading-specific modelled interference from survey TMI.

    For each row of ``df``, selects the coefficients from the nearest
    calibrated heading bin and subtracts the modelled interference from
    ``COL_BTOTAL``.

    Parameters
    ----------
    df:
        Survey DataFrame.  Must contain ``COL_HEADING`` and all columns
        required by ``build_feature_matrix``.
    result:
        Result from ``calibrate_per_heading()``.
    config:
        Pipeline configuration.  Must use the same ``model_terms`` as
        calibration.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with one additional column ``COL_TMI_COMPENSATED``.

    Raises
    ------
    ValueError
        If the feature matrix column count does not match ``result.n_terms``,
        or if ``COL_HEADING`` is absent.
    """
    if COL_HEADING not in df.columns:
        raise ValueError(
            f"Column '{COL_HEADING}' is required for heading-specific compensation "
            f"but was not found. Available columns: {df.columns}"
        )

    feature_matrix = build_feature_matrix(df, config)
    A = feature_matrix.to_numpy()

    n_terms = next(iter(result.per_heading.values())).n_terms
    if A.shape[1] != n_terms:
        raise ValueError(
            f"Feature matrix has {A.shape[1]} columns but HeadingCalibrationResult "
            f"has {n_terms} terms. Ensure the same model_terms are used for both "
            "calibration and compensation."
        )

    headings = np.asarray(df[COL_HEADING].to_numpy(), dtype=np.float64)
    centres = _resolve_bin_centres(config, headings)

    # Route each row to the nearest calibrated heading (no tolerance cut-off —
    # every survey row gets a model even if heading differs from FOM legs).
    interference = np.empty(len(df), dtype=np.float64)
    for h_label, cal in result.per_heading.items():
        mask = np.array(
            [_assign_heading_bin(h, centres, 180.0) == h_label for h in headings],
            dtype=bool,
        )
        if mask.any():
            interference[mask] = A[mask] @ cal.coefficients

    tmi_comp = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64) - interference

    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
```

**Important:** Using `tolerance=180.0` in `_assign_heading_bin` means every row is assigned to exactly one heading bin (the nearest one), since 180° covers the full half-circle. This is intentional for survey compensation.

### Step 4: Run tests to verify they pass

```bash
uv run pytest tests/unit/test_compensation.py -v
```

Expected: all tests including the three new ones `PASSED`.

### Step 5: Run linting

```bash
uv run ruff check lmc/compensation.py
```

Expected: no errors.

### Step 6: Commit

```bash
git add lmc/compensation.py tests/unit/test_compensation.py
git commit -m "feat: add heading-specific compensation routing"
```

---

## Task 5: Export New Public API from `lmc/__init__.py`

**Files:**
- Modify: `lmc/__init__.py`

### Step 1: No test needed — verify via import

There is no dedicated test to write here. Instead, after editing, run:

```bash
uv run python -c "from lmc import HeadingCalibrationResult, calibrate_per_heading, compensate_heading_specific; print('OK')"
```

Expected: prints `OK`.

### Step 2: Update `lmc/__init__.py`

Add imports at the top with the other calibration imports:

```python
from lmc.heading_calibration import HeadingCalibrationResult, calibrate_per_heading
from lmc.compensation import compensate_heading_specific
```

Add to `__all__`:

```python
"HeadingCalibrationResult",
"calibrate_per_heading",
"compensate_heading_specific",
```

Place them near `CalibrationResult`, `calibrate`, and `compensate` in both the import block and `__all__`.

### Step 3: Verify import

```bash
uv run python -c "from lmc import HeadingCalibrationResult, calibrate_per_heading, compensate_heading_specific; print('OK')"
```

Expected: `OK`.

### Step 4: Run full test suite

```bash
uv run pytest -v
```

Expected: all tests pass.

### Step 5: Commit

```bash
git add lmc/__init__.py
git commit -m "feat: export heading-specific calibration API from lmc"
```

---

## Task 6: Integration Test — Variance Improvement with Heading-Specific Models

This test uses `make_fom_dataframe()` from `tests/integration/synthetic.py`. It verifies that heading-specific calibration produces a valid `FomReport` and that the per-heading improvement ratios are reasonable.

**Files:**
- Create: `tests/integration/test_heading_specific_integration.py`

### Step 1: Write failing test

Create `tests/integration/test_heading_specific_integration.py`:

```python
"""Integration test: heading-specific calibration and compensation pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from lmc import (
    HeadingCalibrationResult,
    PipelineConfig,
    calibrate_per_heading,
    compensate_heading_specific,
    compute_fom_report,
    segment_fom,
)
from lmc.columns import COL_DELTA_B, COL_TMI_COMPENSATED
from lmc.earth_field import compute_interference
from tests.integration.synthetic import make_fom_dataframe


def _make_pipeline_data(model_terms: str = "b", seed: int = 42):
    """Return (df_with_delta_b, segments, config) for a 4-heading FOM flight."""
    rng = np.random.default_rng(seed)
    n_terms_map = {"a": 3, "b": 9, "c": 18}
    n_terms = n_terms_map[model_terms]
    c_true = rng.standard_normal(n_terms)

    config = PipelineConfig(
        model_terms=model_terms,
        segment_label_col="segment_label",
        use_heading_specific_calibration=True,
    )

    df = make_fom_dataframe(c_true, n_rows_per_block=60, noise_std=0.5, seed=seed)
    df = compute_interference(df, config)

    segments = segment_fom(df, config)
    return df, segments, config, c_true


def test_calibrate_per_heading_returns_four_heading_keys() -> None:
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    assert isinstance(result, HeadingCalibrationResult)
    assert set(result.per_heading.keys()) == {"N", "E", "S", "W"}


def test_per_heading_vif_available_for_all_headings() -> None:
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    for heading in ("N", "E", "S", "W"):
        assert heading in result.per_heading_vif
        vif = result.per_heading_vif[heading]
        assert vif.shape[0] == result.per_heading[heading].n_terms


def test_compensate_heading_specific_produces_tmi_compensated() -> None:
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    out_df = compensate_heading_specific(df, result, config)
    assert COL_TMI_COMPENSATED in out_df.columns
    assert len(out_df) == len(df)


def test_heading_specific_improvement_ratio_exceeds_one() -> None:
    """After heading-specific compensation, variance should be reduced."""
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)

    # Use the N-heading model result to compute a FOM report.
    from lmc.segmentation import HeadingType
    north_segs = [s for s in segments if s.heading == "N"]
    north_result = result.per_heading["N"]

    report = compute_fom_report(df, north_segs, north_result)
    assert report.improvement_ratio > 1.0, (
        f"Expected improvement_ratio > 1.0, got {report.improvement_ratio}"
    )


def test_condition_number_available_per_heading() -> None:
    """Each per-heading CalibrationResult should carry a condition number."""
    df, segments, config, _ = _make_pipeline_data()
    result = calibrate_per_heading(df, segments, config)
    for heading, cal in result.per_heading.items():
        assert cal.condition_number > 0.0, f"Heading {heading}: bad condition number"
        assert np.isfinite(cal.condition_number), f"Heading {heading}: inf condition number"
```

### Step 2: Run tests to verify they fail

```bash
uv run pytest tests/integration/test_heading_specific_integration.py -v
```

Expected: `FAILED` — module imports may work but `compute_interference` or `make_fom_dataframe` may produce unexpected results until the pipeline is wired together correctly.

### Step 3: Run tests to verify they pass

After Tasks 1–5 are complete, these tests should pass without additional implementation work:

```bash
uv run pytest tests/integration/test_heading_specific_integration.py -v
```

Expected: all 5 tests `PASSED`.

### Step 4: Run full test suite and linting

```bash
uv run pytest -v && uv run make lint
```

Expected: all tests pass, no lint errors.

### Step 5: Commit

```bash
git add tests/integration/test_heading_specific_integration.py
git commit -m "test: add integration tests for heading-specific model selection"
```

---

## Final Verification

Run the full suite once more:

```bash
uv run pytest -v
uv run make lint
```

Expected: all tests green, no lint warnings.

---

## Summary of New Files and Changes

| Action | Path |
|--------|------|
| Create | `lmc/vif.py` |
| Create | `lmc/heading_calibration.py` |
| Modify | `lmc/compensation.py` (add `compensate_heading_specific`) |
| Modify | `lmc/config.py` (add `use_heading_specific_calibration`) |
| Modify | `lmc/__init__.py` (export new symbols) |
| Create | `tests/unit/test_vif.py` |
| Create | `tests/unit/test_heading_calibration.py` |
| Modify | `tests/unit/test_compensation.py` (add 3 tests) |
| Modify | `tests/unit/test_config.py` (add 2 tests) |
| Create | `tests/integration/test_heading_specific_integration.py` |
