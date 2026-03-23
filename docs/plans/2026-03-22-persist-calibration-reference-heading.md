# Persist Calibration Reference Heading in HeadingCalibrationResult

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Store the resolved `reference_heading_deg` inside `HeadingCalibrationResult` during `calibrate_per_heading()` so that `compensate_heading_specific()` re-uses the calibration bin centres instead of re-estimating from survey headings.

**Architecture:** Three files need to change. First, `segmentation.py` gets a public `resolve_reference_heading()` helper and a private `_bin_centres_from_ref()` helper (both extracted from the existing `resolve_bin_centres()`). Second, `heading_calibration.py` gains a `reference_heading_deg: float` field on `HeadingCalibrationResult` and populates it during `calibrate_per_heading()`. Third, `compensation.py` reads that stored value instead of re-calling `resolve_bin_centres()` with survey headings.

**Tech Stack:** Python 3.12, polars, numpy, pytest, ruff (linting)

---

### Task 1: Refactor `resolve_bin_centres` — extract helpers in `segmentation.py`

**Files:**
- Modify: `lmc/segmentation.py:251-281`

**Step 1: Write the failing tests**

Add these tests to `tests/unit/test_segmentation.py` (this file likely doesn't exist yet — check first with `ls tests/unit/`; if it does not exist, create it):

```python
"""Unit tests for lmc.segmentation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from lmc.config import PipelineConfig
from lmc.segmentation import resolve_bin_centres, resolve_reference_heading


def test_resolve_reference_heading_uses_config_value_when_set() -> None:
    config = PipelineConfig(reference_heading_deg=10.0)
    headings = np.array([5.0, 95.0, 185.0, 275.0])  # cardinal-ish
    ref = resolve_reference_heading(config, headings)
    assert ref == pytest.approx(10.0)


def test_resolve_reference_heading_estimates_when_config_is_none() -> None:
    config = PipelineConfig(reference_heading_deg=None)
    # Pure cardinal headings → folded circular mean should be near 0°.
    headings = np.array([0.0, 90.0, 180.0, 270.0])
    ref = resolve_reference_heading(config, headings)
    # The folded circular mean of {0,0,0,0} mod 90 is 0°.
    assert ref == pytest.approx(0.0, abs=1e-10)


def test_resolve_bin_centres_delegates_to_resolve_reference_heading() -> None:
    """resolve_bin_centres output must equal calling resolve_reference_heading first."""
    config = PipelineConfig(reference_heading_deg=None)
    headings = np.array([0.0, 90.0, 180.0, 270.0])
    ref = resolve_reference_heading(config, headings)
    centres_direct = resolve_bin_centres(config, headings)
    # N centre should be the angle closest to 0° produced from ref.
    n_centre = centres_direct["N"]
    d = min(n_centre % 360, 360 - n_centre % 360)
    assert d == pytest.approx(min(ref % 360, 360 - ref % 360), abs=1e-10)
```

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_segmentation.py -v
```

Expected: FAIL — `resolve_reference_heading` not yet defined.

**Step 3: Implement `resolve_reference_heading` and `_bin_centres_from_ref` in `lmc/segmentation.py`**

Extract the reference heading resolution into a public helper and the bin-centre computation into a private helper. Then call them from the existing `resolve_bin_centres`.

Replace the current `resolve_bin_centres` function (`lmc/segmentation.py:251-281`) with the following three functions:

```python
def resolve_reference_heading(
    config: PipelineConfig,
    headings: npt.NDArray[np.float64],
) -> float:
    """Resolve the reference heading used for heading-bin assignment.

    Returns ``config.reference_heading_deg`` when it is set; otherwise
    auto-estimates from ``headings`` via a folded circular mean.
    """
    if config.reference_heading_deg is not None:
        return float(config.reference_heading_deg)
    return _estimate_reference_heading(headings)


def _bin_centres_from_ref(ref: float) -> dict[HeadingType, float]:
    """Return the four cardinal bin centres from a pre-resolved reference heading.

    The centres are 90° apart.  The centre assigned the ``"N"`` label is
    whichever of the four has the smallest *angular distance* to 0°;
    ``"E"``, ``"S"``, ``"W"`` follow in clockwise order.
    """
    raw_centres = [(ref + k * 90.0) % 360.0 for k in range(4)]

    def _dist_to_north(angle: float) -> float:
        d = angle % 360.0
        return min(d, 360.0 - d)

    ordered = sorted(raw_centres, key=_dist_to_north)
    north = ordered[0]
    rest = sorted(ordered[1:], key=lambda a: (a - north) % 360.0)
    clockwise = [north] + rest

    return dict(zip(_HEADING_ORDER, clockwise, strict=True))


def resolve_bin_centres(
    config: PipelineConfig,
    headings: npt.NDArray[np.float64],
) -> dict[HeadingType, float]:
    """Return the four cardinal bin centres keyed by ``HeadingType``.

    The centres are 90° apart.  The centre assigned the ``"N"`` label is
    whichever of the four has the smallest *angular distance* to 0° (i.e.
    ``min(d, 360 - d)``); ``"E"``, ``"S"``, ``"W"`` follow in clockwise
    (ascending-degree) order from there.
    """
    ref = resolve_reference_heading(config, headings)
    return _bin_centres_from_ref(ref)
```

Also update the public exports in `lmc/__init__.py` if `resolve_bin_centres` is exported there (check first).

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_segmentation.py -v
```

Expected: all three new tests PASS. Run the full suite to check no regressions:

```bash
uv run pytest -v
```

Expected: all existing tests still PASS.

**Step 5: Commit**

```bash
git add lmc/segmentation.py tests/unit/test_segmentation.py
git commit -m "refactor: extract resolve_reference_heading and _bin_centres_from_ref helpers"
```

---

### Task 2: Add `reference_heading_deg: float` to `HeadingCalibrationResult`

**Files:**
- Modify: `lmc/heading_calibration.py:19-36`
- Modify: `tests/unit/test_heading_calibration.py`

**Step 1: Write the failing tests**

Add these tests to `tests/unit/test_heading_calibration.py` (after the existing tests):

```python
def test_calibrate_per_heading_stores_reference_heading_deg_as_float() -> None:
    df, segments = _make_multi_heading_data(_CONFIG_A)
    result = calibrate_per_heading(df, segments, _CONFIG_A)
    assert isinstance(result.reference_heading_deg, float)


def test_calibrate_per_heading_reference_heading_deg_matches_config_when_set() -> None:
    """When config.reference_heading_deg is set, the stored value must equal it."""
    config = PipelineConfig(model_terms="a", reference_heading_deg=5.0)
    df, segments = _make_multi_heading_data(config)
    result = calibrate_per_heading(df, segments, config)
    assert result.reference_heading_deg == pytest.approx(5.0)


def test_calibrate_per_heading_reference_heading_deg_is_finite_when_auto() -> None:
    """When config.reference_heading_deg is None, stored value must be a finite float."""
    config = PipelineConfig(model_terms="a", reference_heading_deg=None)
    df, segments = _make_multi_heading_data(config)
    result = calibrate_per_heading(df, segments, config)
    assert np.isfinite(result.reference_heading_deg)
```

Add `import pytest` to the top of the test file if not already present.

**Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/unit/test_heading_calibration.py::test_calibrate_per_heading_stores_reference_heading_deg_as_float -v
```

Expected: FAIL — `HeadingCalibrationResult` has no field `reference_heading_deg`.

**Step 3: Add `reference_heading_deg: float` to `HeadingCalibrationResult`**

In `lmc/heading_calibration.py`, update the dataclass (lines 19-36):

```python
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
    reference_heading_deg:
        Reference heading in degrees resolved during calibration.  Stored
        here so compensation can re-use the identical bin centres rather
        than re-estimating from survey headings.
    """

    per_heading: dict[HeadingType, CalibrationResult]
    per_heading_vif: dict[HeadingType, npt.NDArray[np.float64]]
    reference_heading_deg: float
```

**Step 4: Update `calibrate_per_heading` to populate the new field**

In `lmc/heading_calibration.py`, update the imports at the top:

```python
from lmc.calibration import CalibrationResult, calibrate
from lmc.columns import COL_HEADING
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import HeadingType, Segment, resolve_reference_heading
from lmc.vif import compute_vif
```

Then at the end of `calibrate_per_heading`, just before the `return` statement, add the reference heading resolution. The FOM headings are all heading values across all segments, gathered from `df`:

Replace the final `return` statement (lines 104-107):

```python
    # Resolve the reference heading from the FOM calibration data.
    # We collect all heading values covered by the segments to feed into
    # resolve_reference_heading — this mirrors what segment_fom does internally.
    fom_heading_arrays = [
        df.slice(seg.start_idx, seg.end_idx - seg.start_idx)[COL_HEADING].to_numpy()
        for seg in segments
        if COL_HEADING in df.columns
    ]
    if fom_heading_arrays:
        fom_headings = np.concatenate(fom_heading_arrays).astype(np.float64)
    else:
        # No heading column present (explicit-label mode): fall back to config value
        # or 0.0 as a safe default (bin centres will still be consistent).
        fom_headings = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
    resolved_ref = resolve_reference_heading(config, fom_headings)

    return HeadingCalibrationResult(
        per_heading=per_heading,
        per_heading_vif=per_heading_vif,
        reference_heading_deg=resolved_ref,
    )
```

**Step 5: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_heading_calibration.py -v
```

Expected: all tests PASS. Run full suite:

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 6: Commit**

```bash
git add lmc/heading_calibration.py tests/unit/test_heading_calibration.py
git commit -m "feat: persist resolved reference_heading_deg in HeadingCalibrationResult"
```

---

### Task 3: Fix `compensate_heading_specific` to use stored reference heading

**Files:**
- Modify: `lmc/compensation.py:100-138`
- Modify: `tests/unit/test_compensation.py`

**Step 1: Write the failing regression test**

This test demonstrates the actual bug: calibration headings are cardinal (auto-estimate ref ≈ 0°), but survey headings are offset by 30° (auto-estimate ref ≈ 30°). Without the fix, bin centres differ and rows can be misrouted.

Add this test to `tests/unit/test_compensation.py`:

```python
def test_compensate_heading_specific_uses_calibration_reference_not_survey() -> None:
    """Regression: bin centres must come from calibration ref, not survey headings.

    Calibration FOM headings are pure cardinal (0/90/180/270).
    Survey headings are offset by 30° (30/120/210/300).
    Without the fix, resolve_bin_centres would estimate ref ≈ 30° from the survey
    headings, misrouting rows to the wrong coefficient set.
    With the fix, calibration ref (≈ 0°) is reused and routing is stable.
    """
    config = PipelineConfig(model_terms="a", reference_heading_deg=None)
    rng = np.random.default_rng(42)

    # --- Build calibration data with cardinal headings (0/90/180/270) ---
    n_per = 60
    cal_headings_map = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    cal_blocks: list[pl.DataFrame] = []
    cal_segments: list[Segment] = []
    offset = 0
    for h_label, h_deg in cal_headings_map.items():
        raw = rng.standard_normal((n_per, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        cosines = raw / norms
        b_total = 50_000.0
        df_block = pl.DataFrame({
            COL_TIME: np.arange(offset, offset + n_per, dtype=np.float64),
            COL_LAT: np.full(n_per, 45.0),
            COL_LON: np.full(n_per, -75.0),
            COL_ALT: np.full(n_per, 300.0),
            COL_BTOTAL: np.full(n_per, b_total),
            COL_BX: cosines[:, 0] * b_total,
            COL_BY: cosines[:, 1] * b_total,
            COL_BZ: cosines[:, 2] * b_total,
            COL_HEADING: np.full(n_per, h_deg),
        })
        A = build_feature_matrix(df_block, config).to_numpy()
        c_true = rng.standard_normal(A.shape[1])
        delta_b = (A @ c_true).astype(np.float64)
        df_block = df_block.with_columns(
            pl.Series(COL_DELTA_B, delta_b, dtype=pl.Float64)
        )
        cal_blocks.append(df_block)
        cal_segments.append(
            Segment(maneuver="steady", heading=h_label, start_idx=offset, end_idx=offset + n_per)  # type: ignore[arg-type]
        )
        offset += n_per

    cal_df = pl.concat(cal_blocks)
    cal_result = calibrate_per_heading(cal_df, cal_segments, config)

    # Verify calibration stored ref ≈ 0° (cardinal headings).
    assert cal_result.reference_heading_deg == pytest.approx(0.0, abs=1.0)

    # --- Build survey data with offset headings (30/120/210/300) ---
    survey_headings_map = {"N": 30.0, "E": 120.0, "S": 210.0, "W": 300.0}
    survey_blocks: list[pl.DataFrame] = []
    for h_deg in survey_headings_map.values():
        raw = rng.standard_normal((n_per, 3))
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        cosines = raw / norms
        b_total = 50_000.0
        df_block = pl.DataFrame({
            COL_TIME: np.arange(n_per, dtype=np.float64),
            COL_LAT: np.full(n_per, 45.0),
            COL_LON: np.full(n_per, -75.0),
            COL_ALT: np.full(n_per, 300.0),
            COL_BTOTAL: np.full(n_per, b_total),
            COL_BX: cosines[:, 0] * b_total,
            COL_BY: cosines[:, 1] * b_total,
            COL_BZ: cosines[:, 2] * b_total,
            COL_HEADING: np.full(n_per, h_deg),
        })
        survey_blocks.append(df_block)
    survey_df = pl.concat(survey_blocks)

    # Should not raise; all survey rows must be assigned to a calibrated heading.
    result_df = compensate_heading_specific(survey_df, cal_result, config)
    assert COL_TMI_COMPENSATED in result_df.columns
    assert len(result_df) == len(survey_df)
    # No NaN values — every row was routed to a coefficient set.
    assert not result_df[COL_TMI_COMPENSATED].is_nan().any()
```

**Step 2: Run test to see it fail (or pass trivially for wrong reasons)**

```bash
uv run pytest "tests/unit/test_compensation.py::test_compensate_heading_specific_uses_calibration_reference_not_survey" -v
```

Note: this test may not fail obviously in all cases — the issue is subtle misrouting. The key assertion to verify is `cal_result.reference_heading_deg == pytest.approx(0.0, abs=1.0)`.

**Step 3: Update `compensate_heading_specific` in `lmc/compensation.py`**

Update the import at the top of the function body (line 100):

Replace:
```python
    from lmc.segmentation import HeadingType, assign_heading_bin, resolve_bin_centres
```

With:
```python
    from lmc.segmentation import HeadingType, assign_heading_bin, _bin_centres_from_ref
```

Replace the two lines that call `resolve_bin_centres` (lines 119-124):

```python
    headings = np.asarray(df[COL_HEADING].to_numpy(), dtype=np.float64)
    all_centres = resolve_bin_centres(config, headings)
    # Restrict to calibrated headings so every row routes to an available model.
    centres: dict[HeadingType, float] = {
        k: v for k, v in all_centres.items() if k in result.per_heading
    }
```

With:

```python
    headings = np.asarray(df[COL_HEADING].to_numpy(), dtype=np.float64)
    # Use the reference heading resolved during calibration — not re-estimated from
    # survey headings — so bin centres are identical to those used at calibration time.
    all_centres = _bin_centres_from_ref(result.reference_heading_deg)
    # Restrict to calibrated headings so every row routes to an available model.
    centres: dict[HeadingType, float] = {
        k: v for k, v in all_centres.items() if k in result.per_heading
    }
```

**Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/unit/test_compensation.py -v
```

Expected: all tests PASS. Full suite:

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 5: Lint**

```bash
uv run make lint
```

Fix any ruff or type errors before committing.

**Step 6: Commit**

```bash
git add lmc/compensation.py tests/unit/test_compensation.py
git commit -m "fix: use calibration reference heading in compensate_heading_specific"
```

---

### Task 4: Final verification

**Step 1: Run full test suite**

```bash
uv run pytest -v
```

Expected: all tests PASS.

**Step 2: Run full lint**

```bash
uv run make lint
```

Expected: no errors.

**Step 3: Verify the integration tests still pass**

```bash
uv run pytest tests/integration/ -v
```

Expected: all integration tests PASS.
