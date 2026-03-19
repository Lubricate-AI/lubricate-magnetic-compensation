"""Unit tests for lmc.adaptive."""

from __future__ import annotations

import numpy as np
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
