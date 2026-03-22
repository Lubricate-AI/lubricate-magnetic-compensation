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

        # Rebuild the stacked A matrix for VIF computation.  calibrate() builds
        # the same matrix internally but does not expose it, so we reconstruct
        # it here.  The cost is a second call to build_feature_matrix per heading.
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
