"""Aeromagnetic figures of merit (FOM) for calibration quality assessment."""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.calibration import CalibrationResult
from lmc.columns import COL_DELTA_B
from lmc.segmentation import HeadingType, ManeuverType, Segment


@dataclass(frozen=True)
class ManeuverStats:
    """FOM statistics aggregated across all segments of a given maneuver type.

    Attributes
    ----------
    rms_before:
        RMS of ``δB`` across all segments with this maneuver.
    rms_after:
        RMS of residuals (``A @ coef − δB``) for the same segments.
    n_samples:
        Total row count across contributing segments.
    """

    rms_before: float
    rms_after: float
    n_samples: int


@dataclass(frozen=True)
class HeadingStats:
    """FOM statistics aggregated across all segments of a given heading.

    Attributes
    ----------
    rms_before:
        RMS of ``δB`` across all segments with this heading.
    rms_after:
        RMS of residuals for the same segments.
    n_samples:
        Total row count across contributing segments.
    """

    rms_before: float
    rms_after: float
    n_samples: int


@dataclass(frozen=True)
class FomReport:
    """Figure-of-merit report for a Tolles-Lawson calibration.

    Attributes
    ----------
    per_maneuver:
        Stats grouped by maneuver type.
    per_heading:
        Stats grouped by cardinal heading.
    improvement_ratio:
        ``std(all δB) / std(all residuals)``; higher values indicate better
        compensation.  Returns ``float('inf')`` when residuals are identically zero.
    """

    per_maneuver: dict[ManeuverType, ManeuverStats]
    per_heading: dict[HeadingType, HeadingStats]
    improvement_ratio: float

    def to_json(self) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(dataclasses.asdict(self))


def compute_fom_report(
    df: pl.DataFrame,
    segments: list[Segment],
    result: CalibrationResult,
) -> FomReport:
    """Compute aeromagnetic figures of merit from a calibration result.

    Parameters
    ----------
    df:
        Full calibration DataFrame containing ``COL_DELTA_B``.
    segments:
        Non-empty list of labeled flight segments used during calibration.
    result:
        Calibration result whose ``residuals`` correspond to the concatenated
        segment rows in the order supplied.

    Returns
    -------
    FomReport
        Per-maneuver stats, per-heading stats, and global improvement ratio.

    Raises
    ------
    ValueError
        If ``COL_DELTA_B`` is absent from ``df``, ``segments`` is empty, or
        ``len(result.residuals)`` does not match the total segment row count.
    """
    if COL_DELTA_B not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_B}' is required but was not found in the DataFrame. "
            f"Available columns: {df.columns}"
        )

    if not segments:
        raise ValueError("segments must be non-empty; cannot compute FOM with no data.")

    total_rows = sum(seg.end_idx - seg.start_idx for seg in segments)
    if len(result.residuals) != total_rows:
        raise ValueError(
            f"Length of result.residuals ({len(result.residuals)}) does not match "
            f"the total number of rows across all segments ({total_rows})."
        )

    # Accumulate (db_seg, res_seg) arrays keyed by maneuver and heading.
    maneuver_db: dict[ManeuverType, list[npt.NDArray[np.float64]]] = defaultdict(list)
    maneuver_res: dict[ManeuverType, list[npt.NDArray[np.float64]]] = defaultdict(list)
    heading_db: dict[HeadingType, list[npt.NDArray[np.float64]]] = defaultdict(list)
    heading_res: dict[HeadingType, list[npt.NDArray[np.float64]]] = defaultdict(list)

    all_db: list[npt.NDArray[np.float64]] = []
    all_res: list[npt.NDArray[np.float64]] = []

    offset = 0
    for seg in segments:
        n = seg.end_idx - seg.start_idx
        db_seg = df.slice(seg.start_idx, n)[COL_DELTA_B].to_numpy().astype(np.float64)
        res_seg = result.residuals[offset : offset + n].astype(np.float64)
        offset += n

        maneuver_db[seg.maneuver].append(db_seg)
        maneuver_res[seg.maneuver].append(res_seg)
        heading_db[seg.heading].append(db_seg)
        heading_res[seg.heading].append(res_seg)
        all_db.append(db_seg)
        all_res.append(res_seg)

    def _rms(arr: npt.NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(arr**2)))

    # Build per-maneuver stats.
    per_maneuver: dict[ManeuverType, ManeuverStats] = {}
    for maneuver, db_list in maneuver_db.items():
        db_all = np.concatenate(db_list)
        res_all = np.concatenate(maneuver_res[maneuver])
        per_maneuver[maneuver] = ManeuverStats(
            rms_before=_rms(db_all),
            rms_after=_rms(res_all),
            n_samples=len(db_all),
        )

    # Build per-heading stats.
    per_heading: dict[HeadingType, HeadingStats] = {}
    for heading, db_list in heading_db.items():
        db_all = np.concatenate(db_list)
        res_all = np.concatenate(heading_res[heading])
        per_heading[heading] = HeadingStats(
            rms_before=_rms(db_all),
            rms_after=_rms(res_all),
            n_samples=len(db_all),
        )

    # Global improvement ratio: std(before) / std(after).
    global_db = np.concatenate(all_db)
    global_res = np.concatenate(all_res)
    std_before = float(np.std(global_db))
    std_after = float(np.std(global_res))
    improvement_ratio = float("inf") if std_after == 0.0 else std_before / std_after

    return FomReport(
        per_maneuver=per_maneuver,
        per_heading=per_heading,
        improvement_ratio=improvement_ratio,
    )
