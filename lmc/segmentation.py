"""FOM test flight segmentation into (maneuver, heading) pairs."""

from __future__ import annotations

from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.columns import COL_HEADING, COL_PITCH, COL_ROLL, COL_TIME
from lmc.config import PipelineConfig

ManeuverType = Literal["steady", "pitch", "roll", "yaw"]
HeadingType = Literal["N", "E", "S", "W"]

# Attitude-rate threshold (°/s) below which all axes are considered steady.
_RATE_THRESHOLD: float = 1.0

# Ordered heading labels clockwise from North.
_HEADING_ORDER: tuple[HeadingType, ...] = ("N", "E", "S", "W")


@dataclass(frozen=True)
class Segment:
    """A labeled window of FOM flight data.

    Attributes
    ----------
    maneuver:
        One of ``"steady"``, ``"pitch"``, ``"roll"``, ``"yaw"``.
    heading:
        Cardinal heading label: ``"N"``, ``"E"``, ``"S"``, or ``"W"``.
    start_idx:
        First row index of the segment (inclusive).
    end_idx:
        One-past-last row index of the segment (exclusive, Python slice convention).
    """

    maneuver: ManeuverType
    heading: HeadingType
    start_idx: int
    end_idx: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def segment_fom(df: pl.DataFrame, config: PipelineConfig) -> list[Segment]:
    """Segment a FOM calibration DataFrame into labeled (maneuver, heading) windows.

    Parameters
    ----------
    df:
        Input DataFrame.  Must contain the columns required by the chosen mode.
    config:
        Pipeline configuration.  If ``config.segment_label_col`` is set the
        function uses explicit (pre-labeled) mode; otherwise auto-detection is used.

    Returns
    -------
    list[Segment]
        Segments in DataFrame row order.

    Raises
    ------
    ValueError
        If required columns are missing or label strings are invalid.
    """
    if config.segment_label_col is not None:
        return _explicit_segments(df, config)
    return _auto_detect_segments(df, config)


# ---------------------------------------------------------------------------
# Explicit mode
# ---------------------------------------------------------------------------


def _explicit_segments(df: pl.DataFrame, config: PipelineConfig) -> list[Segment]:
    label_col = config.segment_label_col
    if label_col is None:
        raise ValueError("segment_label_col must be set to use explicit mode.")

    if label_col not in df.columns:
        raise ValueError(
            f"segment_label_col '{label_col}' not found in DataFrame columns: "
            f"{df.columns}"
        )

    labels: list[str] = df[label_col].cast(pl.Utf8).to_list()
    segments: list[Segment] = []

    for raw_label, start, end in _consecutive_runs(labels):
        str_label: str = str(raw_label)
        maneuver, heading = _parse_label(str_label)
        segments.append(
            Segment(maneuver=maneuver, heading=heading, start_idx=start, end_idx=end)
        )

    return segments


# ---------------------------------------------------------------------------
# Auto-detect mode
# ---------------------------------------------------------------------------


def _auto_detect_segments(df: pl.DataFrame, config: PipelineConfig) -> list[Segment]:
    if COL_HEADING not in df.columns:
        raise ValueError(
            f"Auto-detect mode requires column '{COL_HEADING}'. "
            "Add the heading column or use explicit mode via config.segment_label_col."
        )

    missing_attitude = [c for c in (COL_PITCH, COL_ROLL) if c not in df.columns]
    if missing_attitude:
        raise ValueError(
            f"Auto-detect mode requires attitude columns {missing_attitude}. "
            "Add pitch/roll columns or use explicit mode via config.segment_label_col."
        )

    if COL_TIME not in df.columns:
        raise ValueError(f"Auto-detect mode requires column '{COL_TIME}'.")

    headings = np.asarray(df[COL_HEADING].to_numpy(), dtype=np.float64)
    centres = _resolve_bin_centres(config, headings)

    # Assign heading bin to each row (None for transitional rows).
    heading_labels: list[HeadingType | None] = [
        _assign_heading_bin(h, centres, config.heading_tolerance_deg) for h in headings
    ]

    time = np.asarray(df[COL_TIME].to_numpy(), dtype=np.float64)
    pitch = np.asarray(df[COL_PITCH].to_numpy(), dtype=np.float64)
    roll = np.asarray(df[COL_ROLL].to_numpy(), dtype=np.float64)
    heading_unwrapped = np.unwrap(np.deg2rad(headings))

    dpitch_dt = np.gradient(pitch, time)
    droll_dt = np.gradient(roll, time)
    dyaw_dt = np.rad2deg(np.gradient(heading_unwrapped, time))

    # Zero out dyaw_dt at the FIRST row of each new heading bin.
    # np.gradient uses a 3-point stencil: a heading jump from bin A (row i-1)
    # to bin B (row i) contaminates dyaw_dt[i] because it looks back into bin A.
    # Zeroing dyaw_dt[i] prevents a spurious yaw detection at the start of the
    # new block.  We intentionally leave dyaw_dt[i-1] unchanged: its contaminated
    # large value happens to be the correct dominant rate when the preceding
    # maneuver is yaw (the standard final FOM maneuver before a heading change).
    for i in range(1, len(heading_labels)):
        if heading_labels[i] != heading_labels[i - 1]:
            dyaw_dt[i] = 0.0

    maneuver_labels: npt.NDArray[np.object_] = _classify_maneuver(
        dpitch_dt, droll_dt, dyaw_dt
    )

    # Combine heading + maneuver labels per row; None rows are excluded.
    combined: list[tuple[ManeuverType, HeadingType] | None] = []
    for i, h_label in enumerate(heading_labels):
        if h_label is None:
            combined.append(None)
        else:
            combined.append((maneuver_labels[i], h_label))

    segments: list[Segment] = []
    for raw_label, start, end in _consecutive_runs(combined):
        if raw_label is None:
            continue
        pair: tuple[ManeuverType, HeadingType] = raw_label
        segments.append(
            Segment(maneuver=pair[0], heading=pair[1], start_idx=start, end_idx=end)
        )

    return segments


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _consecutive_runs(
    labels: list[Any],
) -> Generator[tuple[Any, int, int], None, None]:
    """Yield ``(label, start, end)`` for each consecutive run of equal values."""
    if not labels:
        return
    current: Any = labels[0]
    start: int = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            yield current, start, i
            current = labels[i]
            start = i
    yield current, start, len(labels)


def _parse_label(label: str) -> tuple[ManeuverType, HeadingType]:
    """Parse ``'<maneuver>_<heading>'`` into a validated pair."""
    parts = label.split("_")
    if len(parts) != 2:
        raise ValueError(
            f"Invalid segment label '{label}'. Expected format '<maneuver>_<heading>', "
            "e.g. 'pitch_N'."
        )
    raw_maneuver, raw_heading = parts
    valid_maneuvers: tuple[str, ...] = ("steady", "pitch", "roll", "yaw")
    valid_headings: tuple[str, ...] = ("N", "E", "S", "W")

    if raw_maneuver not in valid_maneuvers:
        raise ValueError(
            f"Unrecognised maneuver '{raw_maneuver}' in label '{label}'. "
            f"Must be one of {valid_maneuvers}."
        )
    if raw_heading not in valid_headings:
        raise ValueError(
            f"Unrecognised heading '{raw_heading}' in label '{label}'. "
            f"Must be one of {valid_headings}."
        )
    return raw_maneuver, raw_heading  # type: ignore[return-value]


def _estimate_reference_heading(headings: npt.NDArray[np.float64]) -> float:
    """Estimate the northernmost bin centre via folded circular mean.

    Folds all headings into [0°, 90°) then computes the circular mean,
    yielding a reference angle in [0°, 90°).
    """
    h_mod = headings % 90.0
    angles_rad = np.deg2rad(h_mod)
    sin_mean = np.sin(angles_rad).mean()
    cos_mean = np.cos(angles_rad).mean()
    ref = np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 90.0
    return float(ref)


def _resolve_bin_centres(
    config: PipelineConfig,
    headings: npt.NDArray[np.float64],
) -> dict[HeadingType, float]:
    """Return the four cardinal bin centres keyed by ``HeadingType``.

    The centres are 90° apart.  The N centre is the smallest clockwise angle
    to 0° (true north); E, S, W follow in clockwise order.
    """
    if config.reference_heading_deg is not None:
        ref = float(config.reference_heading_deg)
    else:
        ref = _estimate_reference_heading(headings)

    raw_centres = [(ref + k * 90.0) % 360.0 for k in range(4)]

    # Sort to find which centre is closest to 0°.
    def _dist_to_north(angle: float) -> float:
        d = angle % 360.0
        return min(d, 360.0 - d)

    ordered = sorted(raw_centres, key=_dist_to_north)
    # ordered[0] is the "N" centre; the rest follow clockwise.
    # Re-sort the remaining three in clockwise order (ascending degrees mod 360).
    north = ordered[0]
    rest = sorted(ordered[1:], key=lambda a: (a - north) % 360.0)
    clockwise = [north] + rest

    return dict(zip(_HEADING_ORDER, clockwise, strict=True))


def _assign_heading_bin(
    deg: float,
    centres: dict[HeadingType, float],
    tol: float,
) -> HeadingType | None:
    """Map a single compass bearing to the nearest cardinal label within tolerance."""
    best_label: HeadingType | None = None
    best_dist = float("inf")
    for label, centre in centres.items():
        d = abs((deg - centre + 180.0) % 360.0 - 180.0)
        if d < best_dist:
            best_dist = d
            best_label = label
    if best_dist <= tol:
        return best_label
    return None


def _classify_maneuver(
    dpitch_dt: npt.NDArray[np.float64],
    droll_dt: npt.NDArray[np.float64],
    dyaw_dt: npt.NDArray[np.float64],
) -> npt.NDArray[np.object_]:
    """Map per-row attitude-rate arrays to ``ManeuverType`` labels.

    Classification rule (per row):
    - ``"steady"`` if all rates are below ``_RATE_THRESHOLD``
    - ``"pitch"``  if ``|dpitch_dt|`` is the dominant rate
    - ``"roll"``   if ``|droll_dt|`` is the dominant rate
    - ``"yaw"``    if ``|dyaw_dt|`` is the dominant rate
    """
    abs_pitch = np.abs(dpitch_dt)
    abs_roll = np.abs(droll_dt)
    abs_yaw = np.abs(dyaw_dt)

    n = len(dpitch_dt)
    result = np.empty(n, dtype=object)

    for i in range(n):
        ap, ar, ay = abs_pitch[i], abs_roll[i], abs_yaw[i]
        if max(ap, ar, ay) < _RATE_THRESHOLD:
            result[i] = "steady"
        elif ap >= ar and ap >= ay:
            result[i] = "pitch"
        elif ar >= ap and ar >= ay:
            result[i] = "roll"
        else:
            result[i] = "yaw"

    return result
