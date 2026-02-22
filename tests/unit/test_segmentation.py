"""Unit tests for lmc.segmentation."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from lmc.columns import (
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_HEADING,
    COL_PITCH,
    COL_ROLL,
    COL_TIME,
)
from lmc.config import PipelineConfig
from lmc.segmentation import (
    _estimate_reference_heading,  # pyright: ignore[reportPrivateUsage]
    segment_fom,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MANEUVERS = ("steady", "pitch", "roll", "yaw")
_HEADINGS = ("N", "E", "S", "W")
_HEADING_DEGS = {"N": 2.0, "E": 92.0, "S": 182.0, "W": 272.0}
_OBLIQUE_HEADING_DEGS = {"N": 47.0, "E": 137.0, "S": 227.0, "W": 317.0}
_ROWS_PER_SEGMENT = 10

# ---------------------------------------------------------------------------
# Synthetic data factory
# ---------------------------------------------------------------------------


def _make_fom_df(
    rows_per_segment: int = _ROWS_PER_SEGMENT,
    heading_map: dict[str, float] | None = None,
    include_label_col: bool = False,
) -> pl.DataFrame:
    """Build a 160-row FOM DataFrame (16 segments × rows_per_segment rows).

    Order: headings N→E→S→W, maneuvers steady→pitch→roll→yaw within each heading.
    Attitude signals are constructed so the dominant rate matches the maneuver label.
    """
    if heading_map is None:
        heading_map = _HEADING_DEGS

    rows: list[dict[str, float | str]] = []
    t = 0.0
    dt = 0.1  # 10 Hz

    # Rate amplitude to be clearly above the 1 °/s threshold.
    _RATE_AMP = 5.0

    for h_label in _HEADINGS:
        h_deg = heading_map[h_label]
        for maneuver in _MANEUVERS:
            seg_labels: list[str] = [f"{maneuver}_{h_label}"] * rows_per_segment
            for k in range(rows_per_segment):
                # Build attitude signals so the dominant rate is clearly dominant.
                if maneuver == "steady":
                    pitch = 0.0
                    roll = 0.0
                    heading = h_deg
                elif maneuver == "pitch":
                    # increasing pitch → dpitch/dt ≈ _RATE_AMP
                    pitch = k * _RATE_AMP * dt
                    roll = 0.0
                    heading = h_deg
                elif maneuver == "roll":
                    pitch = 0.0
                    roll = k * _RATE_AMP * dt  # increasing roll → droll/dt ≈ _RATE_AMP
                    heading = h_deg
                else:  # yaw
                    pitch = 0.0
                    roll = 0.0
                    # Slowly varying heading → dyaw/dt ≈ _RATE_AMP
                    heading = (h_deg + k * _RATE_AMP * dt) % 360.0

                rows.append(
                    {
                        COL_TIME: t,
                        COL_HEADING: heading,
                        COL_PITCH: pitch,
                        COL_ROLL: roll,
                        COL_BX: 20_000.0,
                        COL_BY: 1_000.0,
                        COL_BZ: 45_000.0,
                        COL_BTOTAL: 50_000.0,
                        "segment": seg_labels[k],
                    }
                )
                t += dt

    df = pl.DataFrame(rows)
    if not include_label_col:
        df = df.drop("segment")
    return df


def _make_fom_df_with_labels(rows_per_segment: int = _ROWS_PER_SEGMENT) -> pl.DataFrame:
    return _make_fom_df(rows_per_segment=rows_per_segment, include_label_col=True)


# ---------------------------------------------------------------------------
# Explicit mode tests
# ---------------------------------------------------------------------------


def test_explicit_returns_16_segments() -> None:
    df = _make_fom_df_with_labels()
    cfg = PipelineConfig(segment_label_col="segment")
    segs = segment_fom(df, cfg)
    assert len(segs) == 16


def test_explicit_segment_order() -> None:
    df = _make_fom_df_with_labels()
    cfg = PipelineConfig(segment_label_col="segment")
    segs = segment_fom(df, cfg)
    # start_idx of each segment must be strictly increasing and non-overlapping.
    for i in range(1, len(segs)):
        assert segs[i].start_idx == segs[i - 1].end_idx


def test_explicit_correct_maneuver_heading() -> None:
    df = _make_fom_df_with_labels()
    cfg = PipelineConfig(segment_label_col="segment")
    segs = segment_fom(df, cfg)
    # Segment 0 should be steady_N, segment 4 should be steady_E, etc.
    expected = [
        ("steady", "N"),
        ("pitch", "N"),
        ("roll", "N"),
        ("yaw", "N"),
        ("steady", "E"),
        ("pitch", "E"),
        ("roll", "E"),
        ("yaw", "E"),
        ("steady", "S"),
        ("pitch", "S"),
        ("roll", "S"),
        ("yaw", "S"),
        ("steady", "W"),
        ("pitch", "W"),
        ("roll", "W"),
        ("yaw", "W"),
    ]
    for i, (exp_maneuver, exp_heading) in enumerate(expected):
        assert segs[i].maneuver == exp_maneuver, f"seg {i} maneuver mismatch"
        assert segs[i].heading == exp_heading, f"seg {i} heading mismatch"


def test_explicit_correct_start_end_idx() -> None:
    n = _ROWS_PER_SEGMENT
    df = _make_fom_df_with_labels(rows_per_segment=n)
    cfg = PipelineConfig(segment_label_col="segment")
    segs = segment_fom(df, cfg)
    for i, seg in enumerate(segs):
        assert seg.start_idx == i * n
        assert seg.end_idx == (i + 1) * n


def test_explicit_missing_column_raises() -> None:
    df = _make_fom_df_with_labels().drop("segment")
    cfg = PipelineConfig(segment_label_col="segment")
    with pytest.raises(ValueError, match="segment"):
        segment_fom(df, cfg)


def test_explicit_invalid_maneuver_raises() -> None:
    df = _make_fom_df_with_labels()
    # Overwrite one label with an invalid maneuver.
    bad_labels = df["segment"].to_list()
    bad_labels[0] = "hover_N"
    df = df.with_columns(pl.Series("segment", bad_labels))
    cfg = PipelineConfig(segment_label_col="segment")
    with pytest.raises(ValueError, match="hover"):
        segment_fom(df, cfg)


def test_explicit_invalid_heading_raises() -> None:
    df = _make_fom_df_with_labels()
    bad_labels = df["segment"].to_list()
    bad_labels[0] = "pitch_X"
    df = df.with_columns(pl.Series("segment", bad_labels))
    cfg = PipelineConfig(segment_label_col="segment")
    with pytest.raises(ValueError, match="X"):
        segment_fom(df, cfg)


def test_explicit_single_segment() -> None:
    df = pl.DataFrame(
        {
            "segment": ["roll_W"] * 5,
            COL_TIME: [0.0, 0.1, 0.2, 0.3, 0.4],
            COL_HEADING: [270.0] * 5,
            COL_PITCH: [0.0] * 5,
            COL_ROLL: [0.0] * 5,
        }
    )
    cfg = PipelineConfig(segment_label_col="segment")
    segs = segment_fom(df, cfg)
    assert len(segs) == 1
    assert segs[0].maneuver == "roll"
    assert segs[0].heading == "W"
    assert segs[0].start_idx == 0
    assert segs[0].end_idx == 5


# ---------------------------------------------------------------------------
# Auto-detect mode tests
# ---------------------------------------------------------------------------


def test_auto_detect_returns_16_segments() -> None:
    df = _make_fom_df()
    cfg = PipelineConfig(reference_heading_deg=2.0)
    segs = segment_fom(df, cfg)
    assert len(segs) == 16


def test_auto_detect_heading_binning() -> None:
    df = _make_fom_df()
    cfg = PipelineConfig(reference_heading_deg=2.0)
    segs = segment_fom(df, cfg)
    headings_seen = {s.heading for s in segs}
    assert headings_seen == {"N", "E", "S", "W"}


def test_auto_detect_steady_classified() -> None:
    """Rows with near-zero attitude rates should be classified as steady."""
    n = _ROWS_PER_SEGMENT
    df = pl.DataFrame(
        {
            COL_TIME: [float(i) * 0.1 for i in range(n)],
            COL_HEADING: [2.0] * n,
            COL_PITCH: [0.0] * n,
            COL_ROLL: [0.0] * n,
        }
    )
    cfg = PipelineConfig(reference_heading_deg=2.0)
    segs = segment_fom(df, cfg)
    assert len(segs) == 1
    assert segs[0].maneuver == "steady"
    assert segs[0].heading == "N"


def test_auto_detect_pitch_classified() -> None:
    """Dominant dpitch/dt → pitch maneuver."""
    n = _ROWS_PER_SEGMENT
    rate = 5.0  # °/s, well above 1.0 threshold
    dt = 0.1
    df = pl.DataFrame(
        {
            COL_TIME: [float(i) * dt for i in range(n)],
            COL_HEADING: [2.0] * n,
            COL_PITCH: [float(i) * rate * dt for i in range(n)],
            COL_ROLL: [0.0] * n,
        }
    )
    cfg = PipelineConfig(reference_heading_deg=2.0)
    segs = segment_fom(df, cfg)
    assert len(segs) >= 1
    assert all(s.maneuver == "pitch" for s in segs)


def test_auto_detect_roll_classified() -> None:
    """Dominant droll/dt → roll maneuver."""
    n = _ROWS_PER_SEGMENT
    rate = 5.0
    dt = 0.1
    df = pl.DataFrame(
        {
            COL_TIME: [float(i) * dt for i in range(n)],
            COL_HEADING: [2.0] * n,
            COL_PITCH: [0.0] * n,
            COL_ROLL: [float(i) * rate * dt for i in range(n)],
        }
    )
    cfg = PipelineConfig(reference_heading_deg=2.0)
    segs = segment_fom(df, cfg)
    assert len(segs) >= 1
    assert all(s.maneuver == "roll" for s in segs)


def test_auto_detect_yaw_classified() -> None:
    """Dominant dyaw/dt → yaw maneuver."""
    n = _ROWS_PER_SEGMENT
    rate = 5.0
    dt = 0.1
    # Heading stays within N bin but changes fast enough to trigger yaw.
    df = pl.DataFrame(
        {
            COL_TIME: [float(i) * dt for i in range(n)],
            COL_HEADING: [2.0 + float(i) * rate * dt for i in range(n)],
            COL_PITCH: [0.0] * n,
            COL_ROLL: [0.0] * n,
        }
    )
    cfg = PipelineConfig(reference_heading_deg=2.0, heading_tolerance_deg=45.0)
    segs = segment_fom(df, cfg)
    assert len(segs) >= 1
    assert all(s.maneuver == "yaw" for s in segs)


def test_auto_detect_missing_heading_column_raises() -> None:
    df = pl.DataFrame({COL_TIME: [0.0], COL_PITCH: [0.0], COL_ROLL: [0.0]})
    cfg = PipelineConfig()
    with pytest.raises(ValueError, match="heading"):
        segment_fom(df, cfg)


def test_auto_detect_missing_attitude_columns_raises() -> None:
    df = pl.DataFrame({COL_TIME: [0.0], COL_HEADING: [0.0]})
    cfg = PipelineConfig()
    with pytest.raises(ValueError, match="pitch"):
        segment_fom(df, cfg)


# ---------------------------------------------------------------------------
# Heading estimation tests
# ---------------------------------------------------------------------------


def test_estimate_reference_heading_cardinal() -> None:
    """Cardinal headings (0/90/180/270) should fold to ref ≈ 0°."""
    headings = np.array([2.0, 92.0, 182.0, 272.0] * 10)
    ref = _estimate_reference_heading(headings)
    # All values fold to ~2°; circular mean of [2,2,...] in [0,90) should be ~2°.
    assert abs(ref - 2.0) < 5.0


def test_estimate_reference_heading_oblique_45() -> None:
    """Oblique headings (45/135/225/315) fold to ref ≈ 45°."""
    headings = np.array([47.0, 137.0, 227.0, 317.0] * 10)
    ref = _estimate_reference_heading(headings)
    assert abs(ref - 47.0) < 5.0


def test_explicit_reference_heading_overrides_auto_detect() -> None:
    """When reference_heading_deg=45, cardinal data is binned to oblique centres."""
    df = _make_fom_df()  # heading values near 2, 92, 182, 272
    # Force oblique bin centres at 45/135/225/315; cardinal rows fall outside all bins.
    cfg = PipelineConfig(reference_heading_deg=45.0, heading_tolerance_deg=20.0)
    segs = segment_fom(df, cfg)
    # All rows are outside the 45°-oblique bins → no segments expected.
    assert len(segs) == 0


def test_auto_detect_oblique_returns_16_segments() -> None:
    """Oblique FOM data (45/135/225/315) with auto-detect should yield 16 segments."""
    df = _make_fom_df(heading_map=_OBLIQUE_HEADING_DEGS)
    cfg = PipelineConfig()  # reference_heading_deg=None → auto-detect
    segs = segment_fom(df, cfg)
    assert len(segs) == 16
