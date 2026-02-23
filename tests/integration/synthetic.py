"""Synthetic FOM calibration data generator for integration tests."""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc import PipelineConfig, build_feature_matrix
from lmc.columns import (
    COL_ALT,
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_HEADING,
    COL_LAT,
    COL_LON,
    COL_PITCH,
    COL_ROLL,
    COL_SEGMENT_LABEL,
    COL_TIME,
)


def _rotation_matrix(
    heading_deg: float, pitch_deg: float, roll_deg: float
) -> npt.NDArray[np.float64]:
    """Compute DCM from geographic (North-East-Down) to body frame.

    R = Rx(roll) @ Ry(pitch) @ Rz(heading)
    """
    h = np.radians(heading_deg)
    p = np.radians(pitch_deg)
    r = np.radians(roll_deg)

    rz: npt.NDArray[np.float64] = np.array(
        [
            [np.cos(h), np.sin(h), 0.0],
            [-np.sin(h), np.cos(h), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    ry: npt.NDArray[np.float64] = np.array(
        [
            [np.cos(p), 0.0, -np.sin(p)],
            [0.0, 1.0, 0.0],
            [np.sin(p), 0.0, np.cos(p)],
        ],
        dtype=np.float64,
    )
    rx: npt.NDArray[np.float64] = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(r), np.sin(r)],
            [0.0, -np.sin(r), np.cos(r)],
        ],
        dtype=np.float64,
    )
    result: npt.NDArray[np.float64] = rx @ ry @ rz
    return result


def make_fom_dataframe(
    c_true: npt.NDArray[np.float64],
    *,
    n_rows_per_block: int = 50,
    noise_std: float = 0.1,
    seed: int = 42,
) -> pl.DataFrame:
    """Generate a synthetic FOM calibration DataFrame.

    Parameters
    ----------
    c_true:
        True Tolles-Lawson coefficients. Length determines model_terms:
        3 -> "a", 9 -> "b", 18 -> "c".
    n_rows_per_block:
        Number of rows in each of the 16 attitude blocks.
    noise_std:
        Standard deviation of Gaussian noise added to interference (nT).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    pl.DataFrame
        DataFrame with all REQUIRED_COLUMNS + heading, pitch, roll, segment columns.
    """
    n_terms = len(c_true)
    term_map: dict[int, Literal["a", "b", "c"]] = {3: "a", 9: "b", 18: "c"}
    if n_terms not in term_map:
        raise ValueError(f"c_true length must be 3, 9, or 18; got {n_terms}")
    model_terms = term_map[n_terms]

    rng = np.random.default_rng(seed)

    # Fixed Earth field in geographic frame [North, East, Down] (nT)
    b_earth: npt.NDArray[np.float64] = np.array(
        [40000.0, 0.0, 20000.0], dtype=np.float64
    )
    b_total_true = float(np.linalg.norm(b_earth))  # ~44721 nT

    headings: dict[str, float] = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
    maneuvers = ["steady", "pitch", "roll", "yaw"]

    row_time_offset = 0
    blocks: list[pl.DataFrame] = []

    for heading_name, h_deg in headings.items():
        for maneuver in maneuvers:
            n = n_rows_per_block
            k = np.arange(n, dtype=np.float64)
            phase = 2.0 * np.pi * k / float(n)

            if maneuver == "steady":
                pitch_arr = np.zeros(n, dtype=np.float64)
                roll_arr = np.zeros(n, dtype=np.float64)
                heading_arr = np.full(n, h_deg, dtype=np.float64)
            elif maneuver == "pitch":
                pitch_arr = 10.0 * np.sin(phase)
                roll_arr = np.zeros(n, dtype=np.float64)
                heading_arr = np.full(n, h_deg, dtype=np.float64)
            elif maneuver == "roll":
                pitch_arr = np.zeros(n, dtype=np.float64)
                roll_arr = 10.0 * np.sin(phase)
                heading_arr = np.full(n, h_deg, dtype=np.float64)
            else:  # yaw
                pitch_arr = np.zeros(n, dtype=np.float64)
                roll_arr = np.zeros(n, dtype=np.float64)
                heading_arr = h_deg + 5.0 * np.sin(phase)

            # Compute body-frame B components via rotation for each row
            bx_arr = np.empty(n, dtype=np.float64)
            by_arr = np.empty(n, dtype=np.float64)
            bz_arr = np.empty(n, dtype=np.float64)
            for i in range(n):
                r_mat = _rotation_matrix(
                    float(heading_arr[i]), float(pitch_arr[i]), float(roll_arr[i])
                )
                b_body: npt.NDArray[np.float64] = r_mat @ b_earth
                bx_arr[i] = b_body[0]
                by_arr[i] = b_body[1]
                bz_arr[i] = b_body[2]

            time_arr = np.arange(row_time_offset, row_time_offset + n, dtype=np.float64)

            block_df = pl.DataFrame(
                {
                    COL_TIME: time_arr,
                    COL_LAT: np.full(n, 45.0, dtype=np.float64),
                    COL_LON: np.full(n, -75.0, dtype=np.float64),
                    COL_ALT: np.full(n, 0.3, dtype=np.float64),
                    COL_BTOTAL: np.full(n, b_total_true, dtype=np.float64),
                    COL_BX: bx_arr,
                    COL_BY: by_arr,
                    COL_BZ: bz_arr,
                }
            )

            # Build feature matrix and inject interference
            config = PipelineConfig(model_terms=model_terms)
            a_mat = build_feature_matrix(block_df, config).to_numpy()
            interference: npt.NDArray[np.float64] = (
                a_mat @ c_true + rng.normal(0.0, noise_std, n)
            ).astype(np.float64)
            b_total_measured = np.full(n, b_total_true, dtype=np.float64) + interference

            segment_label = f"{maneuver}_{heading_name}"
            block_df = block_df.with_columns(
                pl.Series(COL_BTOTAL, b_total_measured, dtype=pl.Float64),
                pl.Series(COL_HEADING, heading_arr, dtype=pl.Float64),
                pl.Series(COL_PITCH, pitch_arr, dtype=pl.Float64),
                pl.Series(COL_ROLL, roll_arr, dtype=pl.Float64),
                pl.Series(COL_SEGMENT_LABEL, [segment_label] * n),
            )

            blocks.append(block_df)
            row_time_offset += n

    return pl.concat(blocks)
