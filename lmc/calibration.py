"""Tolles-Lawson coefficient estimation via least-squares regression."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.columns import COL_DELTA_B
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import Segment


@dataclass(frozen=True)
class CalibrationResult:
    """Result of a Tolles-Lawson calibration regression.

    Attributes
    ----------
    coefficients:
        Fitted model coefficients, shape ``(n_terms,)``.
    residuals:
        Per-sample residuals ``A @ coefficients - δB``, shape ``(n_samples,)``.
    condition_number:
        Condition number of the stacked (un-augmented) A-matrix.
    n_terms:
        Number of model coefficients (3, 9, or 18 depending on ``model_terms``).
    """

    coefficients: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    condition_number: float
    n_terms: int


def calibrate(
    df: pl.DataFrame,
    segments: list[Segment],
    config: PipelineConfig,
) -> CalibrationResult:
    """Fit the Tolles-Lawson linear model to labeled calibration segments.

    Parameters
    ----------
    df:
        Full calibration DataFrame containing all required columns including
        ``COL_DELTA_B``.
    segments:
        Non-empty list of labeled flight segments identifying which rows to use.
    config:
        Pipeline configuration controlling term set, ridge regression, etc.

    Returns
    -------
    CalibrationResult
        Fitted coefficients, per-sample residuals, condition number, and term count.

    Raises
    ------
    ValueError
        If ``segments`` is empty or if ``COL_DELTA_B`` is absent from ``df``.
    """
    if not segments:
        raise ValueError("segments must be non-empty; cannot calibrate with no data.")

    if COL_DELTA_B not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_B}' is required for calibration but was not found "
            f"in the DataFrame. Available columns: {df.columns}"
        )

    a_blocks: list[npt.NDArray[np.float64]] = []
    db_blocks: list[npt.NDArray[np.float64]] = []

    for seg in segments:
        segment_df = df.slice(seg.start_idx, seg.end_idx - seg.start_idx)
        a_seg = build_feature_matrix(segment_df, config).to_numpy()
        db_seg = segment_df[COL_DELTA_B].to_numpy().astype(np.float64)
        a_blocks.append(a_seg)
        db_blocks.append(db_seg)

    A: npt.NDArray[np.float64] = np.vstack(a_blocks)
    dB: npt.NDArray[np.float64] = np.concatenate(db_blocks)
    n_terms = A.shape[1]

    condition_number = float(np.linalg.cond(A))

    if condition_number > config.condition_number_threshold:
        warnings.warn(
            f"Condition number {condition_number:.3e} exceeds threshold "
            f"{config.condition_number_threshold:.3e}. The system may be "
            "ill-conditioned; consider using ridge regression or more"
            " diverse segments.",
            stacklevel=2,
        )

    if config.use_ridge:
        sqrt_alpha = np.sqrt(config.ridge_alpha)
        A_aug: npt.NDArray[np.float64] = np.vstack([A, sqrt_alpha * np.eye(n_terms)])
        dB_aug: npt.NDArray[np.float64] = np.concatenate([dB, np.zeros(n_terms)])
        coefficients, _, _, _ = np.linalg.lstsq(A_aug, dB_aug, rcond=None)
    else:
        coefficients, _, _, _ = np.linalg.lstsq(A, dB, rcond=None)

    coefficients = np.asarray(coefficients, dtype=np.float64)
    residuals = np.asarray(A @ coefficients - dB, dtype=np.float64)

    return CalibrationResult(
        coefficients=coefficients,
        residuals=residuals,
        condition_number=condition_number,
        n_terms=n_terms,
    )
