"""Tolles-Lawson coefficient estimation via least-squares regression."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
)

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
        Rows correspond to the concatenated segment rows in the order supplied to
        ``calibrate()``, **not** to all rows of the input DataFrame.
    condition_number:
        Condition number of the stacked (un-augmented) A-matrix.
    n_terms:
        Number of model coefficients (3, 9, or 18 depending on ``model_terms``).
    selected_alpha:
        Regularisation strength used. ``None`` for OLS.
    effective_dof:
        Effective degrees of freedom consumed by the model.
        For ridge: ``sum(sigma_i^2 / (sigma_i^2 + alpha))``.
        For LASSO/ElasticNet: number of non-zero coefficients.
        ``None`` for OLS.
    """

    coefficients: npt.NDArray[np.float64]
    residuals: npt.NDArray[np.float64]
    condition_number: float
    n_terms: int
    selected_alpha: float | None = field(default=None)
    effective_dof: float | None = field(default=None)


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
        If ``segments`` is empty, if ``COL_DELTA_B`` is absent from ``df``,
        if any segment has ``start_idx >= end_idx`` or indices out of range for ``df``,
        or if all segments produce empty slices.
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
        if not (0 <= seg.start_idx < seg.end_idx <= len(df)):
            raise ValueError(
                f"Segment {seg!r} has invalid bounds for a DataFrame "
                f"of length {len(df)}. "
                "Required: 0 <= start_idx < end_idx <= len(df)."
            )
        segment_df = df.slice(seg.start_idx, seg.end_idx - seg.start_idx)
        a_seg = build_feature_matrix(segment_df, config).to_numpy()
        db_seg = segment_df[COL_DELTA_B].to_numpy().astype(np.float64)
        a_blocks.append(a_seg)
        db_blocks.append(db_seg)

    A: npt.NDArray[np.float64] = np.vstack(a_blocks)
    dB: npt.NDArray[np.float64] = np.concatenate(db_blocks)

    if A.shape[0] == 0:
        raise ValueError(
            "All segments produced empty slices; cannot calibrate with zero rows."
        )

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

    selected_alpha: float | None = None
    effective_dof: float | None = None

    if config.use_ridge:
        sqrt_alpha = np.sqrt(config.ridge_alpha)
        A_aug: npt.NDArray[np.float64] = np.vstack([A, sqrt_alpha * np.eye(n_terms)])
        dB_aug: npt.NDArray[np.float64] = np.concatenate([dB, np.zeros(n_terms)])
        coefficients, _, _, _ = np.linalg.lstsq(A_aug, dB_aug, rcond=None)
        selected_alpha = config.ridge_alpha
        sigma = np.linalg.svd(A, compute_uv=False)
        effective_dof = float(np.sum(sigma**2 / (sigma**2 + config.ridge_alpha)))
    elif config.use_lasso:
        model = Lasso(alpha=config.lasso_alpha, fit_intercept=False, max_iter=10_000)
        model.fit(A, dB)  # pyright: ignore[reportUnknownMemberType]
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        selected_alpha = config.lasso_alpha
        effective_dof = float(np.sum(np.abs(coefficients) > 0.0))
    elif config.use_elastic_net:
        model = ElasticNet(
            alpha=config.elastic_net_alpha,
            l1_ratio=config.elastic_net_l1_ratio,
            fit_intercept=False,
            max_iter=10_000,
        )
        model.fit(A, dB)  # pyright: ignore[reportUnknownMemberType]
        coefficients = np.asarray(model.coef_, dtype=np.float64)
        selected_alpha = config.elastic_net_alpha
        effective_dof = float(np.sum(np.abs(coefficients) > 0.0))
    else:
        coefficients, _, _, _ = np.linalg.lstsq(A, dB, rcond=None)

    coefficients = np.asarray(coefficients, dtype=np.float64)
    residuals = np.asarray(A @ coefficients - dB, dtype=np.float64)

    return CalibrationResult(
        coefficients=coefficients,
        residuals=residuals,
        condition_number=condition_number,
        n_terms=n_terms,
        selected_alpha=selected_alpha,
        effective_dof=effective_dof,
    )
