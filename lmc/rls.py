"""Recursive Least-Squares (RLS) for online Tolles-Lawson coefficient updating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from lmc.calibration import CalibrationResult
from lmc.columns import COL_DELTA_B
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix


@dataclass
class RLSState:
    """Mutable state for Recursive Least-Squares online coefficient updating.

    Attributes
    ----------
    coefficients:
        Current coefficient estimate, shape ``(n_terms,)``.
    covariance:
        Current covariance matrix, shape ``(n_terms, n_terms)``.
        Diagonal entries approximate per-coefficient variance.
        Use ``np.diag(state.covariance)`` to get per-coefficient std devs.
    forgetting_factor:
        Exponential forgetting rate λ ∈ (0, 1]. λ=1 weights all history equally.
        λ<1 down-weights old samples; try λ=0.95–0.99 for slowly drifting systems.
    n_samples:
        Total number of individual samples processed since initialization.
    n_terms:
        Number of model coefficients (3, 9, 18, or 21 depending on model_terms).
    """

    coefficients: npt.NDArray[np.float64]
    covariance: npt.NDArray[np.float64]
    forgetting_factor: float
    n_samples: int
    n_terms: int


def initialize_rls(
    result: CalibrationResult,
    forgetting_factor: float = 1.0,
    *,
    initial_covariance_scale: float = 1.0,
) -> RLSState:
    """Create an RLSState from a batch CalibrationResult.

    Parameters
    ----------
    result:
        Batch calibration result to initialize from.  The ``coefficients``
        become the initial RLS estimate.
    forgetting_factor:
        Exponential forgetting rate λ ∈ (0, 1].  Defaults to 1.0 (no forgetting).
    initial_covariance_scale:
        Diagonal scale for the initial covariance P = scale × I.  Larger values
        allow the RLS to update aggressively from the first samples; smaller
        values anchor coefficients closer to the batch estimate.

    Returns
    -------
    RLSState
        Initialized state ready for incremental updates.

    Raises
    ------
    ValueError
        If ``forgetting_factor`` is not in (0, 1] or ``initial_covariance_scale``
        is not strictly positive.
    """
    if not (0.0 < forgetting_factor <= 1.0):
        raise ValueError(
            f"forgetting_factor must be in (0, 1]; got {forgetting_factor}."
        )
    if initial_covariance_scale <= 0.0:
        raise ValueError(
            f"initial_covariance_scale must be strictly positive; "
            f"got {initial_covariance_scale}."
        )

    n = result.n_terms
    return RLSState(
        coefficients=result.coefficients.copy(),
        covariance=initial_covariance_scale * np.eye(n, dtype=np.float64),
        forgetting_factor=forgetting_factor,
        n_samples=0,
        n_terms=n,
    )


def update_rls(
    state: RLSState,
    a: npt.NDArray[np.float64],
    y: float,
) -> RLSState:
    """Apply one RLS update step using the Kalman gain formulation.

    Parameters
    ----------
    state:
        Current RLS state.  Not mutated — a new state is returned.
    a:
        Feature vector for this sample, shape ``(n_terms,)``.
    y:
        Observed delta_B value for this sample (scalar).

    Returns
    -------
    RLSState
        Updated state with new coefficients and covariance.
    """
    lam = state.forgetting_factor
    theta = state.coefficients
    P = state.covariance

    # Innovation
    e = float(y) - float(a @ theta)

    # Kalman gain: k = P a / (λ + aᵀ P a)
    Pa = P @ a  # (p,)
    gain_denom = lam + float(a @ Pa)
    k = Pa / gain_denom  # (p,)

    # Coefficient update
    new_theta = theta + k * e

    # Covariance update: P′ = (P − k aᵀ P) / λ
    new_P = (P - np.outer(k, a @ P)) / lam
    # Symmetrize to prevent numerical drift
    new_P = (new_P + new_P.T) / 2.0

    return RLSState(
        coefficients=new_theta.astype(np.float64),
        covariance=new_P.astype(np.float64),
        forgetting_factor=lam,
        n_samples=state.n_samples + 1,
        n_terms=state.n_terms,
    )


def update_rls_batch(
    state: RLSState,
    df: pl.DataFrame,
    config: PipelineConfig,
) -> RLSState:
    """Apply RLS updates for every row in a DataFrame segment.

    Builds the feature matrix from ``df`` via ``build_feature_matrix``, then
    iterates row-by-row calling :func:`update_rls`.  Equivalent to calling
    ``update_rls`` in a loop but more convenient for segment-based workflows.

    Parameters
    ----------
    state:
        Current RLS state.  Not mutated — a new state is returned.
    df:
        DataFrame containing all required magnetometer columns plus
        ``COL_DELTA_B``.
    config:
        Pipeline configuration used to build the feature matrix.

    Returns
    -------
    RLSState
        Updated state after processing all rows in ``df``.

    Raises
    ------
    ValueError
        If ``COL_DELTA_B`` is absent from ``df``.
    """
    if COL_DELTA_B not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_B}' is required for RLS updates but was not "
            f"found in the DataFrame. Available columns: {df.columns}"
        )

    A: npt.NDArray[np.float64] = build_feature_matrix(df, config).to_numpy()
    dB: npt.NDArray[np.float64] = df[COL_DELTA_B].to_numpy().astype(np.float64)

    current = state
    for i in range(A.shape[0]):
        current = update_rls(current, A[i], dB[i])
    return current
