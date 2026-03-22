"""Recursive Least-Squares (RLS) for online Tolles-Lawson coefficient updating."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from lmc.calibration import CalibrationResult


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
