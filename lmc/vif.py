"""Variance Inflation Factor (VIF) computation for multicollinearity diagnostics."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_vif(A: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the Variance Inflation Factor for each column of a design matrix.

    VIF_i = 1 / (1 - R²_i), where R²_i is the coefficient of determination
    from regressing column *i* on all other columns.  VIF > 10 is commonly
    used as a threshold for problematic multicollinearity.

    Parameters
    ----------
    A:
        Design matrix of shape ``(n_samples, n_terms)``.  Must have at least
        2 columns.

    Returns
    -------
    npt.NDArray[np.float64]
        VIF values, shape ``(n_terms,)``.  Returns ``inf`` for columns that
        are perfectly predicted by the others (R² == 1).

    Raises
    ------
    ValueError
        If ``A`` has fewer than 2 columns.
    """
    n_terms = A.shape[1]
    if n_terms < 2:
        raise ValueError(
            f"compute_vif requires at least 2 columns; got {n_terms}."
        )

    vif = np.empty(n_terms, dtype=np.float64)

    for i in range(n_terms):
        y = A[:, i]
        X = np.delete(A, i, axis=1)

        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coef

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))

        if ss_tot == 0.0:
            # Constant column — undefined VIF, treat as inf.
            vif[i] = float("inf")
        else:
            # Clamp R² to [0, 1] to guard against tiny numerical overruns.
            r2 = max(0.0, min(1.0, 1.0 - ss_res / ss_tot))
            vif[i] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")

    return vif
