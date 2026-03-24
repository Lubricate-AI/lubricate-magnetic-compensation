"""Supervised MLP model for aeromagnetic interference compensation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from lmc.columns import (
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
    COL_DELTA_B,
    COL_TMI_COMPENSATED,
)
from lmc.segmentation import Segment


@dataclass(frozen=True)
class NNConfig:
    """Hyperparameters for the supervised MLP compensation model.

    Attributes
    ----------
    hidden_layer_sizes:
        Tuple of integers specifying neurons per hidden layer.
        E.g. ``(64, 64)`` = two hidden layers with 64 neurons each.
    activation:
        Activation function: ``'relu'``, ``'tanh'``, or ``'logistic'``.
    max_iter:
        Maximum training iterations (epochs) for each estimator.
    n_estimators:
        Number of bootstrap-resampled models to train for the ensemble.
        More estimators yield tighter uncertainty estimates but take longer.
    random_state:
        Base random seed for reproducibility.  Each estimator uses
        ``random_state + i`` to ensure diverse bootstrap draws.
    """

    hidden_layer_sizes: tuple[int, ...] = field(default=(64, 64))
    activation: str = "relu"
    max_iter: int = 500
    n_estimators: int = 20
    random_state: int = 42


@dataclass
class NNCalibrationResult:
    """Result of a supervised NN calibration.

    Attributes
    ----------
    estimators:
        List of fitted ``MLPRegressor`` instances (one per bootstrap resample).
    input_scaler:
        ``StandardScaler`` fitted on the full training feature matrix.
        Applied before every prediction.
    residuals:
        Per-sample residuals ``mean_prediction - delta_B`` on the training
        segments, shape ``(n_samples,)``.
    n_features:
        Number of input features (3 for raw B_x/B_y/B_z).
    n_estimators:
        Number of bootstrap estimators actually trained.
    """

    estimators: list[MLPRegressor]
    input_scaler: StandardScaler
    residuals: npt.NDArray[np.float64]
    n_features: int
    n_estimators: int


def _extract_nn_features(df: pl.DataFrame) -> npt.NDArray[np.float64]:
    """Extract B_x, B_y, B_z from a polars DataFrame as a (n, 3) float64 array.

    Parameters
    ----------
    df:
        Input DataFrame that must contain ``COL_BX``, ``COL_BY``, ``COL_BZ``.

    Returns
    -------
    numpy array of shape ``(n_samples, 3)``

    Raises
    ------
    ValueError
        If any of the three columns are absent.
    """
    missing = [c for c in (COL_BX, COL_BY, COL_BZ) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns required for NN features are missing: {missing}. "
            f"Available: {df.columns}"
        )
    bx = np.asarray(df[COL_BX].to_numpy(), dtype=np.float64)
    by = np.asarray(df[COL_BY].to_numpy(), dtype=np.float64)
    bz = np.asarray(df[COL_BZ].to_numpy(), dtype=np.float64)
    return np.column_stack([bx, by, bz])


def calibrate_nn(
    df: pl.DataFrame,
    segments: list[Segment],
    config: NNConfig,
) -> NNCalibrationResult:
    """Train a bootstrap-ensemble MLP on calibration segments.

    Each of ``config.n_estimators`` MLPRegressors is trained on a bootstrap
    resample of the stacked calibration data.  A shared ``StandardScaler``
    fitted on the full (non-resampled) data is used for all estimators.

    Parameters
    ----------
    df:
        Full calibration DataFrame.  Must contain ``COL_BX``, ``COL_BY``,
        ``COL_BZ``, and ``COL_DELTA_B``.
    segments:
        Non-empty list of labeled flight segments.
    config:
        NN hyperparameter configuration.

    Returns
    -------
    NNCalibrationResult

    Raises
    ------
    ValueError
        If ``segments`` is empty, ``COL_DELTA_B`` is absent, any segment
        bounds are invalid, or all segments produce zero rows.
    """
    if not segments:
        raise ValueError("segments must be non-empty; cannot calibrate with no data.")

    if config.n_estimators < 1:
        raise ValueError(
            f"n_estimators must be >= 1, got {config.n_estimators}."
        )

    if COL_DELTA_B not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_B}' is required for calibration but was not found. "
            f"Available columns: {df.columns}"
        )

    x_blocks: list[npt.NDArray[np.float64]] = []
    y_blocks: list[npt.NDArray[np.float64]] = []

    for seg in segments:
        if not (0 <= seg.start_idx < seg.end_idx <= len(df)):
            raise ValueError(
                f"Segment {seg!r} has invalid bounds for a DataFrame "
                f"of length {len(df)}."
            )
        seg_df = df.slice(seg.start_idx, seg.end_idx - seg.start_idx)
        x_blocks.append(_extract_nn_features(seg_df))
        y_blocks.append(np.asarray(seg_df[COL_DELTA_B].to_numpy(), dtype=np.float64))

    X: npt.NDArray[np.float64] = np.vstack(x_blocks)
    y: npt.NDArray[np.float64] = np.concatenate(y_blocks)

    if X.shape[0] == 0:
        raise ValueError("All segments produced empty slices; cannot calibrate.")

    scaler = StandardScaler()
    X_scaled: npt.NDArray[np.float64] = scaler.fit_transform(X)  # type: ignore[assignment]

    rng = np.random.default_rng(config.random_state)
    n: int = X_scaled.shape[0]  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    estimators: list[MLPRegressor] = []

    for i in range(config.n_estimators):
        idx = rng.integers(0, n, size=n)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
        mlp = MLPRegressor(
            hidden_layer_sizes=config.hidden_layer_sizes,
            activation=config.activation,
            max_iter=config.max_iter,
            random_state=config.random_state + i,
        )
        mlp.fit(X_scaled[idx], y[idx])  # pyright: ignore[reportUnknownMemberType]
        estimators.append(mlp)

    # Compute training residuals using the ensemble mean prediction.
    _member_preds = [m.predict(X_scaled) for m in estimators]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    preds = np.column_stack(_member_preds)  # pyright: ignore[reportUnknownArgumentType]  # (n, k)
    mean_pred = preds.mean(axis=1)
    residuals = np.asarray(mean_pred - y, dtype=np.float64)

    return NNCalibrationResult(
        estimators=estimators,
        input_scaler=scaler,
        residuals=residuals,
        n_features=X.shape[1],
        n_estimators=config.n_estimators,
    )


def predict_nn(
    df: pl.DataFrame,
    result: NNCalibrationResult,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predict interference correction with uncertainty from a trained NN ensemble.

    Parameters
    ----------
    df:
        Input DataFrame containing ``COL_BX``, ``COL_BY``, ``COL_BZ``.
    result:
        Fitted ``NNCalibrationResult`` from ``calibrate_nn()``.

    Returns
    -------
    mean_prediction:
        Ensemble-mean predicted interference, shape ``(n_samples,)``.
    std_prediction:
        Ensemble standard deviation (spread-based uncertainty proxy),
        shape ``(n_samples,)``. Intervals like ``mean ± 1.96 * std``
        can be used as heuristic uncertainty bands, but they are not
        guaranteed 95 % confidence intervals.
    """
    X = _extract_nn_features(df)
    X_scaled: npt.NDArray[np.float64] = result.input_scaler.transform(X)  # type: ignore[assignment]
    preds = np.column_stack(  # pyright: ignore[reportUnknownArgumentType]
        [m.predict(X_scaled) for m in result.estimators]  # pyright: ignore[reportUnknownMemberType]
    )  # shape (n_samples, n_estimators)
    mean_pred = np.asarray(preds.mean(axis=1), dtype=np.float64)
    std_pred = np.asarray(preds.std(axis=1), dtype=np.float64)
    return mean_pred, std_pred


def compensate_nn(
    df: pl.DataFrame,
    result: NNCalibrationResult,
) -> pl.DataFrame:
    """Subtract NN-predicted interference from survey TMI.

    Parameters
    ----------
    df:
        Survey DataFrame containing ``COL_BTOTAL``, ``COL_BX``,
        ``COL_BY``, ``COL_BZ``.
    result:
        Fitted ``NNCalibrationResult`` from ``calibrate_nn()``.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with one additional column ``COL_TMI_COMPENSATED``
        containing ``B_total - mean_ensemble_prediction``.
    """
    if COL_BTOTAL not in df.columns:
        raise ValueError(
            f"Column '{COL_BTOTAL}' is required for compensation but was not found. "
            f"Available columns: {df.columns}"
        )
    mean_pred, _ = predict_nn(df, result)
    b_total = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64)
    tmi_comp = b_total - mean_pred
    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
