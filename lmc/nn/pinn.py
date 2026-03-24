"""Physics-informed neural network (PINN) for aeromagnetic compensation.

Architecture: B_predicted = TL(A) + NN(TL_features), where
- TL(A) is the Tolles-Lawson linear model (physics backbone)
- NN(TL_features) is a residual corrector trained on TL residuals
- TL_features (direction cosines and products) form the NN input space

The physics constraint is enforced via:
1. NN input space = TL feature space (physically meaningful, not raw B)
2. L2 regularization (physics_lambda) penalises large NN corrections
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from lmc.calibration import CalibrationResult, calibrate
from lmc.columns import COL_BTOTAL, COL_DELTA_B, COL_TMI_COMPENSATED
from lmc.config import PipelineConfig
from lmc.features import build_feature_matrix
from lmc.segmentation import Segment


@dataclass(frozen=True)
class PINNConfig:
    """Hyperparameters for the PINN compensation model.

    Attributes
    ----------
    hidden_layer_sizes:
        Tuple of integers specifying neurons per hidden layer.
    activation:
        Activation function: ``'relu'``, ``'tanh'``, or ``'logistic'``.
    max_iter:
        Maximum training iterations for each estimator.
    n_estimators:
        Number of bootstrap-resampled models for uncertainty quantification.
    random_state:
        Base random seed. Each estimator uses ``random_state + i``.
    physics_lambda:
        L2 regularization strength for the residual NN.  Larger values keep
        the NN corrections small, forcing the TL model to carry more weight.
        Maps to ``MLPRegressor(alpha=physics_lambda)``.
    tl_model_terms:
        Tolles-Lawson term set for the physics backbone calibration.
        ``'c'`` (18 terms) is recommended for full eddy-current coverage.
    nn_feature_terms:
        TL term set to use as NN input features.  ``'b'`` (9 terms) is
        recommended: captures permanent + induced physics without requiring
        time derivatives, which simplifies prediction.
    tl_pipeline_config:
        Full ``PipelineConfig`` to use for the Tolles-Lawson backbone
        calibration.  When provided, its ``model_terms`` overrides
        ``tl_model_terms``.  Use this to opt into regularisation
        (``use_ridge``, ``use_lasso``), cross-validation (``use_cv``),
        IMU-rate derivatives (``use_imu_rates``), or other TL options.
        When ``None`` (default), a minimal config is constructed from
        ``tl_model_terms`` with all other options at their defaults.
    """

    hidden_layer_sizes: tuple[int, ...] = field(default=(64, 64))
    activation: str = "relu"
    max_iter: int = 500
    n_estimators: int = 20
    random_state: int = 42
    physics_lambda: float = 1e-3
    tl_model_terms: Literal["a", "b", "c", "d"] = "c"
    nn_feature_terms: Literal["a", "b", "c", "d"] = "b"
    tl_pipeline_config: PipelineConfig | None = None


@dataclass
class PINNCalibrationResult:
    """Result of a PINN calibration.

    Attributes
    ----------
    tl_result:
        Fitted ``CalibrationResult`` from the Tolles-Lawson backbone.
    estimators:
        Bootstrap-ensemble MLP regressors trained on TL residuals.
    input_scaler:
        ``StandardScaler`` fitted on the full NN training feature matrix.
    tl_residuals:
        Per-sample TL residuals ``TL_pred - delta_B`` on the training
        segments, shape ``(n_samples,)``.
    pinn_residuals:
        Final combined residuals ``(TL_pred + NN_mean_pred) - delta_B``,
        shape ``(n_samples,)``.
    n_nn_features:
        Number of NN input features (columns in the TL feature matrix used
        for NN inputs, e.g. 9 for ``nn_feature_terms='b'``).
    n_estimators:
        Number of bootstrap estimators actually trained.
    tl_model_terms:
        Tolles-Lawson term set used for the physics backbone during calibration.
        Stored so ``predict_pinn`` and ``compensate_pinn`` can rebuild the TL
        feature matrix without requiring the caller to re-supply a config.
    nn_feature_terms:
        TL term set used as NN input features during calibration.
    """

    tl_result: CalibrationResult
    estimators: list[MLPRegressor]
    input_scaler: StandardScaler
    tl_residuals: npt.NDArray[np.float64]
    pinn_residuals: npt.NDArray[np.float64]
    n_nn_features: int
    n_estimators: int
    tl_model_terms: Literal["a", "b", "c", "d"]
    nn_feature_terms: Literal["a", "b", "c", "d"]


def _extract_pinn_features(
    df: pl.DataFrame,
    nn_feature_terms: Literal["a", "b", "c", "d"],
) -> npt.NDArray[np.float64]:
    """Extract TL feature matrix for NN input.

    Uses ``build_feature_matrix`` with a minimal PipelineConfig to produce
    the direction-cosine feature space used as NN inputs.

    Parameters
    ----------
    df:
        Input DataFrame containing at minimum the columns required by
        ``build_feature_matrix`` (see ``lmc.validation.REQUIRED_COLUMNS``).
    nn_feature_terms:
        TL term set for the NN feature matrix.  ``'b'`` is recommended
        (9 features): permanent + induced terms, no time derivatives.

    Returns
    -------
    numpy array of shape ``(n_samples, n_features)`` with dtype float64.
    """
    cfg = PipelineConfig(model_terms=nn_feature_terms)
    feature_df = build_feature_matrix(df, cfg)
    return np.asarray(feature_df.to_numpy(), dtype=np.float64)


def calibrate_pinn(
    df: pl.DataFrame,
    segments: list[Segment],
    config: PINNConfig,
) -> PINNCalibrationResult:
    """Calibrate PINN: TL backbone + bootstrap-ensemble residual NN.

    Two-phase approach:
    1. Fit TL coefficients on calibration segments using ``lmc.calibration.calibrate``.
    2. Train a bootstrap-ensemble MLP on TL residuals, with NN inputs drawn
       from the TL feature space (``config.nn_feature_terms``).

    The physics constraint is enforced by:
    - NN input space = TL feature matrix (direction cosines and products)
    - L2 regularization ``config.physics_lambda`` penalises large NN corrections

    Parameters
    ----------
    df:
        Full calibration DataFrame.  Must contain all required magnetometer
        columns plus ``COL_DELTA_B``.
    segments:
        Non-empty list of labeled flight segments.
    config:
        PINN hyperparameter configuration.

    Returns
    -------
    PINNCalibrationResult

    Raises
    ------
    ValueError
        If ``segments`` is empty, ``n_estimators < 1``, ``COL_DELTA_B`` is
        absent, any segment bounds are invalid, or all segments produce
        zero rows.
    """
    if not segments:
        raise ValueError("segments must be non-empty; cannot calibrate with no data.")

    if config.n_estimators < 1:
        raise ValueError(f"n_estimators must be >= 1, got {config.n_estimators}.")

    if COL_DELTA_B not in df.columns:
        raise ValueError(
            f"Column '{COL_DELTA_B}' is required for calibration but was not found. "
            f"Available columns: {df.columns}"
        )

    # --- Phase 1: Tolles-Lawson backbone calibration ---
    tl_pipeline_cfg = (
        config.tl_pipeline_config
        if config.tl_pipeline_config is not None
        else PipelineConfig(model_terms=config.tl_model_terms)
    )
    tl_result = calibrate(df, segments, tl_pipeline_cfg)

    # Build stacked segment data for NN training
    x_blocks: list[npt.NDArray[np.float64]] = []
    y_blocks: list[npt.NDArray[np.float64]] = []

    for seg in segments:
        if not (0 <= seg.start_idx < seg.end_idx <= len(df)):
            raise ValueError(
                f"Segment {seg!r} has invalid bounds for a DataFrame "
                f"of length {len(df)}."
            )
        seg_df = df.slice(seg.start_idx, seg.end_idx - seg.start_idx)
        x_blocks.append(_extract_pinn_features(seg_df, config.nn_feature_terms))
        y_blocks.append(np.asarray(seg_df[COL_DELTA_B].to_numpy(), dtype=np.float64))

    X: npt.NDArray[np.float64] = np.vstack(x_blocks)
    delta_b_all: npt.NDArray[np.float64] = np.concatenate(y_blocks)

    if X.shape[0] == 0:
        raise ValueError("All segments produced empty slices; cannot calibrate.")

    # tl_result.residuals = A @ coef - delta_B
    # NN target: learn the negative of TL residuals (the TL shortfall)
    tl_residuals: npt.NDArray[np.float64] = tl_result.residuals
    nn_targets: npt.NDArray[np.float64] = -tl_residuals

    # --- Phase 2: Bootstrap-ensemble MLP on TL residuals ---
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
            alpha=config.physics_lambda,  # L2 physics constraint
            random_state=config.random_state + i,
        )
        mlp.fit(X_scaled[idx], nn_targets[idx])  # pyright: ignore[reportUnknownMemberType]
        estimators.append(mlp)

    # Compute combined PINN residuals on training data
    _member_preds = [m.predict(X_scaled) for m in estimators]  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    nn_mean_pred = np.column_stack(_member_preds).mean(axis=1)  # pyright: ignore[reportUnknownArgumentType]
    # tl_pred = delta_b_all + tl_residuals  (since tl_residuals = A@coef - delta_B)
    tl_pred = delta_b_all + tl_residuals
    pinn_pred = tl_pred + nn_mean_pred
    pinn_residuals = np.asarray(pinn_pred - delta_b_all, dtype=np.float64)

    return PINNCalibrationResult(
        tl_result=tl_result,
        estimators=estimators,
        input_scaler=scaler,
        tl_residuals=tl_residuals,
        pinn_residuals=pinn_residuals,
        n_nn_features=X.shape[1],
        n_estimators=config.n_estimators,
        tl_model_terms=tl_pipeline_cfg.model_terms,
        nn_feature_terms=config.nn_feature_terms,
    )


def predict_pinn(
    df: pl.DataFrame,
    result: PINNCalibrationResult,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Predict total interference correction with uncertainty.

    Combines TL backbone prediction with ensemble NN residual correction.
    The term settings are read from ``result`` (stored at calibration time),
    so callers do not need to re-supply a config.

    Parameters
    ----------
    df:
        Input DataFrame containing all required magnetometer columns.
    result:
        Fitted ``PINNCalibrationResult`` from ``calibrate_pinn()``.

    Returns
    -------
    mean_prediction:
        Combined ``TL_pred + NN_mean_pred`` interference estimate,
        shape ``(n_samples,)``.
    std_prediction:
        Ensemble standard deviation of the NN residual component,
        shape ``(n_samples,)``. Use as a proxy for prediction uncertainty.
    """
    # TL prediction: A @ coefficients
    tl_cfg = PipelineConfig(model_terms=result.tl_model_terms)
    feature_matrix = build_feature_matrix(df, tl_cfg)
    A = np.asarray(feature_matrix.to_numpy(), dtype=np.float64)
    if A.shape[1] != result.tl_result.n_terms:
        raise ValueError(
            f"Tolles-Lawson feature matrix has {A.shape[1]} columns, but the "
            f"calibrated TL model expects {result.tl_result.n_terms} terms. "
            "This is an internal consistency error — the stored "
            "'tl_model_terms' does not match the calibrated coefficients."
        )
    tl_pred = np.asarray(A @ result.tl_result.coefficients, dtype=np.float64)

    # NN residual prediction
    X = _extract_pinn_features(df, result.nn_feature_terms)
    X_scaled: npt.NDArray[np.float64] = result.input_scaler.transform(X)  # type: ignore[assignment]
    nn_preds = np.column_stack(
        [m.predict(X_scaled) for m in result.estimators]  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
    )  # shape (n_samples, n_estimators)
    nn_mean = np.asarray(nn_preds.mean(axis=1), dtype=np.float64)
    nn_std = np.asarray(nn_preds.std(axis=1), dtype=np.float64)

    return tl_pred + nn_mean, nn_std


def compensate_pinn(
    df: pl.DataFrame,
    result: PINNCalibrationResult,
) -> pl.DataFrame:
    """Subtract PINN-predicted interference from survey TMI.

    Parameters
    ----------
    df:
        Survey DataFrame containing ``COL_BTOTAL`` and all columns
        required by ``predict_pinn``.
    result:
        Fitted ``PINNCalibrationResult`` from ``calibrate_pinn()``.

    Returns
    -------
    pl.DataFrame
        Input DataFrame with one additional column ``COL_TMI_COMPENSATED``
        containing ``B_total - (TL_pred + NN_mean_pred)``.

    Raises
    ------
    ValueError
        If ``COL_BTOTAL`` is absent from ``df``.
    """
    if COL_BTOTAL not in df.columns:
        raise ValueError(
            f"Column '{COL_BTOTAL}' is required for compensation but was not found. "
            f"Available columns: {df.columns}"
        )
    mean_pred, _ = predict_pinn(df, result)
    b_total = np.asarray(df[COL_BTOTAL].to_numpy(), dtype=np.float64)
    tmi_comp = b_total - mean_pred
    return df.with_columns(pl.Series(COL_TMI_COMPENSATED, tmi_comp, dtype=pl.Float64))
