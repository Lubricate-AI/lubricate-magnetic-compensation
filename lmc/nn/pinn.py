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
    """

    hidden_layer_sizes: tuple[int, ...] = field(default=(64, 64))
    activation: str = "relu"
    max_iter: int = 500
    n_estimators: int = 20
    random_state: int = 42
    physics_lambda: float = 1e-3
    tl_model_terms: Literal["a", "b", "c", "d"] = "c"
    nn_feature_terms: Literal["a", "b", "c", "d"] = "b"
