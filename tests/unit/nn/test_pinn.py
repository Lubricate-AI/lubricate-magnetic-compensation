# tests/unit/nn/test_pinn.py
"""Unit tests for lmc.nn.pinn."""

from __future__ import annotations

from lmc.nn.pinn import PINNConfig


def test_pinn_config_defaults() -> None:
    cfg = PINNConfig()
    assert cfg.hidden_layer_sizes == (64, 64)
    assert cfg.activation == "relu"
    assert cfg.max_iter == 500
    assert cfg.n_estimators == 20
    assert cfg.random_state == 42
    assert cfg.physics_lambda == 1e-3
    assert cfg.tl_model_terms == "c"
    assert cfg.nn_feature_terms == "b"


def test_pinn_config_custom() -> None:
    cfg = PINNConfig(hidden_layer_sizes=(32,), n_estimators=5, physics_lambda=0.1)
    assert cfg.hidden_layer_sizes == (32,)
    assert cfg.n_estimators == 5
    assert cfg.physics_lambda == 0.1
