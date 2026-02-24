"""Unit tests for _validate_coef_dict in lmc.cli.commands."""

from __future__ import annotations

import math

import pytest

from lmc.cli.commands import _validate_coef_dict  # pyright: ignore[reportPrivateUsage]

_VALID: dict[str, object] = {
    "model_terms": "c",
    "coefficients": [float(i) for i in range(1, 19)],
    "n_terms": 18,
    "condition_number": 42.5,
}


def test_valid_passes() -> None:
    _validate_coef_dict(_VALID)


def test_non_dict_root() -> None:
    with pytest.raises(ValueError, match="JSON object at the root"):
        _validate_coef_dict([1, 2, 3])


def test_non_dict_root_string() -> None:
    with pytest.raises(ValueError, match="JSON object at the root"):
        _validate_coef_dict("not a dict")


def test_missing_all_keys() -> None:
    with pytest.raises(ValueError, match="Missing required keys"):
        _validate_coef_dict({})


def test_partial_keys_reports_all_errors() -> None:
    """Present keys with wrong values and absent keys are all reported."""
    with pytest.raises(ValueError) as exc_info:
        _validate_coef_dict({"model_terms": "x", "coefficients": []})
    msg = str(exc_info.value)
    assert "Missing required keys" in msg
    assert "model_terms" in msg
    assert "coefficients" in msg


def test_invalid_model_terms_type() -> None:
    with pytest.raises(ValueError, match="'model_terms' must be a string"):
        _validate_coef_dict({**_VALID, "model_terms": 1})


def test_invalid_model_terms_unhashable() -> None:
    with pytest.raises(ValueError, match="'model_terms' must be a string"):
        _validate_coef_dict({**_VALID, "model_terms": ["c"]})


def test_invalid_model_terms_value() -> None:
    with pytest.raises(ValueError, match="'model_terms' must be one of"):
        _validate_coef_dict({**_VALID, "model_terms": "x"})


def test_empty_coefficients() -> None:
    with pytest.raises(ValueError, match="'coefficients' must be a non-empty list"):
        _validate_coef_dict({**_VALID, "coefficients": []})


def test_coefficients_not_a_list() -> None:
    with pytest.raises(ValueError, match="'coefficients' must be a non-empty list"):
        _validate_coef_dict({**_VALID, "coefficients": "bad"})


def test_non_numeric_coefficients() -> None:
    data = {**_VALID, "coefficients": ["bad", "value"], "n_terms": 2}
    with pytest.raises(ValueError, match="'coefficients' must contain only numbers"):
        _validate_coef_dict(data)


def test_n_terms_not_integer() -> None:
    with pytest.raises(ValueError, match="'n_terms' must be an integer"):
        _validate_coef_dict({**_VALID, "n_terms": "18"})


def test_n_terms_bool_rejected() -> None:
    with pytest.raises(ValueError, match="'n_terms' must be an integer"):
        _validate_coef_dict({**_VALID, "n_terms": True})


def test_n_terms_count_mismatch() -> None:
    with pytest.raises(ValueError, match="'n_terms' is 5 but 'coefficients' has 18"):
        _validate_coef_dict({**_VALID, "n_terms": 5})


def test_n_terms_model_mismatch() -> None:
    data = {**_VALID, "model_terms": "a", "n_terms": 9, "coefficients": [1.0] * 9}
    with pytest.raises(ValueError, match="'n_terms' for model_terms="):
        _validate_coef_dict(data)


def test_condition_number_invalid_type() -> None:
    with pytest.raises(ValueError, match="'condition_number' must be a finite number"):
        _validate_coef_dict({**_VALID, "condition_number": "bad"})


def test_condition_number_bool_rejected() -> None:
    with pytest.raises(ValueError, match="'condition_number' must be a finite number"):
        _validate_coef_dict({**_VALID, "condition_number": True})


def test_condition_number_inf() -> None:
    with pytest.raises(ValueError, match="'condition_number' must be finite"):
        _validate_coef_dict({**_VALID, "condition_number": math.inf})


def test_condition_number_nan() -> None:
    with pytest.raises(ValueError, match="'condition_number' must be finite"):
        _validate_coef_dict({**_VALID, "condition_number": math.nan})
