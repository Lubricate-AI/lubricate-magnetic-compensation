"""Unit tests for lmc.config."""

import datetime

import pytest
from pydantic import ValidationError

from lmc.config import PipelineConfig


def test_pipeline_config_default_construction() -> None:
    cfg = PipelineConfig()
    assert cfg.sample_rate_hz == 10.0
    assert cfg.bandpass_low_hz == 0.1
    assert cfg.bandpass_high_hz == 0.9
    assert cfg.model_terms == "c"
    assert cfg.use_ridge is False
    assert cfg.ridge_alpha == 1e-3
    assert cfg.earth_field_method == "igrf"
    assert cfg.igrf_date == datetime.date.today()
    assert cfg.segment_label_col is None
    assert cfg.reference_heading_deg is None
    assert cfg.heading_tolerance_deg == 45.0


def test_pipeline_config_is_immutable() -> None:
    cfg = PipelineConfig()
    with pytest.raises(ValidationError):
        cfg.sample_rate_hz = 20.0  # type: ignore[misc]


def test_pipeline_config_rejects_non_positive_sample_rate() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(sample_rate_hz=0.0)


def test_pipeline_config_rejects_bandpass_low_above_high() -> None:
    with pytest.raises(ValueError, match="bandpass_low_hz"):
        PipelineConfig(bandpass_low_hz=0.9, bandpass_high_hz=0.1)


def test_pipeline_config_rejects_equal_bandpass_frequencies() -> None:
    with pytest.raises(ValueError, match="bandpass_low_hz"):
        PipelineConfig(bandpass_low_hz=0.5, bandpass_high_hz=0.5)


def test_pipeline_config_rejects_high_above_nyquist() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        PipelineConfig(sample_rate_hz=2.0, bandpass_low_hz=0.1, bandpass_high_hz=1.0)


def test_pipeline_config_accepts_valid_model_terms() -> None:
    for term in ("a", "b", "c"):
        cfg = PipelineConfig(model_terms=term)  # type: ignore[arg-type]
        assert cfg.model_terms == term


def test_pipeline_config_rejects_invalid_model_terms() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(model_terms="d")  # type: ignore[arg-type]


def test_pipeline_config_rejects_negative_ridge_alpha() -> None:
    with pytest.raises(ValidationError):
        PipelineConfig(ridge_alpha=-0.1)


def test_pipeline_config_custom_values_roundtrip() -> None:
    cfg = PipelineConfig(
        sample_rate_hz=100.0,
        bandpass_low_hz=1.0,
        bandpass_high_hz=10.0,
        model_terms="a",
        use_ridge=True,
        ridge_alpha=0.5,
    )
    assert cfg.sample_rate_hz == 100.0
    assert cfg.use_ridge is True
    assert cfg.ridge_alpha == 0.5


def test_pipeline_config_igrf_requires_date() -> None:
    with pytest.raises(ValueError, match="igrf_date"):
        PipelineConfig(earth_field_method="igrf", igrf_date=None)


def test_pipeline_config_igrf_with_date_succeeds() -> None:
    cfg = PipelineConfig(
        earth_field_method="igrf",
        igrf_date=datetime.date(2024, 1, 1),
    )
    assert cfg.igrf_date == datetime.date(2024, 1, 1)


def test_pipeline_config_steady_mean_without_date_succeeds() -> None:
    cfg = PipelineConfig(earth_field_method="steady_mean")
    assert cfg.earth_field_method == "steady_mean"


def test_pipeline_config_reference_heading_accepts_negative() -> None:
    """Negative values are permissible; normalisation happens downstream."""
    cfg = PipelineConfig(reference_heading_deg=-90.0)
    assert cfg.reference_heading_deg == -90.0


def test_pipeline_config_reference_heading_accepts_above_360() -> None:
    """Values >= 360 are permissible; normalisation happens downstream."""
    cfg = PipelineConfig(reference_heading_deg=450.0)
    assert cfg.reference_heading_deg == 450.0


def test_pipeline_config_use_imu_rates_defaults_false() -> None:
    assert PipelineConfig().use_imu_rates is False


def test_pipeline_config_use_imu_rates_can_be_set_true() -> None:
    assert PipelineConfig(use_imu_rates=True).use_imu_rates is True
