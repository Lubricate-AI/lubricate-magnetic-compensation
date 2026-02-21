"""Pipeline configuration for the Tolles-Lawson magnetic compensation model."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PipelineConfig(BaseModel):
    """Configuration for the Tolles-Lawson magnetic compensation pipeline."""

    model_config = ConfigDict(frozen=True)

    sample_rate_hz: float = Field(
        default=10.0,
        gt=0.0,
        description="Sensor sampling rate in Hz.",
    )
    bandpass_low_hz: float = Field(
        default=0.1,
        gt=0.0,
        description="Lower cutoff frequency of the band-pass filter [Hz].",
    )
    bandpass_high_hz: float = Field(
        default=0.9,
        gt=0.0,
        description="Upper cutoff frequency of the band-pass filter [Hz].",
    )
    model_terms: Literal["a", "b", "c"] = Field(
        default="c",
        description=(
            "Tolles-Lawson term set: "
            "'a' = permanent (3 terms), "
            "'b' = permanent + induced (9 terms), "
            "'c' = full permanent + induced + eddy (18 terms)."
        ),
    )
    use_ridge: bool = Field(
        default=False,
        description="Use ridge (L2-regularised) least squares instead of OLS.",
    )
    ridge_alpha: float = Field(
        default=1e-3,
        ge=0.0,
        description="Ridge regularisation strength. Ignored when use_ridge is False.",
    )

    @model_validator(mode="after")
    def _check_bandpass(self) -> PipelineConfig:
        if self.bandpass_low_hz >= self.bandpass_high_hz:
            raise ValueError(
                f"bandpass_low_hz ({self.bandpass_low_hz}) must be strictly less than "
                f"bandpass_high_hz ({self.bandpass_high_hz})."
            )
        nyquist = self.sample_rate_hz / 2.0
        if self.bandpass_high_hz >= nyquist:
            raise ValueError(
                f"bandpass_high_hz ({self.bandpass_high_hz}) must be below the "
                f"Nyquist frequency ({nyquist} Hz)."
            )
        return self
