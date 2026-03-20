"""Pipeline configuration for the Tolles-Lawson magnetic compensation model."""

from __future__ import annotations

import datetime
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
    model_terms: Literal["a", "b", "c", "d"] = Field(
        default="c",
        description=(
            "Tolles-Lawson term set: "
            "'a' = permanent (3 terms), "
            "'b' = permanent + induced (9 terms), "
            "'c' = full permanent + induced + eddy (18 terms), "
            "'d' = full permanent + induced + eddy + rate derivatives (21 terms)."
        ),
    )
    earth_field_method: Literal["igrf", "steady_mean"] = Field(
        default="igrf",
        description=(
            "Earth field baseline method: "
            "'igrf' evaluates the IGRF model at each sample (primary path); "
            "'steady_mean' uses the mean B_total of steady-maneuver segments"
            " (fallback)."
        ),
    )
    igrf_date: datetime.date | None = Field(
        default_factory=datetime.date.today,
        description=(
            "Date for IGRF model evaluation. "
            "Required when earth_field_method is 'igrf'."
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
    segment_label_col: str | None = Field(
        default=None,
        description=(
            "Name of the column containing pre-existing '<maneuver>_<heading>' labels "
            "(e.g. 'pitch_N'). None triggers auto-detection."
        ),
    )
    reference_heading_deg: float | None = Field(
        default=None,
        description=(
            "Reference heading of the northernmost flight leg in degrees. "
            "Accepts any real number — values are normalised modulo 360° internally. "
            "None = auto-detect via folded circular mean of heading values. "
            "Example: 0.0 for cardinal N/E/S/W; 45.0 for a 45° oblique flight."
        ),
    )
    heading_tolerance_deg: float = Field(
        default=45.0,
        gt=0.0,
        description="Half-width of each cardinal heading bin [degrees].",
    )
    condition_number_threshold: float = Field(
        default=1e6,
        gt=0.0,
        description=(
            "Condition number threshold above which a warning is emitted "
            "and, in adaptive compensation, the pitch/roll/yaw blending "
            "weight is suppressed to zero, falling back to baseline "
            "coefficients."
        ),
    )
    use_imu_rates: bool = Field(
        default=False,
        description=(
            "When True, use IMU angular rate channels (roll_rate, pitch_rate, yaw_rate)"
            " in place of numerically-differentiated direction cosines for eddy-current"
            " term estimation. Only applied when model_terms='c' or 'd'. "
            "Requires all three IMU columns to be present in the input DataFrame."
        ),
    )
    compensation_strategy: Literal["standard", "adaptive_maneuver"] = Field(
        default="standard",
        description=(
            "Compensation strategy: 'standard' uses a single coefficient set; "
            "'adaptive_maneuver' blends per-maneuver coefficients based on "
            "detected maneuver intensity."
        ),
    )
    maneuver_detection_window: int = Field(
        default=50,
        gt=0,
        description=(
            "Rolling window size [samples] for computing direction-cosine variance "
            "used to detect maneuver intensity. "
            "At 10 Hz, 50 samples ≈ 5 seconds."
        ),
    )
    maneuver_baseline_weight: float = Field(
        default=0.1,
        gt=0.0,
        description=(
            "Constant additive weight for baseline coefficients during blending. "
            "Keeps a small baseline contribution even when a strong maneuver is "
            "detected, preventing zero-weight extrapolation."
        ),
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

    @model_validator(mode="after")
    def _check_igrf_date(self) -> PipelineConfig:
        if self.earth_field_method == "igrf" and self.igrf_date is None:
            raise ValueError("igrf_date is required when earth_field_method is 'igrf'.")
        return self
