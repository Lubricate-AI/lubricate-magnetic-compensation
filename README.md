# lubricate-magnetic-compensation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://lubricate-ai.github.io/lubricate-magnetic-compensation/)
[![CI Status](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/actions/workflows/pull-request.yml/badge.svg)](https://github.com/Lubricate-AI/lubricate-magnetic-compensation/actions/workflows/pull-request.yml)
[![Ruff](https://img.shields.io/badge/linter-ruff-blue)](https://github.com/astral-sh/ruff)
[![Type Checking](https://img.shields.io/badge/type%20checking-pyright-blue)](https://github.com/microsoft/pyright)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

A Python library for aircraft magnetic compensation using the Tolles-Lawson model. Estimate compensation coefficients from figure-of-merit (FOM) calibration flights and remove aircraft-induced magnetic interference from survey data.

## Features

- Three Tolles-Lawson term sets: permanent (`a`, 3 terms), permanent + induced (`b`, 9 terms), full (`c`, 18 terms)
- IGRF earth field baseline (via `ppigrf`) or steady-maneuver mean
- Automatic FOM flight segmentation with optional pre-labelled column
- Ridge (L2) regularisation option
- Optional IMU angular rates for eddy-current term estimation
- Figure-of-merit (FOM) quality report
- CLI (`lubricate-magnetic-compensation calibrate / compensate`) and Python API
- Pydantic-validated pipeline configuration

## Installation

Install from source:

```bash
git clone https://github.com/Lubricate-AI/lubricate-magnetic-compensation.git
cd lubricate-magnetic-compensation
make install
```

## Quick Start

### CLI

```bash
# Calibrate from FOM flight CSV → coefficients.json
lubricate-magnetic-compensation calibrate fom_flight.csv

# Apply compensation to survey data
lubricate-magnetic-compensation compensate survey.csv --coefficients coefficients.json
```

### Python API

```python
import polars as pl
from lmc import PipelineConfig, calibrate, compensate, compute_fom_report, segment_fom, validate_dataframe, compute_interference

df = pl.read_csv("fom_flight.csv")
df = validate_dataframe(df)

config = PipelineConfig(model_terms="c")
segments = segment_fom(df, config)
df = df.with_columns(compute_interference(df, config))

result = calibrate(df, segments, config)
report = compute_fom_report(df, segments, result)
print(report.to_json())
```

## CLI Reference

### `calibrate`

```bash
lubricate-magnetic-compensation calibrate [OPTIONS] INPUT_CSV
```

| Option | Default | Description |
|---|---|---|
| `--output` | `coefficients.json` | Output path for coefficients JSON |
| `--igrf-date` | today | Date for IGRF evaluation (YYYY-MM-DD) |
| `--model-terms` | `c` | Term set: `a` (3), `b` (9), `c` (18) |
| `--earth-field-method` | `igrf` | `igrf` or `steady_mean` |
| `--use-ridge / --no-use-ridge` | off | Enable ridge regression |
| `--ridge-alpha` | `0.001` | Ridge regularisation strength |
| `--segment-label-col` | auto | Pre-labelled segment column name |

### `compensate`

```bash
lubricate-magnetic-compensation compensate [OPTIONS] INPUT_CSV
```

| Option | Default | Description |
|---|---|---|
| `--coefficients` | required | Path to coefficients JSON |
| `--output` | `compensated.csv` | Output path for compensated CSV |

## Python API

Public symbols exported from `lmc`:

| Symbol | Description |
|---|---|
| `calibrate`, `compensate` | Core pipeline functions |
| `PipelineConfig` | Pydantic model for all pipeline settings |
| `CalibrationResult` | Dataclass with coefficients, residuals, condition number |
| `validate_dataframe` | Validate and normalise an input DataFrame |
| `build_feature_matrix` | Construct the Tolles-Lawson design matrix |
| `compute_interference` | Compute aircraft interference field expression |
| `segment_fom` | Segment a FOM calibration flight into maneuver windows |
| `compute_fom_report` | Compute a figure-of-merit quality report |
| `FomReport`, `Segment` | Result types for FOM reporting |
| `COL_*`, `REQUIRED_COLUMNS` | Column name constants |

## Development

```bash
make install       # install / update dependencies
make lint          # ruff + typos + yamllint
make type-checking # pyright
make format        # auto-format
make test          # pytest
```

## License

MIT — see [LICENSE](LICENSE).
