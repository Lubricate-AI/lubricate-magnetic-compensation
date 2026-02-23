"""CLI commands for lubricate-magnetic-compensation."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import polars as pl
import typer

from lmc import (
    CalibrationResult,
    PipelineConfig,
    calibrate,
    compensate,
    compute_fom_report,
    compute_interference,
    segment_fom,
    validate_dataframe,
)

app = typer.Typer(
    name="lubricate-magnetic-compensation",
    help="Calculating magnetic compensation coefficients using the Tolles-Lawson model",
    add_completion=False,
)


@app.command("calibrate")
def calibrate_cmd(
    input_csv: Annotated[Path, typer.Argument(help="FOM calibration data CSV")],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output path for coefficients JSON."),
    ] = Path("coefficients.json"),
    igrf_date: Annotated[
        str | None,
        typer.Option(
            "--igrf-date",
            help="Date for IGRF model evaluation (YYYY-MM-DD). Defaults to today.",
        ),
    ] = None,
    model_terms: Annotated[
        Literal["a", "b", "c"],
        typer.Option(
            "--model-terms",
            help="Tolles-Lawson term set: a (permanent), b (+induced), c (full).",
        ),
    ] = "c",
    earth_field_method: Annotated[
        Literal["igrf", "steady_mean"],
        typer.Option(
            "--earth-field-method",
            help="Earth field baseline method: igrf or steady_mean.",
        ),
    ] = "igrf",
    use_ridge: Annotated[
        bool,
        typer.Option("--use-ridge/--no-use-ridge", help="Use ridge (L2) regression."),
    ] = False,
    ridge_alpha: Annotated[
        float,
        typer.Option("--ridge-alpha", help="Ridge regularisation strength."),
    ] = 1e-3,
    segment_label_col: Annotated[
        str | None,
        typer.Option("--segment-label-col", help="Pre-labeled segment column name."),
    ] = None,
) -> None:
    """Calibrate Tolles-Lawson model from FOM flight data."""
    try:
        parsed_date: datetime.date
        if igrf_date is not None:
            try:
                parsed_date = datetime.date.fromisoformat(igrf_date)
            except ValueError:
                typer.echo(
                    typer.style(
                        f"Invalid --igrf-date {igrf_date!r}. Expected YYYY-MM-DD.",
                        fg=typer.colors.RED,
                    ),
                    err=True,
                )
                raise typer.Exit(code=1) from None
        else:
            parsed_date = datetime.date.today()

        config = PipelineConfig(
            model_terms=model_terms,
            earth_field_method=earth_field_method,
            igrf_date=parsed_date,
            use_ridge=use_ridge,
            ridge_alpha=ridge_alpha,
            segment_label_col=segment_label_col,
        )

        df = pl.read_csv(input_csv)
        df = validate_dataframe(df)
        segments = segment_fom(df, config)

        steady_mask: pl.Series | None = None
        if earth_field_method == "steady_mean":
            mask_arr = np.zeros(df.height, dtype=bool)
            for seg in segments:
                if seg.maneuver == "steady":
                    mask_arr[seg.start_idx : seg.end_idx] = True
            steady_mask = pl.Series(mask_arr, dtype=pl.Boolean)

        delta_b = compute_interference(df, config, steady_mask=steady_mask)
        df = df.with_columns(delta_b)

        result = calibrate(df, segments, config)
        report = compute_fom_report(df, segments, result)

        typer.echo(report.to_json())

        coef_data = {
            "model_terms": model_terms,
            "coefficients": result.coefficients.tolist(),
            "n_terms": result.n_terms,
            "condition_number": result.condition_number,
        }
        output.write_text(json.dumps(coef_data, indent=2))

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(
            typer.style(f"Error: {e}", fg=typer.colors.RED),
            err=True,
        )
        raise typer.Exit(code=1) from None


@app.command("compensate")
def compensate_cmd(
    input_csv: Annotated[Path, typer.Argument(help="Survey data CSV")],
    coefficients: Annotated[
        Path,
        typer.Option(
            "--coefficients",
            help="Path to coefficients JSON produced by the calibrate command.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Output path for compensated CSV."),
    ] = Path("compensated.csv"),
) -> None:
    """Apply Tolles-Lawson compensation to survey data."""
    try:
        df = pl.read_csv(input_csv)
        df = validate_dataframe(df)

        coef_data: dict[str, object] = json.loads(coefficients.read_text())
        model_terms = cast(Literal["a", "b", "c"], coef_data["model_terms"])
        coefs = np.array(coef_data["coefficients"], dtype=np.float64)
        n_terms = int(coef_data["n_terms"])  # type: ignore[arg-type]
        condition_number = float(coef_data["condition_number"])  # type: ignore[arg-type]

        result = CalibrationResult(
            coefficients=coefs,
            residuals=np.empty(0, dtype=np.float64),
            condition_number=condition_number,
            n_terms=n_terms,
        )

        config = PipelineConfig(model_terms=model_terms)
        df_result = compensate(df, result, config)
        df_result.write_csv(output)

    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(
            typer.style(f"Error: {e}", fg=typer.colors.RED),
            err=True,
        )
        raise typer.Exit(code=1) from None
