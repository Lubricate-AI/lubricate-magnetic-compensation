"""CLI commands for lubricate-magnetic-compensation."""

import datetime

import typer

app = typer.Typer(
    name="lubricate-magnetic-compensation",
    help="Calculating magnetic compensation coefficients using the Tolles-Lawson model",
    add_completion=False,
)


@app.command("run")
def run(
    igrf_date: str = typer.Option(
        ...,
        "--igrf-date",
        help="Date for IGRF model evaluation in YYYY-MM-DD format.",
    ),
) -> None:
    """Run the magnetic compensation calculation."""
    try:
        date = datetime.date.fromisoformat(igrf_date)
    except ValueError:
        typer.echo(
            typer.style(
                f"Invalid --igrf-date {igrf_date!r}. Expected YYYY-MM-DD.",
                fg=typer.colors.RED,
            ),
            err=True,
        )
        raise typer.Exit(code=1) from None

    try:
        typer.echo(typer.style(f"✓ IGRF date: {date}", fg=typer.colors.GREEN))

    except Exception as e:
        typer.echo(typer.style(f"Unexpected error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from None
