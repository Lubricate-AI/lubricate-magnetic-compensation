"""CLI commands for lubricate-magnetic-compensation."""

import typer

app = typer.Typer(
    name="lubricate-magnetic-compensation",
    help="Calculating magnetic compensation coefficients using the Tolles-Lawson model",
    add_completion=False,
)


@app.command("run")
def run() -> None:
    """Run the magnetic compensation calculation."""
    try:
        typer.echo(typer.style("✓ Hello world", fg=typer.colors.GREEN))

    except Exception as e:
        typer.echo(typer.style(f"Unexpected error: {e}", fg=typer.colors.RED), err=True)
        raise typer.Exit(code=1) from None
