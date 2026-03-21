"""Unit tests for lmc.cli.commands."""

from __future__ import annotations

from typer.testing import CliRunner

from lmc.cli.commands import app

runner = CliRunner()


def test_calibrate_rejects_conflicting_regularization_flags() -> None:
    """Passing --use-lasso and --use-ridge together should exit with code 1."""
    result = runner.invoke(
        app,
        [
            "calibrate",
            "dummy.csv",
            "--use-lasso",
            "--use-ridge",
        ],
    )
    assert result.exit_code == 1
