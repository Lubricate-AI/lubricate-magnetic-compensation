"""Unit tests for lmc.cli.commands."""

from __future__ import annotations

from typer.testing import CliRunner

from lmc.cli.commands import app

runner = CliRunner()


def test_calibrate_cmd_accepts_use_cv_flag() -> None:
    """--use-cv must be a recognised option."""
    result = runner.invoke(app, ["calibrate", "nonexistent.csv", "--use-cv"])
    assert "No such option: --use-cv" not in (result.output or "")


def test_calibrate_cmd_accepts_cv_folds_option() -> None:
    """--cv-folds must be a recognised option."""
    result = runner.invoke(app, ["calibrate", "nonexistent.csv", "--cv-folds", "10"])
    assert "No such option: --cv-folds" not in (result.output or "")


def test_calibrate_cmd_accepts_auto_regularize_flag() -> None:
    """--auto-regularize must be a recognised option."""
    result = runner.invoke(app, ["calibrate", "nonexistent.csv", "--auto-regularize"])
    assert "No such option: --auto-regularize" not in (result.output or "")


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
