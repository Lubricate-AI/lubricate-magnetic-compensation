"""DataFrame validation against the lmc input schema."""

import polars as pl

from lmc.columns import COL_TIME, REQUIRED_COLUMNS


def validate_dataframe(df: object) -> pl.DataFrame:
    """Validate that a DataFrame conforms to the lmc input schema.

    Parameters
    ----------
    df
        The input DataFrame to validate.

    Returns
    -------
    pl.DataFrame
        The original DataFrame, unchanged, if it passes all checks.

    Raises
    ------
    TypeError
        If ``df`` is not a ``polars.DataFrame``.
    ValueError
        If any validation checks fail. The message lists every failing check.

    """
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"Expected a polars.DataFrame, got {type(df).__name__}.")

    errors: list[str] = []

    if df.is_empty():
        errors.append("DataFrame must not be empty.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}.")

    if not missing:
        wrong_dtype = [col for col in REQUIRED_COLUMNS if df[col].dtype != pl.Float64]
        if wrong_dtype:
            errors.append(
                "Columns must be Float64, but these have wrong dtype: "
                + ", ".join(f"{col}={df[col].dtype}" for col in wrong_dtype)
                + "."
            )

        null_cols = [col for col in REQUIRED_COLUMNS if df[col].null_count() > 0]
        if null_cols:
            errors.append(f"Columns contain null values: {null_cols}.")

    if COL_TIME in df.columns and df[COL_TIME].dtype == pl.Float64:
        diffs = df[COL_TIME].diff().drop_nulls()
        if not (diffs > 0).all():
            errors.append(
                f"Column '{COL_TIME}' must be strictly monotonically increasing."
            )

    if errors:
        raise ValueError(
            "DataFrame validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return df
