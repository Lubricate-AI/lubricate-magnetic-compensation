"""Column name constants for the lmc input DataFrame schema."""

COL_TIME: str = "time"
COL_LAT: str = "lat"
COL_LON: str = "lon"
COL_BTOTAL: str = "B_total"
COL_BX: str = "B_x"
COL_BY: str = "B_y"
COL_BZ: str = "B_z"

REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_TIME,
    COL_LAT,
    COL_LON,
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
)
