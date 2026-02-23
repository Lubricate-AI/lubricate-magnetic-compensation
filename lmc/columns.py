"""Column name constants for the lmc input DataFrame schema."""

COL_TIME: str = "time"
COL_LAT: str = "lat"
COL_LON: str = "lon"
COL_ALT: str = "alt"
COL_BTOTAL: str = "B_total"
COL_BX: str = "B_x"
COL_BY: str = "B_y"
COL_BZ: str = "B_z"

COL_DELTA_B: str = "delta_B"
COL_TMI_COMPENSATED: str = "tmi_compensated"

# --- Attitude / heading columns ---
# COL_HEADING doubles as the yaw signal for maneuver detection.
COL_HEADING: str = "heading"  # aircraft compass heading [0, 360) degrees
COL_PITCH: str = "pitch"  # aircraft pitch angle [degrees]
COL_ROLL: str = "roll"  # aircraft roll angle [degrees]
COL_SEGMENT_LABEL: str = "segment"  # pre-labeled segment column, e.g. "pitch_N"

# --- Optional IMU angular rate columns (not in REQUIRED_COLUMNS) ---
COL_ROLL_RATE: str = "roll_rate"  # aircraft roll rate [rad/s]
COL_PITCH_RATE: str = "pitch_rate"  # aircraft pitch rate [rad/s]
COL_YAW_RATE: str = "yaw_rate"  # aircraft yaw rate [rad/s]

REQUIRED_COLUMNS: tuple[str, ...] = (
    COL_TIME,
    COL_LAT,
    COL_LON,
    COL_ALT,
    COL_BTOTAL,
    COL_BX,
    COL_BY,
    COL_BZ,
)

# --- Tolles-Lawson A-matrix column names ---

# Permanent terms (3)
COL_COS_X: str = "cos_x"
COL_COS_Y: str = "cos_y"
COL_COS_Z: str = "cos_z"

# Induced terms (6)
COL_COS_X2: str = "cos_x_cos_x"
COL_COS_XY: str = "cos_x_cos_y"
COL_COS_XZ: str = "cos_x_cos_z"
COL_COS_Y2: str = "cos_y_cos_y"
COL_COS_YZ: str = "cos_y_cos_z"
COL_COS_Z2: str = "cos_z_cos_z"

# Eddy current terms (9)
COL_COS_X_DCOS_X: str = "cos_x_dcos_x"
COL_COS_X_DCOS_Y: str = "cos_x_dcos_y"
COL_COS_X_DCOS_Z: str = "cos_x_dcos_z"
COL_COS_Y_DCOS_X: str = "cos_y_dcos_x"
COL_COS_Y_DCOS_Y: str = "cos_y_dcos_y"
COL_COS_Y_DCOS_Z: str = "cos_y_dcos_z"
COL_COS_Z_DCOS_X: str = "cos_z_dcos_x"
COL_COS_Z_DCOS_Y: str = "cos_z_dcos_y"
COL_COS_Z_DCOS_Z: str = "cos_z_dcos_z"
