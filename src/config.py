"""
Configuration module for the Predictive Maintenance Feature Engineering project.

Task:
    Task 2 - Feature Engineering

Purpose:
    Centralizes project paths, dataset configuration, feature-engineering
    parameters, visualization settings, train/validation/test settings,
    reporting configuration, and real-time feature-engineering settings.

Design principles:
    - Reproducible
    - Configuration-driven
    - Leakage-safe
    - Compatible with AI4I 2020 reference data
    - Extensible to real timestamp + machine_id datasets
    - Suitable for batch and real-time processing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ============================================================================
# DIRECTORY STRUCTURE
# ============================================================================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LOGS_DIR = PROJECT_ROOT / "logs"

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_NAME = "AI4I 2020 Predictive Maintenance Dataset"

DATASET_SOURCE = (
    "UCI Machine Learning Repository - AI4I 2020 Predictive Maintenance Dataset"
)

DATASET_DOI = "10.24432/C5HS5C"

DATASET_URL = "https://archive.ics.uci.edu/dataset/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdat"

RAW_DATA_FILENAME = "ai4i2020.csv"

RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME


# ============================================================================
# OPERATING MODE
# ============================================================================

# Available modes:
#
# "ai4i"
#     Used for the UCI AI4I reference dataset. UID is treated as an identifier,
#     not as a genuine machine timestamp.
#
# "timeseries"
#     Used for a real industrial dataset containing timestamp and machine ID.
#
# "auto"
#     Automatically detects whether a valid timestamp/machine identifier exists.

DATA_MODE = "auto"


# ============================================================================
# COLUMN CONFIGURATION
# ============================================================================

# AI4I original column names
#
# The official CSV commonly contains:
# UID
# Product ID
# Type
# Air temperature [K]
# Process temperature [K]
# Rotational speed [rpm]
# Torque [Nm]
# Tool wear [min]
# Machine failure
# TWF
# HDF
# PWF
# OSF
# RNF

ID_COLUMNS: List[str] = [
    "UID",
    "Product ID",
]

CATEGORICAL_COLUMNS: List[str] = [
    "Type",
]

TARGET_COLUMN = "Machine failure"

FAILURE_MODE_COLUMNS: List[str] = [
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]

# These columns identify the physical sensor/process measurements
NUMERIC_SENSOR_COLUMNS: List[str] = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# ============================================================================
# REAL TIME-SERIES COLUMN CONFIGURATION
# ============================================================================

# For an actual production dataset, these values can be changed.

TIMESTAMP_COLUMN = "timestamp"

MACHINE_ID_COLUMN = "machine_id"

# Optional columns that may exist in a real industrial dataset.
# The feature-engineering implementation will use them only when available.

OPTIONAL_TIME_COLUMNS: List[str] = [
    TIMESTAMP_COLUMN,
]

OPTIONAL_GROUP_COLUMNS: List[str] = [
    MACHINE_ID_COLUMN,
]


# ============================================================================
# TRAIN / VALIDATION / TEST CONFIGURATION
# ============================================================================

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

SPLIT_RANDOM_STATE = 42

# For AI4I/reference data, a chronological-style split based on row ordering
# is preferable to randomly mixing observations when evaluating temporal
# feature behavior.
#
# The actual implementation will determine whether chronological splitting
# is appropriate based on DATA_MODE.

USE_CHRONOLOGICAL_SPLIT = True

# Whether grouped splitting should be used when machine_id is available.
USE_GROUPED_SPLIT = True


# ============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ============================================================================

# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

LAG_PERIODS: List[int] = [
    1,
    2,
    3,
    5,
    10,
]

# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------

ROLLING_WINDOWS: List[int] = [
    3,
    5,
    10,
    20,
]

# Minimum observations required before a rolling statistic is calculated.
#
# Using min_periods=1 prevents unnecessary missing values at the beginning
# of a sequence while preserving causality.

ROLLING_MIN_PERIODS = 1


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

ROLLING_STATISTICS: List[str] = [
    "mean",
    "std",
    "min",
    "max",
    "median",
]


# ---------------------------------------------------------------------------
# Exponentially weighted moving averages
# ---------------------------------------------------------------------------

EWM_SPANS: List[int] = [
    3,
    5,
    10,
]


# ============================================================================
# RATE-OF-CHANGE CONFIGURATION
# ============================================================================

ROC_PERIODS: List[int] = [
    1,
    3,
    5,
]


# ============================================================================
# Z-SCORE / ANOMALY CONFIGURATION
# ============================================================================

Z_SCORE_WINDOWS: List[int] = [
    5,
    10,
    20,
]

ANOMALY_Z_THRESHOLD = 3.0


# ============================================================================
# PHYSICAL FEATURE CONFIGURATION
# ============================================================================

# AI4I uses:
#
# Power = Torque × angular velocity
#
# angular velocity = rotational speed × 2π / 60
#
# resulting unit: watts

CREATE_MECHANICAL_POWER = True

CREATE_TORQUE_SPEED_INTERACTION = True

CREATE_TEMPERATURE_GAP = True

CREATE_TOOL_WEAR_TRANSFORMATIONS = True

CREATE_SENSOR_DIFFERENCES = True

CREATE_SENSOR_RATIOS = True

CREATE_INTERACTION_FEATURES = True


# ============================================================================
# SENSOR FEATURE DEFINITIONS
# ============================================================================

TEMPERATURE_COLUMNS = {
    "air": "Air temperature [K]",
    "process": "Process temperature [K]",
}

ROTATIONAL_SPEED_COLUMN = "Rotational speed [rpm]"

TORQUE_COLUMN = "Torque [Nm]"

TOOL_WEAR_COLUMN = "Tool wear [min]"


# ============================================================================
# FEATURE QUALITY CONFIGURATION
# ============================================================================

# Remove features where the same value occurs in every observation.

REMOVE_CONSTANT_FEATURES = True

# Remove exact duplicate columns.

REMOVE_DUPLICATE_FEATURES = True

# Missing-value threshold.
#
# A feature with more missing values than this proportion can be removed.

MAX_MISSING_RATIO = 0.40

# High-cardinality categorical feature threshold.

MAX_CATEGORICAL_CARDINALITY = 100


# ============================================================================
# NUMERICAL SAFETY
# ============================================================================

# Small value used when a denominator may be zero.

EPSILON = 1e-8

# Clip extreme engineered numerical values to prevent numerical instability.

CLIP_EXTREME_VALUES = True

LOWER_QUANTILE_CLIP = 0.001

UPPER_QUANTILE_CLIP = 0.999


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

FIGURE_DPI = 180

FIGURE_WIDTH = 12

FIGURE_HEIGHT = 7

FIGURE_FORMAT = "png"

# Professional reporting palette.
#
# These colors are centralized here so all plots use a consistent visual
# identity.

COLORS = {
    "primary": "#17365D",
    "secondary": "#2F75B5",
    "accent": "#00A6A6",
    "warning": "#F4B183",
    "danger": "#C00000",
    "success": "#70AD47",
    "neutral": "#7F8C8D",
    "dark": "#1F2937",
    "light": "#F3F6F9",
    "white": "#FFFFFF",
}


# ============================================================================
# FEATURE VISUALIZATION CONFIGURATION
# ============================================================================

TOP_N_FEATURES_FOR_PLOTS = 20

TOP_N_IMPORTANT_FEATURES = 25

CORRELATION_THRESHOLD = 0.85

DISTRIBUTION_SAMPLE_SIZE = 5000

ROLLING_VISUALIZATION_SAMPLE_SIZE = 2000


# ============================================================================
# REPORT CONFIGURATION
# ============================================================================

HTML_REPORT_FILENAME = "feature_engineering_report.html"

HTML_REPORT_PATH = REPORTS_DIR / HTML_REPORT_FILENAME

DOCX_REPORT_FILENAME = "feature_engineering_documentation.docx"

DOCX_REPORT_PATH = REPORTS_DIR / DOCX_REPORT_FILENAME


# ============================================================================
# ENGINEERED DATA OUTPUTS
# ============================================================================

ENGINEERED_DATA_FILENAME = "engineered_features.csv"

TRAIN_FEATURES_FILENAME = "train_features.csv"

VALIDATION_FEATURES_FILENAME = "validation_features.csv"

TEST_FEATURES_FILENAME = "test_features.csv"

FEATURE_METADATA_FILENAME = "feature_metadata.csv"

FEATURE_QUALITY_FILENAME = "feature_quality_report.csv"

FEATURE_CORRELATION_FILENAME = "feature_correlation_matrix.csv"


ENGINEERED_DATA_PATH = PROCESSED_DATA_DIR / ENGINEERED_DATA_FILENAME

TRAIN_FEATURES_PATH = PROCESSED_DATA_DIR / TRAIN_FEATURES_FILENAME

VALIDATION_FEATURES_PATH = PROCESSED_DATA_DIR / VALIDATION_FEATURES_FILENAME

TEST_FEATURES_PATH = PROCESSED_DATA_DIR / TEST_FEATURES_FILENAME

FEATURE_METADATA_PATH = PROCESSED_DATA_DIR / FEATURE_METADATA_FILENAME

FEATURE_QUALITY_PATH = PROCESSED_DATA_DIR / FEATURE_QUALITY_FILENAME

FEATURE_CORRELATION_PATH = PROCESSED_DATA_DIR / FEATURE_CORRELATION_FILENAME


# ============================================================================
# REAL-TIME FEATURE ENGINEERING
# ============================================================================

REALTIME_STATE_MAX_ROWS = 100

REALTIME_DEFAULT_MACHINE_ID = "unknown_machine"

# Whether real-time processing should reject observations that arrive
# out-of-order.

REALTIME_REJECT_OUT_OF_ORDER = True

# Whether the online processor should maintain separate state per machine.

REALTIME_GROUP_BY_MACHINE = True


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_FILENAME = "task2_feature_engineering.log"

LOG_PATH = LOGS_DIR / LOG_FILENAME

LOG_LEVEL = "INFO"


# ============================================================================
# RANDOMNESS / REPRODUCIBILITY
# ============================================================================

RANDOM_STATE = 42


# ============================================================================
# MODELING / LEAKAGE SAFETY
# ============================================================================

# Target and failure-mode columns must NEVER be used to generate predictive
# features unless explicitly requested for a separate diagnostic analysis.
#
# This is extremely important because AI4I contains individual failure-mode
# labels such as TWF, HDF, PWF, OSF and RNF.

EXCLUDED_FROM_FEATURE_GENERATION: List[str] = [
    TARGET_COLUMN,
    *FAILURE_MODE_COLUMNS,
]


# ============================================================================
# AI4I PHYSICAL THRESHOLDS
# ============================================================================

# These values come from the official AI4I dataset documentation and are
# useful for creating diagnostic/operating-condition features.
#
# They should NOT be interpreted as learned model parameters.

AI4I_THRESHOLDS: Dict[str, float] = {
    "heat_temperature_gap_failure_threshold_k": 8.6,
    "heat_rotational_speed_failure_threshold_rpm": 1380.0,
    "power_failure_lower_w": 3500.0,
    "power_failure_upper_w": 9000.0,
    "tool_wear_failure_lower_min": 200.0,
    "tool_wear_failure_upper_min": 240.0,
}


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

@dataclass(frozen=True)
class SplitConfig:
    """Configuration for train/validation/test splitting."""

    train_size: float = TRAIN_SIZE
    validation_size: float = VALIDATION_SIZE
    test_size: float = TEST_SIZE
    random_state: int = SPLIT_RANDOM_STATE
    chronological: bool = USE_CHRONOLOGICAL_SPLIT
    grouped: bool = USE_GROUPED_SPLIT

    def validate(self) -> None:
        """Validate split proportions."""

        total = (
            self.train_size
            + self.validation_size
            + self.test_size
        )

        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1.0. "
                f"Received {total:.6f}."
            )

        if not 0 < self.train_size < 1:
            raise ValueError("train_size must be between 0 and 1.")

        if not 0 < self.validation_size < 1:
            raise ValueError("validation_size must be between 0 and 1.")

        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration controlling engineered feature generation."""

    lag_periods: Tuple[int, ...] = tuple(LAG_PERIODS)

    rolling_windows: Tuple[int, ...] = tuple(ROLLING_WINDOWS)

    rolling_min_periods: int = ROLLING_MIN_PERIODS

    rolling_statistics: Tuple[str, ...] = tuple(ROLLING_STATISTICS)

    ewm_spans: Tuple[int, ...] = tuple(EWM_SPANS)

    roc_periods: Tuple[int, ...] = tuple(ROC_PERIODS)

    z_score_windows: Tuple[int, ...] = tuple(Z_SCORE_WINDOWS)

    anomaly_z_threshold: float = ANOMALY_Z_THRESHOLD

    create_mechanical_power: bool = CREATE_MECHANICAL_POWER

    create_torque_speed_interaction: bool = CREATE_TORQUE_SPEED_INTERACTION

    create_temperature_gap: bool = CREATE_TEMPERATURE_GAP

    create_tool_wear_transformations: bool = CREATE_TOOL_WEAR_TRANSFORMATIONS

    create_sensor_differences: bool = CREATE_SENSOR_DIFFERENCES

    create_sensor_ratios: bool = CREATE_SENSOR_RATIOS

    create_interaction_features: bool = CREATE_INTERACTION_FEATURES


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for feature-quality validation."""

    remove_constant_features: bool = REMOVE_CONSTANT_FEATURES

    remove_duplicate_features: bool = REMOVE_DUPLICATE_FEATURES

    max_missing_ratio: float = MAX_MISSING_RATIO

    max_categorical_cardinality: int = MAX_CATEGORICAL_CARDINALITY

    correlation_threshold: float = CORRELATION_THRESHOLD

    def validate(self) -> None:
        """Validate feature-quality configuration."""

        if not 0 <= self.max_missing_ratio <= 1:
            raise ValueError(
                "max_missing_ratio must be between 0 and 1."
            )

        if not 0 <= self.correlation_threshold <= 1:
            raise ValueError(
                "correlation_threshold must be between 0 and 1."
            )


@dataclass(frozen=True)
class VisualizationConfig:
    """Configuration for professional visualizations."""

    dpi: int = FIGURE_DPI

    width: float = FIGURE_WIDTH

    height: float = FIGURE_HEIGHT

    figure_format: str = FIGURE_FORMAT

    top_n_features: int = TOP_N_FEATURES_FOR_PLOTS

    top_n_important_features: int = TOP_N_IMPORTANT_FEATURES

    distribution_sample_size: int = DISTRIBUTION_SAMPLE_SIZE

    rolling_sample_size: int = ROLLING_VISUALIZATION_SAMPLE_SIZE

    colors: Dict[str, str] = field(
        default_factory=lambda: dict(COLORS)
    )


@dataclass(frozen=True)
class RealtimeConfig:
    """Configuration for online/real-time feature generation."""

    state_max_rows: int = REALTIME_STATE_MAX_ROWS

    default_machine_id: str = REALTIME_DEFAULT_MACHINE_ID

    reject_out_of_order: bool = REALTIME_REJECT_OUT_OF_ORDER

    group_by_machine: bool = REALTIME_GROUP_BY_MACHINE


# ============================================================================
# GLOBAL CONFIGURATION INSTANCES
# ============================================================================

SPLIT_CONFIG = SplitConfig()

FEATURE_CONFIG = FeatureEngineeringConfig()

QUALITY_CONFIG = QualityConfig()

VISUALIZATION_CONFIG = VisualizationConfig()

REALTIME_CONFIG = RealtimeConfig()


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_configuration() -> None:
    """
    Validate all major configuration values.

    Raises
    ------
    ValueError
        If an invalid configuration is detected.
    """

    SPLIT_CONFIG.validate()

    QUALITY_CONFIG.validate()

    if DATA_MODE not in {"auto", "ai4i", "timeseries"}:
        raise ValueError(
            "DATA_MODE must be one of: 'auto', 'ai4i', 'timeseries'."
        )

    if not LAG_PERIODS:
        raise ValueError("LAG_PERIODS cannot be empty.")

    if not ROLLING_WINDOWS:
        raise ValueError("ROLLING_WINDOWS cannot be empty.")

    if not EWM_SPANS:
        raise ValueError("EWM_SPANS cannot be empty.")

    if not ROC_PERIODS:
        raise ValueError("ROC_PERIODS cannot be empty.")

    if not Z_SCORE_WINDOWS:
        raise ValueError("Z_SCORE_WINDOWS cannot be empty.")

    if any(period <= 0 for period in LAG_PERIODS):
        raise ValueError("All lag periods must be positive integers.")

    if any(window <= 0 for window in ROLLING_WINDOWS):
        raise ValueError("All rolling windows must be positive integers.")

    if any(span <= 0 for span in EWM_SPANS):
        raise ValueError("All EWM spans must be positive integers.")

    if any(period <= 0 for period in ROC_PERIODS):
        raise ValueError("All ROC periods must be positive integers.")

    if any(window <= 0 for window in Z_SCORE_WINDOWS):
        raise ValueError("All z-score windows must be positive integers.")

    if ANOMALY_Z_THRESHOLD <= 0:
        raise ValueError(
            "ANOMALY_Z_THRESHOLD must be greater than zero."
        )

    if EPSILON <= 0:
        raise ValueError("EPSILON must be greater than zero.")

    if LOWER_QUANTILE_CLIP < 0 or LOWER_QUANTILE_CLIP >= 1:
        raise ValueError(
            "LOWER_QUANTILE_CLIP must be between 0 and 1."
        )

    if UPPER_QUANTILE_CLIP <= 0 or UPPER_QUANTILE_CLIP > 1:
        raise ValueError(
            "UPPER_QUANTILE_CLIP must be between 0 and 1."
        )

    if LOWER_QUANTILE_CLIP >= UPPER_QUANTILE_CLIP:
        raise ValueError(
            "LOWER_QUANTILE_CLIP must be smaller than "
            "UPPER_QUANTILE_CLIP."
        )

    if REALTIME_STATE_MAX_ROWS <= 0:
        raise ValueError(
            "REALTIME_STATE_MAX_ROWS must be greater than zero."
        )


# ============================================================================
# DIRECTORY INITIALIZATION
# ============================================================================

def create_project_directories() -> None:
    """
    Create all directories required by the project.

    This function is intentionally safe to call repeatedly.
    """

    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        NOTEBOOKS_DIR,
        SRC_DIR,
        SCRIPTS_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
        MODELS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================================
# CONFIGURATION SUMMARY
# ============================================================================

def get_configuration_summary() -> Dict[str, object]:
    """
    Return the most important project configuration values.

    Returns
    -------
    dict
        Configuration summary suitable for logging or reporting.
    """

    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_name": DATASET_NAME,
        "dataset_source": DATASET_SOURCE,
        "dataset_doi": DATASET_DOI,
        "data_mode": DATA_MODE,
        "raw_data_path": str(RAW_DATA_PATH),
        "target_column": TARGET_COLUMN,
        "timestamp_column": TIMESTAMP_COLUMN,
        "machine_id_column": MACHINE_ID_COLUMN,
        "train_size": TRAIN_SIZE,
        "validation_size": VALIDATION_SIZE,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "lag_periods": list(LAG_PERIODS),
        "rolling_windows": list(ROLLING_WINDOWS),
        "ewm_spans": list(EWM_SPANS),
        "roc_periods": list(ROC_PERIODS),
        "z_score_windows": list(Z_SCORE_WINDOWS),
        "anomaly_z_threshold": ANOMALY_Z_THRESHOLD,
        "max_missing_ratio": MAX_MISSING_RATIO,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "engineered_data_path": str(ENGINEERED_DATA_PATH),
        "train_features_path": str(TRAIN_FEATURES_PATH),
        "validation_features_path": str(VALIDATION_FEATURES_PATH),
        "test_features_path": str(TEST_FEATURES_PATH),
        "html_report_path": str(HTML_REPORT_PATH),
        "docx_report_path": str(DOCX_REPORT_PATH),
    }


# ============================================================================
# MODULE INITIALIZATION
# ============================================================================

# Validate configuration as soon as the module is imported.
validate_configuration()

# Ensure the project directory structure exists.
create_project_directories()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import pprint

    print("=" * 80)
    print("PREDICTIVE MAINTENANCE PROJECT - TASK 2 CONFIGURATION")
    print("=" * 80)

    pprint.pprint(
        get_configuration_summary(),
        sort_dicts=False,
    )

    print("\nProject directories verified successfully.")