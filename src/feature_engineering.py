"""
Core feature-engineering module for predictive maintenance.

Project:
    Predictive Maintenance Project

Task:
    Task 2 - Feature Engineering

Purpose:
    Create robust, leakage-safe features for equipment-failure prediction.

Feature families implemented
-----------------------------
1. Time-based features
2. Lag features
3. Rolling mean
4. Rolling standard deviation
5. Rolling minimum
6. Rolling maximum
7. Rolling median
8. Exponentially weighted moving averages
9. Sensor differences
10. Sensor ratios
11. Temperature gap
12. Mechanical power
13. Torque × rotational-speed interaction
14. Tool-wear transformations
15. Rate-of-change features
16. Rolling z-score/anomaly features
17. Interaction features
18. Operating-condition features
19. Missingness indicators
20. Safe numerical handling
21. Constant/duplicate feature handling
22. Train/validation/test-compatible transformations

Important leakage rules
-----------------------
The following columns are never used as predictive inputs:

    Machine failure
    TWF
    HDF
    PWF
    OSF
    RNF

For genuine time-series data, lag/rolling/EWM features are generated from
past observations only.

For the AI4I dataset, UID is NOT interpreted as a genuine timestamp.

The class follows a fit/transform design:

    train_features = engineer.fit_transform(train)
    valid_features = engineer.transform(valid)
    test_features  = engineer.transform(test)

Statistics learned during fit are therefore available for later
leakage-safe preprocessing.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from .config import (
        AI4I_THRESHOLDS,
        ANOMALY_Z_THRESHOLD,
        CATEGORICAL_COLUMNS,
        CREATE_INTERACTION_FEATURES,
        CREATE_MECHANICAL_POWER,
        CREATE_SENSOR_DIFFERENCES,
        CREATE_SENSOR_RATIOS,
        CREATE_TEMPERATURE_GAP,
        CREATE_TOOL_WEAR_TRANSFORMATIONS,
        CREATE_TORQUE_SPEED_INTERACTION,
        EPSILON,
        EXCLUDED_FROM_FEATURE_GENERATION,
        EWM_SPANS,
        FEATURE_CONFIG,
        ID_COLUMNS,
        LAG_PERIODS,
        MACHINE_ID_COLUMN,
        NUMERIC_SENSOR_COLUMNS,
        REALTIME_CONFIG,
        ROC_PERIODS,
        ROLLING_MIN_PERIODS,
        ROLLING_STATISTICS,
        ROLLING_WINDOWS,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        Z_SCORE_WINDOWS,
    )
except ImportError:
    from config import (
        AI4I_THRESHOLDS,
        ANOMALY_Z_THRESHOLD,
        CATEGORICAL_COLUMNS,
        CREATE_INTERACTION_FEATURES,
        CREATE_MECHANICAL_POWER,
        CREATE_SENSOR_DIFFERENCES,
        CREATE_SENSOR_RATIOS,
        CREATE_TEMPERATURE_GAP,
        CREATE_TOOL_WEAR_TRANSFORMATIONS,
        CREATE_TORQUE_SPEED_INTERACTION,
        EPSILON,
        EXCLUDED_FROM_FEATURE_GENERATION,
        EWM_SPANS,
        FEATURE_CONFIG,
        ID_COLUMNS,
        LAG_PERIODS,
        MACHINE_ID_COLUMN,
        NUMERIC_SENSOR_COLUMNS,
        REALTIME_CONFIG,
        ROC_PERIODS,
        ROLLING_MIN_PERIODS,
        ROLLING_STATISTICS,
        ROLLING_WINDOWS,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
        Z_SCORE_WINDOWS,
    )


# ============================================================================
# LOGGING
# ============================================================================

LOGGER = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

SECONDS_PER_MINUTE = 60.0

RPM_TO_RAD_PER_SECOND = (
    2.0 * math.pi / SECONDS_PER_MINUTE
)


# ============================================================================
# AI4I FEATURE SOURCE COLUMNS
# ============================================================================

AI4I_SENSOR_COLUMNS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


# ============================================================================
# FEATURE GROUP DEFINITIONS
# ============================================================================

FEATURE_GROUP_TIME = "time"

FEATURE_GROUP_LAG = "lag"

FEATURE_GROUP_ROLLING = "rolling"

FEATURE_GROUP_EWM = "ewm"

FEATURE_GROUP_DIFFERENCE = "difference"

FEATURE_GROUP_RATIO = "ratio"

FEATURE_GROUP_PHYSICAL = "physical"

FEATURE_GROUP_TOOL_WEAR = "tool_wear"

FEATURE_GROUP_ROC = "rate_of_change"

FEATURE_GROUP_ANOMALY = "anomaly"

FEATURE_GROUP_INTERACTION = "interaction"

FEATURE_GROUP_OPERATING_CONDITION = (
    "operating_condition"
)

FEATURE_GROUP_MISSINGNESS = "missingness"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _safe_name(value: object) -> str:
    """
    Convert arbitrary text into a safe feature-name component.

    Examples
    --------
    "Air temperature [K]"
        -> "air_temperature_k"

    "Torque [Nm]"
        -> "torque_nm"
    """

    text = str(value).strip().lower()

    text = text.replace(
        "[",
        "_",
    )

    text = text.replace(
        "]",
        "",
    )

    text = text.replace(
        "×",
        "x",
    )

    text = text.replace(
        "×",
        "x",
    )

    text = text.replace(
        "²",
        "2",
    )

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    text = re.sub(
        r"_+",
        "_",
        text,
    )

    return text.strip("_")


def _ensure_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Convert a Series safely to numeric.
    """

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    epsilon: float = EPSILON,
) -> pd.Series:
    """
    Perform numerically safe element-wise division.

    Very small denominators are treated as missing rather than allowing
    extreme numerical values to be generated.
    """

    numerator = _ensure_numeric(
        numerator
    )

    denominator = _ensure_numeric(
        denominator
    )

    safe_denominator = denominator.copy()

    near_zero = (
        safe_denominator.abs()
        < epsilon
    )

    safe_denominator.loc[
        near_zero
    ] = np.nan

    result = (
        numerator
        / safe_denominator
    )

    return result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )


def _safe_log1p(
    series: pd.Series,
) -> pd.Series:
    """
    Apply log1p safely.

    Negative values below -1 become NaN.
    """

    values = _ensure_numeric(
        series
    )

    values = values.where(
        values >= -1.0,
        np.nan,
    )

    return np.log1p(
        values
    )


def _safe_sqrt(
    series: pd.Series,
) -> pd.Series:
    """
    Apply square root only to non-negative values.
    """

    values = _ensure_numeric(
        series
    )

    return np.sqrt(
        values.clip(
            lower=0
        )
    )


def _feature_name(
    group: str,
    source: str,
    operation: str,
) -> str:
    """
    Construct a standardized feature name.
    """

    return (
        f"{group}__"
        f"{_safe_name(source)}__"
        f"{_safe_name(operation)}"
    )


# ============================================================================
# FEATURE METADATA
# ============================================================================

@dataclass
class FeatureMetadata:
    """
    Metadata describing an engineered feature.
    """

    feature_name: str

    feature_group: str

    source_columns: List[str] = field(
        default_factory=list
    )

    operation: str = ""

    causal: bool = True

    leakage_safe: bool = True

    description: str = ""

    def to_dict(self) -> Dict[str, object]:
        """
        Convert metadata to a dictionary.
        """

        return {
            "feature_name": self.feature_name,
            "feature_group": self.feature_group,
            "source_columns": ", ".join(
                self.source_columns
            ),
            "operation": self.operation,
            "causal": self.causal,
            "leakage_safe": self.leakage_safe,
            "description": self.description,
        }


# ============================================================================
# FIT STATE
# ============================================================================

@dataclass
class FeatureEngineerState:
    """
    Learned state of the feature-engineering pipeline.

    This state contains only information that may safely be learned from
    the training data.
    """

    fitted: bool = False

    input_columns: List[str] = field(
        default_factory=list
    )

    numeric_columns: List[str] = field(
        default_factory=list
    )

    categorical_columns: List[str] = field(
        default_factory=list
    )

    feature_columns: List[str] = field(
        default_factory=list
    )

    median_values: Dict[str, float] = field(
        default_factory=dict
    )

    mean_values: Dict[str, float] = field(
        default_factory=dict
    )

    std_values: Dict[str, float] = field(
        default_factory=dict
    )

    constant_columns: List[str] = field(
        default_factory=list
    )

    duplicate_columns: List[str] = field(
        default_factory=list
    )


# ============================================================================
# MAIN FEATURE ENGINEER
# ============================================================================

class PredictiveMaintenanceFeatureEngineer:
    """
    Feature-engineering pipeline for predictive maintenance.

    Parameters
    ----------
    timestamp_column:
        Timestamp column for genuine time-series datasets.

    machine_id_column:
        Machine/equipment identifier.

    lags:
        Lag periods.

    rolling_windows:
        Rolling windows.

    rolling_statistics:
        Rolling statistics to generate.

    ewm_spans:
        EWM spans.

    roc_periods:
        Rate-of-change periods.

    z_score_windows:
        Rolling z-score windows.

    anomaly_threshold:
        Absolute z-score threshold for anomaly flags.

    add_missing_indicators:
        Whether to create missingness indicators.

    keep_identifiers:
        Whether identifier columns are preserved in output.

    Notes
    -----
    This class intentionally does not use target/failure-mode columns to
    generate predictors.
    """

    def __init__(
        self,
        *,
        timestamp_column: str = TIMESTAMP_COLUMN,
        machine_id_column: str = MACHINE_ID_COLUMN,
        lags: Optional[
            Sequence[int]
        ] = None,
        rolling_windows: Optional[
            Sequence[int]
        ] = None,
        rolling_statistics: Optional[
            Sequence[str]
        ] = None,
        ewm_spans: Optional[
            Sequence[int]
        ] = None,
        roc_periods: Optional[
            Sequence[int]
        ] = None,
        z_score_windows: Optional[
            Sequence[int]
        ] = None,
        anomaly_threshold: float = ANOMALY_Z_THRESHOLD,
        add_missing_indicators: bool = True,
        keep_identifiers: bool = True,
    ) -> None:

        self.timestamp_column = (
            timestamp_column
        )

        self.machine_id_column = (
            machine_id_column
        )

        self.lags = tuple(
            lags
            if lags is not None
            else LAG_PERIODS
        )

        self.rolling_windows = tuple(
            rolling_windows
            if rolling_windows is not None
            else ROLLING_WINDOWS
        )

        self.rolling_statistics = tuple(
            rolling_statistics
            if rolling_statistics is not None
            else ROLLING_STATISTICS
        )

        self.ewm_spans = tuple(
            ewm_spans
            if ewm_spans is not None
            else EWM_SPANS
        )

        self.roc_periods = tuple(
            roc_periods
            if roc_periods is not None
            else ROC_PERIODS
        )

        self.z_score_windows = tuple(
            z_score_windows
            if z_score_windows is not None
            else Z_SCORE_WINDOWS
        )

        self.anomaly_threshold = float(
            anomaly_threshold
        )

        self.add_missing_indicators = (
            add_missing_indicators
        )

        self.keep_identifiers = (
            keep_identifiers
        )

        self.state = FeatureEngineerState()

        self.metadata: List[
            FeatureMetadata
        ] = []

        self._validate_parameters()

    # ========================================================================
    # VALIDATION
    # ========================================================================

    def _validate_parameters(self) -> None:
        """
        Validate feature-engineering parameters.
        """

        if any(
            period <= 0
            for period in self.lags
        ):
            raise ValueError(
                "All lag periods must be positive."
            )

        if any(
            window <= 0
            for window in self.rolling_windows
        ):
            raise ValueError(
                "All rolling windows must be positive."
            )

        if any(
            span <= 0
            for span in self.ewm_spans
        ):
            raise ValueError(
                "All EWM spans must be positive."
            )

        if any(
            period <= 0
            for period in self.roc_periods
        ):
            raise ValueError(
                "All ROC periods must be positive."
            )

        if any(
            window <= 0
            for window in self.z_score_windows
        ):
            raise ValueError(
                "All z-score windows must be positive."
            )

        if self.anomaly_threshold <= 0:
            raise ValueError(
                "anomaly_threshold must be positive."
            )

        allowed_statistics = {
            "mean",
            "std",
            "min",
            "max",
            "median",
        }

        invalid_statistics = set(
            self.rolling_statistics
        ) - allowed_statistics

        if invalid_statistics:

            raise ValueError(
                "Unsupported rolling statistics: "
                f"{sorted(invalid_statistics)}"
            )

    # ========================================================================
    # PUBLIC API
    # ========================================================================

    def fit(
        self,
        df: pd.DataFrame,
    ) -> "PredictiveMaintenanceFeatureEngineer":
        """
        Fit the feature-engineering pipeline on training data.

        Only training data is used to learn statistics.

        Parameters
        ----------
        df:
            Training DataFrame.

        Returns
        -------
        PredictiveMaintenanceFeatureEngineer
            Fitted instance.
        """

        if df.empty:
            raise ValueError(
                "Cannot fit feature engineer on an empty DataFrame."
            )

        LOGGER.info(
            "Fitting predictive-maintenance feature engineer "
            "on %d rows and %d columns.",
            df.shape[0],
            df.shape[1],
        )

        prepared = self._prepare_input(
            df
        )

        self.state.input_columns = (
            prepared.columns.tolist()
        )

        self.state.numeric_columns = (
            prepared.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        self.state.categorical_columns = (
            prepared.select_dtypes(
                include=[
                    "object",
                    "category",
                    "string",
                ]
            )
            .columns
            .tolist()
        )

        # Learn imputation statistics only from training data.
        for column in self.state.numeric_columns:

            values = _ensure_numeric(
                prepared[column]
            )

            median = values.median()
            mean = values.mean()
            std = values.std(
                ddof=0
            )

            if pd.notna(median):
                self.state.median_values[
                    column
                ] = float(median)

            if pd.notna(mean):
                self.state.mean_values[
                    column
                ] = float(mean)

            if (
                pd.notna(std)
                and std > EPSILON
            ):
                self.state.std_values[
                    column
                ] = float(std)
            else:
                self.state.std_values[
                    column
                ] = 1.0

        # Generate a representative feature matrix in order to identify
        # constants and duplicate features.
        engineered = self._generate_features(
            prepared,
            fit_stage=True,
        )

        # Consolidate pandas blocks after bulk feature generation.
        engineered = engineered.copy()

        self.state.constant_columns = (
            self._find_constant_columns(
                engineered
            )
        )

        self.state.duplicate_columns = (
            self._find_duplicate_columns(
                engineered
            )
        )

        removable = set(
            self.state.constant_columns
            + self.state.duplicate_columns
        )

        self.state.feature_columns = [
            column
            for column in engineered.columns
            if column not in removable
        ]

        self.state.fitted = True

        LOGGER.info(
            "Feature engineer fitted successfully. "
            "Candidate features: %d; retained features: %d.",
            engineered.shape[1],
            len(
                self.state.feature_columns
            ),
        )

        return self

    def transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform data using a previously fitted pipeline.

        Parameters
        ----------
        df:
            DataFrame to transform.

        Returns
        -------
        pd.DataFrame
            Engineered DataFrame.

        Raises
        ------
        RuntimeError
            If fit() has not been called.
        """

        if not self.state.fitted:
            raise RuntimeError(
                "Feature engineer has not been fitted. "
                "Call fit() before transform()."
            )

        if df.empty:
            raise ValueError(
                "Cannot transform an empty DataFrame."
            )

        LOGGER.info(
            "Transforming %d rows.",
            len(df),
        )

        prepared = self._prepare_input(
            df
        )

        engineered = self._generate_features(
            prepared,
            fit_stage=False,
        )

        # Ensure columns expected by the fitted pipeline exist.
        # Add all missing columns in one operation to avoid DataFrame
        # fragmentation caused by repeated column insertion.
        missing_feature_columns = [
            column
            for column in self.state.feature_columns
            if column not in engineered.columns
        ]

        if missing_feature_columns:

            missing_frame = pd.DataFrame(
                np.nan,
                index=engineered.index,
                columns=missing_feature_columns,
            )

            engineered = pd.concat(
                [
                    engineered,
                    missing_frame,
                ],
                axis=1,
            )

        # Preserve the learned feature order.
        output_columns = [
            column
            for column in self.state.feature_columns
            if column in engineered.columns
        ]

        # Optionally preserve identifiers and target.
        if self.keep_identifiers:

            identifier_columns = [
                column
                for column in [
                    *ID_COLUMNS,
                    self.timestamp_column,
                    self.machine_id_column,
                    TARGET_COLUMN,
                    *EXCLUDED_FROM_FEATURE_GENERATION,
                ]
                if (
                    column in engineered.columns
                    and column not in output_columns
                )
            ]

            output_columns = (
                identifier_columns
                + output_columns
            )

        # Remove accidental duplicates while preserving order.
        output_columns = list(
            dict.fromkeys(
                output_columns
            )
        )

        result = engineered[
            output_columns
        ].copy()

        # Remove learned constant/duplicate columns.
        removable = set(
            self.state.constant_columns
            + self.state.duplicate_columns
        )

        removable.discard(
            TARGET_COLUMN
        )

        result = result.drop(
            columns=[
                column
                for column in result.columns
                if column in removable
                and column not in {
                    TARGET_COLUMN,
                    *ID_COLUMNS,
                    self.timestamp_column,
                    self.machine_id_column,
                }
            ],
            errors="ignore",
        )

        LOGGER.info(
            "Transformation completed: %d rows × %d columns.",
            result.shape[0],
            result.shape[1],
        )

        return result

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit and transform training data.

        Parameters
        ----------
        df:
            Training DataFrame.

        Returns
        -------
        pd.DataFrame
            Engineered training features.
        """

        self.fit(
            df
        )

        return self.transform(
            df
        )

    # ========================================================================
    # INPUT PREPARATION
    # ========================================================================

    def _prepare_input(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Prepare input without using target-derived information.

        Operations:
            - copy
            - numeric conversion
            - timestamp conversion
            - timestamp sorting
            - machine grouping validation
        """

        result = df.copy()

        # Prevent duplicate column names.
        if result.columns.duplicated().any():

            duplicates = (
                result.columns[
                    result.columns.duplicated(
                        keep=False
                    )
                ]
                .tolist()
            )

            raise ValueError(
                "Input contains duplicate columns: "
                f"{sorted(set(duplicates))}"
            )

        # Convert known physical sensor columns.
        for column in NUMERIC_SENSOR_COLUMNS:

            if column in result.columns:

                result[
                    column
                ] = _ensure_numeric(
                    result[column]
                )

        # Convert all obvious numerical columns that are already numeric.
        numeric_columns = (
            result.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        for column in numeric_columns:

            result[
                column
            ] = _ensure_numeric(
                result[column]
            )

        # Timestamp conversion.
        if self.timestamp_column in result.columns:

            result[
                self.timestamp_column
            ] = pd.to_datetime(
                result[
                    self.timestamp_column
                ],
                errors="coerce",
                utc=True,
            )

            sort_columns = []

            if (
                self.machine_id_column
                in result.columns
            ):
                sort_columns.append(
                    self.machine_id_column
                )

            sort_columns.append(
                self.timestamp_column
            )

            result = result.sort_values(
                sort_columns,
                kind="mergesort",
            )

            result = result.reset_index(
                drop=True
            )

        return result

    # ========================================================================
    # CORE FEATURE GENERATION
    # ========================================================================

    def _generate_features(
        self,
        df: pd.DataFrame,
        *,
        fit_stage: bool,
    ) -> pd.DataFrame:
        """
        Generate all engineered features.

        Parameters
        ----------
        df:
            Prepared input.

        fit_stage:
            Whether this call is being performed while fitting.
        """

        result = df.copy()

        # Clear metadata only during a fresh fit.
        if fit_stage:
            self.metadata = []

        # ---------------------------------------------------------------
        # Missingness features
        # ---------------------------------------------------------------

        if self.add_missing_indicators:

            result = (
                self._add_missingness_features(
                    result
                )
            )

        # ---------------------------------------------------------------
        # Time features
        # ---------------------------------------------------------------

        result = (
            self._add_time_features(
                result
            )
        )

        # ---------------------------------------------------------------
        # Physical/process features
        # ---------------------------------------------------------------

        result = (
            self._add_physical_features(
                result
            )
        )

        # ---------------------------------------------------------------
        # Tool-wear transformations
        # ---------------------------------------------------------------

        if CREATE_TOOL_WEAR_TRANSFORMATIONS:

            result = (
                self._add_tool_wear_features(
                    result
                )
            )

        # ---------------------------------------------------------------
        # Sensor differences
        # ---------------------------------------------------------------

        if CREATE_SENSOR_DIFFERENCES:

            result = (
                self._add_sensor_difference_features(
                    result
                )
            )

        # ---------------------------------------------------------------
        # Sensor ratios
        # ---------------------------------------------------------------

        if CREATE_SENSOR_RATIOS:

            result = (
                self._add_sensor_ratio_features(
                    result
                )
            )

        # ---------------------------------------------------------------
        # Interaction features
        # ---------------------------------------------------------------

        if CREATE_INTERACTION_FEATURES:

            result = (
                self._add_interaction_features(
                    result
                )
            )

        # ---------------------------------------------------------------
        # Operating conditions
        # ---------------------------------------------------------------

        result = (
            self._add_operating_condition_features(
                result
            )
        )

        # ---------------------------------------------------------------
        # Time-series features
        # ---------------------------------------------------------------

        # Only genuine timestamp-based data gets explicit temporal
        # features. AI4I UID is intentionally excluded.
        if self._has_true_timestamp(
            result
        ):

            result = (
                self._add_lag_features(
                    result
                )
            )

            result = (
                self._add_rolling_features(
                    result
                )
            )

            result = (
                self._add_ewm_features(
                    result
                )
            )

            result = (
                self._add_rate_of_change_features(
                    result
                )
            )

            result = (
                self._add_anomaly_features(
                    result
                )
            )

        # ---------------------------------------------------------------
        # Learned numeric imputation
        # ---------------------------------------------------------------

        if not fit_stage:

            result = (
                self._apply_learned_numeric_imputation(
                    result
                )
            )

        else:

            # During fitting, we do not alter raw values before generating
            # the representative matrix. Missing-value handling for model
            # features is learned from the training partition.
            #
            # We still fill engineered infinities.
            result = (
                self._replace_invalid_values(
                    result
                )
            )

        return result

    # ========================================================================
    # TIME FEATURES
    # ========================================================================

    def _add_time_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add calendar/time-based features.

        Only generated when a genuine timestamp exists.
        """

        result = df.copy()

        if not self._has_true_timestamp(
            result
        ):
            return result

        timestamp = result[
            self.timestamp_column
        ]

        prefix = "time"

        features: Dict[
            str,
            pd.Series,
        ] = {}

        features[
            f"{prefix}__hour"
        ] = timestamp.dt.hour.astype(
            "float64"
        )

        features[
            f"{prefix}__day_of_week"
        ] = timestamp.dt.dayofweek.astype(
            "float64"
        )

        features[
            f"{prefix}__day_of_month"
        ] = timestamp.dt.day.astype(
            "float64"
        )

        features[
            f"{prefix}__day_of_year"
        ] = timestamp.dt.dayofyear.astype(
            "float64"
        )

        features[
            f"{prefix}__week_of_year"
        ] = (
            timestamp.dt.isocalendar()
            .week
            .astype("float64")
        )

        features[
            f"{prefix}__month"
        ] = timestamp.dt.month.astype(
            "float64"
        )

        features[
            f"{prefix}__quarter"
        ] = timestamp.dt.quarter.astype(
            "float64"
        )

        features[
            f"{prefix}__is_weekend"
        ] = (
            timestamp.dt.dayofweek
            >= 5
        ).astype("int8")

        features[
            f"{prefix}__is_month_start"
        ] = (
            timestamp.dt.is_month_start
        ).astype("int8")

        features[
            f"{prefix}__is_month_end"
        ] = (
            timestamp.dt.is_month_end
        ).astype("int8")

        features[
            f"{prefix}__is_quarter_start"
        ] = (
            timestamp.dt.is_quarter_start
        ).astype("int8")

        features[
            f"{prefix}__is_quarter_end"
        ] = (
            timestamp.dt.is_quarter_end
        ).astype("int8")

        # Cyclical encoding prevents the model from interpreting 23:00 and
        # 00:00 as maximally distant.
        features[
            f"{prefix}__hour_sin"
        ] = np.sin(
            2.0
            * np.pi
            * features[
                f"{prefix}__hour"
            ]
            / 24.0
        )

        features[
            f"{prefix}__hour_cos"
        ] = np.cos(
            2.0
            * np.pi
            * features[
                f"{prefix}__hour"
            ]
            / 24.0
        )

        features[
            f"{prefix}__day_of_week_sin"
        ] = np.sin(
            2.0
            * np.pi
            * features[
                f"{prefix}__day_of_week"
            ]
            / 7.0
        )

        features[
            f"{prefix}__day_of_week_cos"
        ] = np.cos(
            2.0
            * np.pi
            * features[
                f"{prefix}__day_of_week"
            ]
            / 7.0
        )

        features[
            f"{prefix}__month_sin"
        ] = np.sin(
            2.0
            * np.pi
            * features[
                f"{prefix}__month"
            ]
            / 12.0
        )

        features[
            f"{prefix}__month_cos"
        ] = np.cos(
            2.0
            * np.pi
            * features[
                f"{prefix}__month"
            ]
            / 12.0
        )

        for name, values in features.items():

            result[name] = values

            self._register_feature(
                name,
                FEATURE_GROUP_TIME,
                [
                    self.timestamp_column
                ],
                "calendar/cyclical time extraction",
                causal=True,
                description=(
                    "Timestamp-derived operating-time feature."
                ),
            )

        # Elapsed time from the first observed timestamp.
        if len(timestamp) > 0:

            first_timestamp = timestamp.iloc[
                0
            ]

            if pd.notna(first_timestamp):

                elapsed_seconds = (
                    timestamp
                    - first_timestamp
                ).dt.total_seconds()

                feature_name = (
                    "time__elapsed_seconds"
                )

                result[
                    feature_name
                ] = elapsed_seconds

                self._register_feature(
                    feature_name,
                    FEATURE_GROUP_TIME,
                    [
                        self.timestamp_column
                    ],
                    "elapsed time",
                    causal=True,
                    description=(
                        "Elapsed time since the beginning "
                        "of the supplied sequence."
                    ),
                )

        return result

    # ========================================================================
    # PHYSICAL FEATURES
    # ========================================================================

    def _add_physical_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add physically meaningful process features.
        """

        result = df.copy()

        air = self._get_numeric(
            result,
            "Air temperature [K]",
        )

        process = self._get_numeric(
            result,
            "Process temperature [K]",
        )

        rpm = self._get_numeric(
            result,
            "Rotational speed [rpm]",
        )

        torque = self._get_numeric(
            result,
            "Torque [Nm]",
        )

        if (
            CREATE_TEMPERATURE_GAP
            and air is not None
            and process is not None
        ):

            feature_name = (
                "physical__temperature_gap_k"
            )

            result[
                feature_name
            ] = process - air

            self._register_feature(
                feature_name,
                FEATURE_GROUP_PHYSICAL,
                [
                    "Process temperature [K]",
                    "Air temperature [K]",
                ],
                "process temperature - air temperature",
                causal=True,
                description=(
                    "Temperature gap between process and "
                    "surrounding air."
                ),
            )

        if (
            CREATE_MECHANICAL_POWER
            and rpm is not None
            and torque is not None
        ):

            angular_velocity = (
                rpm
                * RPM_TO_RAD_PER_SECOND
            )

            feature_name = (
                "physical__mechanical_power_w"
            )

            result[
                feature_name
            ] = torque * angular_velocity

            self._register_feature(
                feature_name,
                FEATURE_GROUP_PHYSICAL,
                [
                    "Torque [Nm]",
                    "Rotational speed [rpm]",
                ],
                "torque × angular velocity",
                causal=True,
                description=(
                    "Mechanical power calculated from torque "
                    "and rotational speed."
                ),
            )

            # Power in kW is easier to interpret.
            kw_name = (
                "physical__mechanical_power_kw"
            )

            result[
                kw_name
            ] = (
                result[
                    feature_name
                ]
                / 1000.0
            )

            self._register_feature(
                kw_name,
                FEATURE_GROUP_PHYSICAL,
                [
                    feature_name
                ],
                "mechanical power / 1000",
                causal=True,
                description=(
                    "Mechanical power expressed in kilowatts."
                ),
            )

        if (
            CREATE_TORQUE_SPEED_INTERACTION
            and rpm is not None
            and torque is not None
        ):

            feature_name = (
                "physical__torque_speed_interaction"
            )

            result[
                feature_name
            ] = torque * rpm

            self._register_feature(
                feature_name,
                FEATURE_GROUP_INTERACTION,
                [
                    "Torque [Nm]",
                    "Rotational speed [rpm]",
                ],
                "torque × RPM",
                causal=True,
                description=(
                    "Interaction between applied torque and "
                    "rotational speed."
                ),
            )

        # Torque-to-speed ratio.
        if (
            rpm is not None
            and torque is not None
        ):

            feature_name = (
                "physical__torque_per_rpm"
            )

            result[
                feature_name
            ] = _safe_divide(
                torque,
                rpm,
            )

            self._register_feature(
                feature_name,
                FEATURE_GROUP_RATIO,
                [
                    "Torque [Nm]",
                    "Rotational speed [rpm]",
                ],
                "torque / RPM",
                causal=True,
                description=(
                    "Torque relative to rotational speed."
                ),
            )

        # Angular velocity itself.
        if rpm is not None:

            feature_name = (
                "physical__angular_velocity_rad_s"
            )

            result[
                feature_name
            ] = (
                rpm
                * RPM_TO_RAD_PER_SECOND
            )

            self._register_feature(
                feature_name,
                FEATURE_GROUP_PHYSICAL,
                [
                    "Rotational speed [rpm]"
                ],
                "RPM × 2π / 60",
                causal=True,
                description=(
                    "Rotational speed converted to radians per second."
                ),
            )

        return result

    # ========================================================================
    # TOOL WEAR FEATURES
    # ========================================================================

    def _add_tool_wear_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add nonlinear tool-wear transformations.
        """

        result = df.copy()

        if "Tool wear [min]" not in result.columns:
            return result

        wear = self._get_numeric(
            result,
            "Tool wear [min]",
        )

        if wear is None:
            return result

        transformations = {
            "tool_wear__square": wear ** 2,
            "tool_wear__sqrt": _safe_sqrt(wear),
            "tool_wear__log1p": _safe_log1p(wear),
            "tool_wear__cumulative_normalized": (
                _safe_divide(
                    wear,
                    pd.Series(
                        np.maximum(
                            wear.max(),
                            1.0
                        ),
                        index=wear.index,
                    ),
                )
            ),
        }

        for name, values in transformations.items():

            result[name] = values

            self._register_feature(
                name,
                FEATURE_GROUP_TOOL_WEAR,
                [
                    "Tool wear [min]"
                ],
                name.split(
                    "__"
                )[-1],
                causal=True,
                description=(
                    "Nonlinear transformation of tool wear."
                ),
            )

        # Tool wear threshold proximity.
        threshold = (
            AI4I_THRESHOLDS.get(
                "tool_wear_failure_lower_min",
                200.0,
            )
        )

        feature_name = (
            "tool_wear__distance_to_200_min"
        )

        result[
            feature_name
        ] = wear - threshold

        self._register_feature(
            feature_name,
            FEATURE_GROUP_TOOL_WEAR,
            [
                "Tool wear [min]"
            ],
            "wear - threshold",
            causal=True,
            description=(
                "Distance from the documented 200-minute "
                "tool-wear condition."
            ),
        )

        return result

    # ========================================================================
    # SENSOR DIFFERENCES
    # ========================================================================

    def _add_sensor_difference_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add physically interpretable sensor differences.
        """

        result = df.copy()

        pairs = [
            (
                "Process temperature [K]",
                "Air temperature [K]",
                "process_minus_air_temperature",
            ),
        ]

        for left, right, suffix in pairs:

            if (
                left not in result.columns
                or right not in result.columns
            ):
                continue

            left_values = self._get_numeric(
                result,
                left,
            )

            right_values = self._get_numeric(
                result,
                right,
            )

            if (
                left_values is None
                or right_values is None
            ):
                continue

            name = (
                f"difference__{suffix}"
            )

            result[name] = (
                left_values
                - right_values
            )

            self._register_feature(
                name,
                FEATURE_GROUP_DIFFERENCE,
                [
                    left,
                    right,
                ],
                f"{left} - {right}",
                causal=True,
                description=(
                    "Difference between related process sensors."
                ),
            )

        return result

    # ========================================================================
    # SENSOR RATIOS
    # ========================================================================

    def _add_sensor_ratio_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add safe ratios between physically related variables.
        """

        result = df.copy()

        ratios = [
            (
                "Process temperature [K]",
                "Air temperature [K]",
                "process_to_air_temperature_ratio",
            ),
            (
                "Torque [Nm]",
                "Rotational speed [rpm]",
                "torque_to_speed_ratio",
            ),
        ]

        for numerator_column, denominator_column, suffix in ratios:

            if (
                numerator_column not in result.columns
                or denominator_column not in result.columns
            ):
                continue

            numerator = self._get_numeric(
                result,
                numerator_column,
            )

            denominator = self._get_numeric(
                result,
                denominator_column,
            )

            if (
                numerator is None
                or denominator is None
            ):
                continue

            name = (
                f"ratio__{suffix}"
            )

            result[name] = _safe_divide(
                numerator,
                denominator,
            )

            self._register_feature(
                name,
                FEATURE_GROUP_RATIO,
                [
                    numerator_column,
                    denominator_column,
                ],
                f"{numerator_column} / {denominator_column}",
                causal=True,
                description=(
                    "Safe ratio between process measurements."
                ),
            )

        return result

    # ========================================================================
    # INTERACTIONS
    # ========================================================================

    def _add_interaction_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add nonlinear sensor interactions.
        """

        result = df.copy()

        columns = {
            "air": "Air temperature [K]",
            "process": "Process temperature [K]",
            "rpm": "Rotational speed [rpm]",
            "torque": "Torque [Nm]",
            "wear": "Tool wear [min]",
        }

        available = {
            key: self._get_numeric(
                result,
                column,
            )
            for key, column in columns.items()
            if column in result.columns
        }

        # Temperature × tool wear.
        if (
            "process" in available
            and "wear" in available
        ):

            name = (
                "interaction__process_temperature_x_tool_wear"
            )

            result[name] = (
                available["process"]
                * available["wear"]
            )

            self._register_feature(
                name,
                FEATURE_GROUP_INTERACTION,
                [
                    columns["process"],
                    columns["wear"],
                ],
                "process temperature × tool wear",
                causal=True,
                description=(
                    "Interaction between process temperature "
                    "and tool wear."
                ),
            )

        # Torque × tool wear.
        if (
            "torque" in available
            and "wear" in available
        ):

            name = (
                "interaction__torque_x_tool_wear"
            )

            result[name] = (
                available["torque"]
                * available["wear"]
            )

            self._register_feature(
                name,
                FEATURE_GROUP_INTERACTION,
                [
                    columns["torque"],
                    columns["wear"],
                ],
                "torque × tool wear",
                causal=True,
                description=(
                    "Interaction between mechanical load and tool wear."
                ),
            )

        # RPM × tool wear.
        if (
            "rpm" in available
            and "wear" in available
        ):

            name = (
                "interaction__rpm_x_tool_wear"
            )

            result[name] = (
                available["rpm"]
                * available["wear"]
            )

            self._register_feature(
                name,
                FEATURE_GROUP_INTERACTION,
                [
                    columns["rpm"],
                    columns["wear"],
                ],
                "RPM × tool wear",
                causal=True,
                description=(
                    "Interaction between rotational speed and tool wear."
                ),
            )

        # Temperature gap × torque.
        if (
            "process" in available
            and "air" in available
            and "torque" in available
        ):

            temperature_gap = (
                available["process"]
                - available["air"]
            )

            name = (
                "interaction__temperature_gap_x_torque"
            )

            result[name] = (
                temperature_gap
                * available["torque"]
            )

            self._register_feature(
                name,
                FEATURE_GROUP_INTERACTION,
                [
                    columns["process"],
                    columns["air"],
                    columns["torque"],
                ],
                "temperature gap × torque",
                causal=True,
                description=(
                    "Thermal-mechanical interaction."
                ),
            )

        return result

    # ========================================================================
    # OPERATING CONDITIONS
    # ========================================================================

    def _add_operating_condition_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add operating-condition indicators.

        These features use documented engineering thresholds and raw sensor
        values only. They do not use Machine failure or failure-mode labels.
        """

        result = df.copy()

        rpm = self._get_numeric(
            result,
            "Rotational speed [rpm]",
        )

        torque = self._get_numeric(
            result,
            "Torque [Nm]",
        )

        wear = self._get_numeric(
            result,
            "Tool wear [min]",
        )

        process = self._get_numeric(
            result,
            "Process temperature [K]",
        )

        air = self._get_numeric(
            result,
            "Air temperature [K]",
        )

        # ---------------------------------------------------------------
        # Low/high RPM
        # ---------------------------------------------------------------

        if rpm is not None:

            name = (
                "operating_condition__low_rpm_flag"
            )

            result[name] = (
                rpm
                < AI4I_THRESHOLDS[
                    "heat_rotational_speed_failure_threshold_rpm"
                ]
            ).astype("int8")

            self._register_feature(
                name,
                FEATURE_GROUP_OPERATING_CONDITION,
                [
                    "Rotational speed [rpm]"
                ],
                "RPM below documented threshold",
                causal=True,
                description=(
                    "Indicator for low rotational speed."
                ),
            )

        # ---------------------------------------------------------------
        # Temperature gap condition
        # ---------------------------------------------------------------

        if (
            process is not None
            and air is not None
        ):

            gap = (
                process - air
            )

            name = (
                "operating_condition__small_temperature_gap_flag"
            )

            result[name] = (
                gap
                < AI4I_THRESHOLDS[
                    "heat_temperature_gap_failure_threshold_k"
                ]
            ).astype("int8")

            self._register_feature(
                name,
                FEATURE_GROUP_OPERATING_CONDITION,
                [
                    "Process temperature [K]",
                    "Air temperature [K]",
                ],
                "temperature gap below threshold",
                causal=True,
                description=(
                    "Indicator for unusually small process-to-air "
                    "temperature gap."
                ),
            )

        # ---------------------------------------------------------------
        # Tool wear condition
        # ---------------------------------------------------------------

        if wear is not None:

            lower = AI4I_THRESHOLDS[
                "tool_wear_failure_lower_min"
            ]

            upper = AI4I_THRESHOLDS[
                "tool_wear_failure_upper_min"
            ]

            name = (
                "operating_condition__high_tool_wear_flag"
            )

            result[name] = (
                wear >= lower
            ).astype("int8")

            self._register_feature(
                name,
                FEATURE_GROUP_OPERATING_CONDITION,
                [
                    "Tool wear [min]"
                ],
                "tool wear >= threshold",
                causal=True,
                description=(
                    "Indicator that tool wear has reached "
                    "the documented high-wear region."
                ),
            )

            name = (
                "operating_condition__critical_tool_wear_window_flag"
            )

            result[name] = (
                (
                    wear >= lower
                )
                &
                (
                    wear <= upper
                )
            ).astype("int8")

            self._register_feature(
                name,
                FEATURE_GROUP_OPERATING_CONDITION,
                [
                    "Tool wear [min]"
                ],
                "tool wear within documented critical window",
                causal=True,
                description=(
                    "Indicator for the documented 200–240 minute "
                    "tool-wear interval."
                ),
            )

        # ---------------------------------------------------------------
        # Mechanical power condition
        # ---------------------------------------------------------------

        if (
            rpm is not None
            and torque is not None
        ):

            power = (
                torque
                * rpm
                * RPM_TO_RAD_PER_SECOND
            )

            lower = AI4I_THRESHOLDS[
                "power_failure_lower_w"
            ]

            upper = AI4I_THRESHOLDS[
                "power_failure_upper_w"
            ]

            name = (
                "operating_condition__power_outside_range_flag"
            )

            result[name] = (
                (
                    power < lower
                )
                |
                (
                    power > upper
                )
            ).astype("int8")

            self._register_feature(
                name,
                FEATURE_GROUP_OPERATING_CONDITION,
                [
                    "Torque [Nm]",
                    "Rotational speed [rpm]",
                ],
                "mechanical power outside documented range",
                causal=True,
                description=(
                    "Indicator for mechanical power outside "
                    "the documented reference range."
                ),
            )

        return result

    # ========================================================================
    # MISSINGNESS
    # ========================================================================

    def _add_missingness_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add missing-value indicators.

        Missingness can itself carry useful equipment-state information,
        especially in real sensor systems.
        """

        result = df.copy()

        excluded = set(
            EXCLUDED_FROM_FEATURE_GENERATION
        )

        # Do not create target missingness features.
        candidate_columns = [
            column
            for column in result.columns
            if column not in excluded
        ]

        for column in candidate_columns:

            if (
                column
                in {
                    self.timestamp_column,
                    self.machine_id_column,
                }
            ):
                continue

            if not (
                pd.api.types.is_numeric_dtype(
                    result[column]
                )
                or
                pd.api.types.is_object_dtype(
                    result[column]
                )
                or
                pd.api.types.is_categorical_dtype(
                    result[column]
                )
            ):
                continue

            name = (
                f"missingness__"
                f"{_safe_name(column)}"
            )

            result[name] = (
                result[column].isna()
            ).astype("int8")

            self._register_feature(
                name,
                FEATURE_GROUP_MISSINGNESS,
                [
                    column
                ],
                "is_missing",
                causal=True,
                description=(
                    "Indicator that the source sensor value is missing."
                ),
            )

        # Overall sensor missing count.
        sensor_candidates = [
            column
            for column in NUMERIC_SENSOR_COLUMNS
            if column in result.columns
        ]

        if sensor_candidates:

            name = (
                "missingness__sensor_missing_count"
            )

            result[name] = (
                result[
                    sensor_candidates
                ]
                .isna()
                .sum(axis=1)
            )

            self._register_feature(
                name,
                FEATURE_GROUP_MISSINGNESS,
                sensor_candidates,
                "row-wise missing sensor count",
                causal=True,
                description=(
                    "Number of missing physical sensor readings "
                    "in the observation."
                ),
            )

        return result

    # ========================================================================
    # TIME-SERIES GROUP HANDLING
    # ========================================================================

    def _has_true_timestamp(
        self,
        df: pd.DataFrame,
    ) -> bool:
        """
        Determine whether a genuine timestamp exists.

        UID is deliberately not accepted.
        """

        if (
            self.timestamp_column
            not in df.columns
        ):
            return False

        if (
            self.timestamp_column
            == "UID"
        ):
            return False

        values = pd.to_datetime(
            df[
                self.timestamp_column
            ],
            errors="coerce",
            utc=True,
        )

        return (
            values.notna().mean()
            >= 0.95
        )

    def _grouped_series(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> Iterable[
        Tuple[
            object,
            pd.Series,
        ]
    ]:
        """
        Yield machine-specific Series.

        If no machine_id is available, the entire DataFrame is one sequence.
        """

        if (
            self.machine_id_column
            in df.columns
        ):

            for machine_id, group in (
                df.groupby(
                    self.machine_id_column,
                    sort=False,
                    dropna=False,
                )
            ):

                yield machine_id, group[
                    column
                ]

        else:

            yield "__all__", df[
                column
            ]

    # ========================================================================
    # LAG FEATURES
    # ========================================================================

    def _add_lag_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add causal lag features.

        A lag of one means the immediately preceding observation for the
        same machine.

        The current observation is never included.
        """

        result = df.copy()

        source_columns = self._select_temporal_sensor_columns(
            result
        )

        if not source_columns:
            return result

        for column in source_columns:

            safe_source = _safe_name(
                column
            )

            for lag in self.lags:

                name = (
                    f"{FEATURE_GROUP_LAG}__"
                    f"{safe_source}__lag_{lag}"
                )

                result[name] = np.nan

                if (
                    self.machine_id_column
                    in result.columns
                ):

                    shifted = (
                        result.groupby(
                            self.machine_id_column,
                            sort=False,
                            dropna=False,
                        )[column]
                        .shift(lag)
                    )

                    result[name] = shifted

                else:

                    result[name] = (
                        result[column]
                        .shift(lag)
                    )

                self._register_feature(
                    name,
                    FEATURE_GROUP_LAG,
                    [
                        column
                    ],
                    f"lag {lag}",
                    causal=True,
                    description=(
                        f"Previous {lag}-step value of "
                        f"{column}."
                    ),
                )

        return result

    # ========================================================================
    # ROLLING FEATURES
    # ========================================================================

    def _add_rolling_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add causal rolling statistics.

        IMPORTANT:
            The current observation is excluded from the rolling window.

        This avoids a subtle but important form of target leakage when the
        current sensor state itself is highly related to the target.
        """

        result = df.copy()

        source_columns = self._select_temporal_sensor_columns(
            result
        )

        if not source_columns:
            return result

        for column in source_columns:

            safe_source = _safe_name(
                column
            )

            for window in self.rolling_windows:

                if (
                    self.machine_id_column
                    in result.columns
                ):

                    # The grouped helper preserves row order and calculates
                    # every requested statistic from observations strictly before
                    # the current row.
                    result = self._add_grouped_rolling_columns(
                        result,
                        column,
                        safe_source,
                        window,
                    )

                else:

                    historical = (
                        result[column]
                        .shift(1)
                    )

                    rolling = (
                        historical.rolling(
                            window=window,
                            min_periods=ROLLING_MIN_PERIODS,
                        )
                    )

                    for statistic in self.rolling_statistics:

                        name = (
                            f"{FEATURE_GROUP_ROLLING}__"
                            f"{safe_source}__"
                            f"{statistic}_{window}"
                        )

                        if statistic == "mean":
                            values = rolling.mean()

                        elif statistic == "std":
                            values = rolling.std(
                                ddof=0
                            )

                        elif statistic == "min":
                            values = rolling.min()

                        elif statistic == "max":
                            values = rolling.max()

                        elif statistic == "median":
                            values = rolling.median()

                        else:
                            continue

                        result[name] = values

                        self._register_feature(
                            name,
                            FEATURE_GROUP_ROLLING,
                            [
                                column
                            ],
                            (
                                f"historical rolling "
                                f"{statistic}, window={window}"
                            ),
                            causal=True,
                            description=(
                                "Historical rolling statistic excluding "
                                "the current observation."
                            ),
                        )

        return result

    def _add_grouped_rolling_columns(
        self,
        df: pd.DataFrame,
        column: str,
        safe_source: str,
        window: int,
    ) -> pd.DataFrame:
        """
        Add machine-specific rolling features while preserving row order.
        """

        result = df.copy()

        machine_groups = (
            result[
                self.machine_id_column
            ]
            .groupby(
                result[
                    self.machine_id_column
                ],
                sort=False,
                dropna=False,
            )
        )

        # Create a result Series indexed exactly like the DataFrame.
        for statistic in self.rolling_statistics:

            name = (
                f"{FEATURE_GROUP_ROLLING}__"
                f"{safe_source}__"
                f"{statistic}_{window}"
            )

            output = pd.Series(
                np.nan,
                index=result.index,
                dtype="float64",
            )

            for _, indices in machine_groups.groups.items():

                group_indices = list(
                    indices
                )

                values = result.loc[
                    group_indices,
                    column,
                ]

                historical = (
                    values.shift(1)
                )

                rolling = historical.rolling(
                    window=window,
                    min_periods=ROLLING_MIN_PERIODS,
                )

                if statistic == "mean":
                    calculated = rolling.mean()

                elif statistic == "std":
                    calculated = rolling.std(
                        ddof=0
                    )

                elif statistic == "min":
                    calculated = rolling.min()

                elif statistic == "max":
                    calculated = rolling.max()

                elif statistic == "median":
                    calculated = rolling.median()

                else:
                    continue

                output.loc[
                    group_indices
                ] = calculated.to_numpy()

            result[name] = output

            self._register_feature(
                name,
                FEATURE_GROUP_ROLLING,
                [
                    column,
                    self.machine_id_column,
                ],
                (
                    f"machine-specific historical rolling "
                    f"{statistic}, window={window}"
                ),
                causal=True,
                description=(
                    "Machine-specific rolling statistic using only "
                    "previous observations."
                ),
            )

        return result

    # ========================================================================
    # EWM FEATURES
    # ========================================================================

    def _add_ewm_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add historical exponentially weighted moving averages.

        The current observation is excluded.
        """

        result = df.copy()

        source_columns = self._select_temporal_sensor_columns(
            result
        )

        for column in source_columns:

            safe_source = _safe_name(
                column
            )

            for span in self.ewm_spans:

                name = (
                    f"{FEATURE_GROUP_EWM}__"
                    f"{safe_source}__span_{span}"
                )

                output = pd.Series(
                    np.nan,
                    index=result.index,
                    dtype="float64",
                )

                if (
                    self.machine_id_column
                    in result.columns
                ):

                    grouped = (
                        result.groupby(
                            self.machine_id_column,
                            sort=False,
                            dropna=False,
                        )
                    )

                    for _, group in grouped:

                        indices = group.index

                        historical = (
                            group[column]
                            .shift(1)
                        )

                        values = (
                            historical
                            .ewm(
                                span=span,
                                adjust=False,
                                min_periods=1,
                            )
                            .mean()
                        )

                        output.loc[
                            indices
                        ] = values.to_numpy()

                else:

                    historical = (
                        result[column]
                        .shift(1)
                    )

                    output = (
                        historical
                        .ewm(
                            span=span,
                            adjust=False,
                            min_periods=1,
                        )
                        .mean()
                    )

                result[name] = output

                self._register_feature(
                    name,
                    FEATURE_GROUP_EWM,
                    [
                        column
                    ],
                    f"historical EWM span={span}",
                    causal=True,
                    description=(
                        "Exponentially weighted historical average "
                        "excluding the current observation."
                    ),
                )

        return result

    # ========================================================================
    # RATE OF CHANGE
    # ========================================================================

    def _add_rate_of_change_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add percentage-change and absolute-change features.

        These use current versus historical values and therefore are available
        at prediction time when the current sensor observation is known.
        """

        result = df.copy()

        source_columns = self._select_temporal_sensor_columns(
            result
        )

        for column in source_columns:

            safe_source = _safe_name(
                column
            )

            for period in self.roc_periods:

                if (
                    self.machine_id_column
                    in result.columns
                ):

                    previous = (
                        result.groupby(
                            self.machine_id_column,
                            sort=False,
                            dropna=False,
                        )[column]
                        .shift(period)
                    )

                else:

                    previous = (
                        result[column]
                        .shift(period)
                    )

                current = result[
                    column
                ]

                difference_name = (
                    f"{FEATURE_GROUP_ROC}__"
                    f"{safe_source}__"
                    f"absolute_change_{period}"
                )

                result[
                    difference_name
                ] = (
                    current
                    - previous
                )

                self._register_feature(
                    difference_name,
                    FEATURE_GROUP_ROC,
                    [
                        column
                    ],
                    f"current - lag {period}",
                    causal=True,
                    description=(
                        "Absolute change from a previous sensor reading."
                    ),
                )

                ratio_name = (
                    f"{FEATURE_GROUP_ROC}__"
                    f"{safe_source}__"
                    f"pct_change_{period}"
                )

                result[
                    ratio_name
                ] = _safe_divide(
                    current - previous,
                    previous.abs(),
                )

                self._register_feature(
                    ratio_name,
                    FEATURE_GROUP_ROC,
                    [
                        column
                    ],
                    f"(current - lag {period}) / abs(lag {period})",
                    causal=True,
                    description=(
                        "Relative rate of change from a previous sensor "
                        "reading."
                    ),
                )

        return result

    # ========================================================================
    # ANOMALY / Z-SCORE FEATURES
    # ========================================================================

    def _add_anomaly_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add historical rolling z-score and anomaly flags.

        Baseline mean/std are computed only from previous observations.
        """

        result = df.copy()

        source_columns = self._select_temporal_sensor_columns(
            result
        )

        for column in source_columns:

            safe_source = _safe_name(
                column
            )

            for window in self.z_score_windows:

                mean_name = (
                    f"anomaly__"
                    f"{safe_source}__"
                    f"historical_mean_{window}"
                )

                std_name = (
                    f"anomaly__"
                    f"{safe_source}__"
                    f"historical_std_{window}"
                )

                z_name = (
                    f"anomaly__"
                    f"{safe_source}__"
                    f"zscore_{window}"
                )

                flag_name = (
                    f"anomaly__"
                    f"{safe_source}__"
                    f"flag_{window}"
                )

                output_mean = pd.Series(
                    np.nan,
                    index=result.index,
                    dtype="float64",
                )

                output_std = pd.Series(
                    np.nan,
                    index=result.index,
                    dtype="float64",
                )

                if (
                    self.machine_id_column
                    in result.columns
                ):

                    grouped = result.groupby(
                        self.machine_id_column,
                        sort=False,
                        dropna=False,
                    )

                    for _, group in grouped:

                        indices = group.index

                        historical = (
                            group[column]
                            .shift(1)
                        )

                        rolling = historical.rolling(
                            window=window,
                            min_periods=ROLLING_MIN_PERIODS,
                        )

                        output_mean.loc[
                            indices
                        ] = (
                            rolling.mean()
                            .to_numpy()
                        )

                        output_std.loc[
                            indices
                        ] = (
                            rolling.std(
                                ddof=0
                            )
                            .to_numpy()
                        )

                else:

                    historical = (
                        result[column]
                        .shift(1)
                    )

                    rolling = historical.rolling(
                        window=window,
                        min_periods=ROLLING_MIN_PERIODS,
                    )

                    output_mean = (
                        rolling.mean()
                    )

                    output_std = (
                        rolling.std(
                            ddof=0
                        )
                    )

                result[
                    mean_name
                ] = output_mean

                result[
                    std_name
                ] = output_std

                result[
                    z_name
                ] = _safe_divide(
                    result[column]
                    - output_mean,
                    output_std,
                )

                result[
                    flag_name
                ] = (
                    result[
                        z_name
                    ].abs()
                    >= self.anomaly_threshold
                ).astype("int8")

                self._register_feature(
                    mean_name,
                    FEATURE_GROUP_ANOMALY,
                    [
                        column
                    ],
                    f"historical mean, window={window}",
                    causal=True,
                    description=(
                        "Historical rolling mean used as anomaly baseline."
                    ),
                )

                self._register_feature(
                    std_name,
                    FEATURE_GROUP_ANOMALY,
                    [
                        column
                    ],
                    f"historical standard deviation, window={window}",
                    causal=True,
                    description=(
                        "Historical rolling standard deviation used as "
                        "anomaly baseline."
                    ),
                )

                self._register_feature(
                    z_name,
                    FEATURE_GROUP_ANOMALY,
                    [
                        column
                    ],
                    f"historical rolling z-score, window={window}",
                    causal=True,
                    description=(
                        "Current sensor value relative to its historical "
                        "rolling baseline."
                    ),
                )

                self._register_feature(
                    flag_name,
                    FEATURE_GROUP_ANOMALY,
                    [
                        column
                    ],
                    (
                        f"absolute z-score >= "
                        f"{self.anomaly_threshold}"
                    ),
                    causal=True,
                    description=(
                        "Historical z-score anomaly indicator."
                    ),
                )

        return result

    # ========================================================================
    # SENSOR SELECTION
    # ========================================================================

    def _select_temporal_sensor_columns(
        self,
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Select numeric sensor/process variables suitable for temporal
        features.

        Target and failure-mode columns are always excluded.
        """

        excluded = set(
            EXCLUDED_FROM_FEATURE_GENERATION
        )

        excluded.update(
            {
                "UID",
                "Product ID",
                "Type",
                self.timestamp_column,
                self.machine_id_column,
            }
        )

        # Prefer known physical sensor columns.
        candidates = [
            column
            for column in NUMERIC_SENSOR_COLUMNS
            if column in df.columns
        ]

        # Extend with additional raw numeric sensor columns from real datasets.
        #
        # IMPORTANT:
        # This method is called repeatedly while feature generation is in
        # progress.  Therefore previously engineered numeric columns must NOT
        # become new temporal source columns.  Otherwise the pipeline can start
        # creating rolling/EWM/ROC features from lag and rolling features, which
        # causes a combinatorial feature explosion and severe DataFrame
        # fragmentation.
        engineered_prefixes = (
            f"{FEATURE_GROUP_TIME}__",
            f"{FEATURE_GROUP_LAG}__",
            f"{FEATURE_GROUP_ROLLING}__",
            f"{FEATURE_GROUP_EWM}__",
            f"{FEATURE_GROUP_ROC}__",
            f"{FEATURE_GROUP_ANOMALY}__",
            f"{FEATURE_GROUP_MISSINGNESS}__",
            f"{FEATURE_GROUP_DIFFERENCE}__",
            f"{FEATURE_GROUP_RATIO}__",
            f"{FEATURE_GROUP_PHYSICAL}__",
            f"{FEATURE_GROUP_TOOL_WEAR}__",
            f"{FEATURE_GROUP_INTERACTION}__",
            f"{FEATURE_GROUP_OPERATING_CONDITION}__",
        )

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        for column in numeric_columns:

            if column in excluded:
                continue

            if str(column).startswith(engineered_prefixes):
                continue

            if column not in candidates:
                candidates.append(
                    column
                )

        return candidates

    def _get_numeric(
        self,
        df: pd.DataFrame,
        column: str,
    ) -> Optional[pd.Series]:
        """
        Get a numeric source column or None.
        """

        if column not in df.columns:
            return None

        return _ensure_numeric(
            df[column]
        )

    # ========================================================================
    # IMPUTATION
    # ========================================================================

    def _apply_learned_numeric_imputation(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply numeric imputation statistics learned from training data.

        This method never learns statistics from validation/test data.
        """

        result = df.copy()

        for column, median in (
            self.state.median_values.items()
        ):

            if column not in result.columns:
                continue

            if pd.isna(median):
                continue

            # Do not overwrite timestamp or target fields.
            if column in {
                TARGET_COLUMN,
                self.timestamp_column,
            }:
                continue

            if pd.api.types.is_numeric_dtype(
                result[column]
            ):

                result[column] = (
                    result[column]
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .fillna(
                        median
                    )
                )

        return self._replace_invalid_values(
            result
        )

    def _replace_invalid_values(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Replace numerical infinities with NaN.
        """

        result = df.copy()

        numeric_columns = (
            result.select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        if numeric_columns:

            result[
                numeric_columns
            ] = result[
                numeric_columns
            ].replace(
                [
                    np.inf,
                    -np.inf,
                ],
                np.nan,
            )

        return result

    # ========================================================================
    # QUALITY FILTERING
    # ========================================================================

    @staticmethod
    def _find_constant_columns(
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Identify constant columns.
        """

        constants = []

        for column in df.columns:

            if column in {
                TARGET_COLUMN,
            }:
                continue

            if df[column].nunique(
                dropna=False
            ) <= 1:

                constants.append(
                    column
                )

        return constants

    @staticmethod
    def _find_duplicate_columns(
        df: pd.DataFrame,
    ) -> List[str]:
        """
        Identify exact duplicate columns.

        The first occurrence is retained.
        """

        duplicate_columns = []

        columns = df.columns.tolist()

        seen: Dict[int, List[str]] = {}

        # Hash-based grouping reduces unnecessary pairwise comparisons.
        for column in columns:

            series = df[column]

            try:
                hash_value = hash(
                    pd.util.hash_pandas_object(
                        series,
                        index=False,
                    ).values.tobytes()
                )
            except Exception:
                hash_value = hash(
                    str(
                        series.tolist()
                    )
                )

            seen.setdefault(
                hash_value,
                [],
            ).append(
                column
            )

        for candidates in seen.values():

            if len(candidates) <= 1:
                continue

            first = candidates[0]

            for candidate in candidates[1:]:

                try:
                    identical = (
                        df[first].equals(
                            df[candidate]
                        )
                    )
                except Exception:
                    identical = False

                if identical:
                    duplicate_columns.append(
                        candidate
                    )

        return duplicate_columns

    # ========================================================================
    # METADATA
    # ========================================================================

    def _register_feature(
        self,
        feature_name: str,
        feature_group: str,
        source_columns: Sequence[str],
        operation: str,
        *,
        causal: bool,
        description: str,
    ) -> None:
        """
        Register feature metadata.

        Duplicate metadata registrations are avoided.
        """

        existing = {
            item.feature_name
            for item in self.metadata
        }

        if feature_name in existing:
            return

        self.metadata.append(
            FeatureMetadata(
                feature_name=feature_name,
                feature_group=feature_group,
                source_columns=list(
                    source_columns
                ),
                operation=operation,
                causal=causal,
                leakage_safe=True,
                description=description,
            )
        )

    def get_feature_metadata(
        self,
    ) -> pd.DataFrame:
        """
        Return feature metadata as a DataFrame.
        """

        if not self.metadata:

            return pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_group",
                    "source_columns",
                    "operation",
                    "causal",
                    "leakage_safe",
                    "description",
                ]
            )

        return pd.DataFrame(
            [
                item.to_dict()
                for item in self.metadata
            ]
        )

    def get_feature_groups(
        self,
    ) -> Dict[str, List[str]]:
        """
        Group engineered features by feature family.
        """

        groups: Dict[
            str,
            List[str],
        ] = {}

        for item in self.metadata:

            groups.setdefault(
                item.feature_group,
                [],
            ).append(
                item.feature_name
            )

        return groups

    # ========================================================================
    # LEAKAGE CHECK
    # ========================================================================

    def validate_no_target_leakage(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, object]:
        """
        Verify that target/failure-mode columns are not used as engineered
        predictors.

        The target may still be present as an output label.

        Returns
        -------
        dict
            Leakage validation report.
        """

        engineered_columns = [
            column
            for column in features.columns
            if column not in {
                TARGET_COLUMN,
                *ID_COLUMNS,
                self.timestamp_column,
                self.machine_id_column,
            }
        ]

        exact_leakage = [
            column
            for column in engineered_columns
            if column in EXCLUDED_FROM_FEATURE_GENERATION
        ]

        name_based_leakage = [
            column
            for column in engineered_columns
            if any(
                token in column.lower()
                for token in [
                    "machine_failure",
                    "twf",
                    "hdf",
                    "pwf",
                    "osf",
                    "rnf",
                ]
            )
        ]

        # Metadata source validation.
        metadata_leakage = []

        for item in self.metadata:

            if any(
                source in EXCLUDED_FROM_FEATURE_GENERATION
                for source in item.source_columns
            ):
                metadata_leakage.append(
                    item.feature_name
                )

        leakage_detected = bool(
            exact_leakage
            or name_based_leakage
            or metadata_leakage
        )

        return {
            "leakage_detected": leakage_detected,
            "exact_leakage_columns": exact_leakage,
            "name_based_leakage_columns": name_based_leakage,
            "metadata_leakage_features": metadata_leakage,
            "checked_feature_count": len(
                engineered_columns
            ),
        }

    # ========================================================================
    # FEATURE SUMMARY
    # ========================================================================

    def get_feature_summary(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce a feature-level statistical summary.
        """

        rows = []

        for column in features.columns:

            series = features[column]

            numeric = (
                pd.api.types.is_numeric_dtype(
                    series
                )
            )

            if numeric:

                rows.append(
                    {
                        "feature": column,
                        "dtype": str(
                            series.dtype
                        ),
                        "non_null": int(
                            series.notna().sum()
                        ),
                        "missing": int(
                            series.isna().sum()
                        ),
                        "missing_ratio": float(
                            series.isna().mean()
                        ),
                        "n_unique": int(
                            series.nunique(
                                dropna=True
                            )
                        ),
                        "mean": float(
                            series.mean()
                        )
                        if series.notna().any()
                        else np.nan,
                        "std": float(
                            series.std(
                                ddof=0
                            )
                        )
                        if series.notna().any()
                        else np.nan,
                        "min": float(
                            series.min()
                        )
                        if series.notna().any()
                        else np.nan,
                        "median": float(
                            series.median()
                        )
                        if series.notna().any()
                        else np.nan,
                        "max": float(
                            series.max()
                        )
                        if series.notna().any()
                        else np.nan,
                    }
                )

            else:

                rows.append(
                    {
                        "feature": column,
                        "dtype": str(
                            series.dtype
                        ),
                        "non_null": int(
                            series.notna().sum()
                        ),
                        "missing": int(
                            series.isna().sum()
                        ),
                        "missing_ratio": float(
                            series.isna().mean()
                        ),
                        "n_unique": int(
                            series.nunique(
                                dropna=True
                            )
                        ),
                        "mean": np.nan,
                        "std": np.nan,
                        "min": np.nan,
                        "median": np.nan,
                        "max": np.nan,
                    }
                )

        return pd.DataFrame(
            rows
        )

    # ========================================================================
    # ONLINE FEATURE SUPPORT
    # ========================================================================

    def generate_online_features(
        self,
        observation: Dict[str, object],
        history: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate features for a single incoming observation.

        This helper is intended for real-time inference.

        Parameters
        ----------
        observation:
            Current sensor observation.

        history:
            Previously observed rows for the same machine.

        Returns
        -------
        pd.DataFrame
            One-row engineered feature set.

        Notes
        -----
        The current observation is included only where the feature logically
        requires current state, such as physical interactions or current-vs-
        historical rate of change.

        Historical rolling/EWM features are calculated from history only.
        """

        current = pd.DataFrame(
            [observation]
        )

        if history is not None and not history.empty:

            combined = pd.concat(
                [
                    history.copy(),
                    current,
                ],
                ignore_index=True,
            )

        else:

            combined = current.copy()

        transformed = self.transform(
            combined
        )

        return transformed.tail(
            1
        ).reset_index(
            drop=True
        )

    # ========================================================================
    # STATE SUMMARY
    # ========================================================================

    def get_state_summary(
        self,
    ) -> Dict[str, object]:
        """
        Return the fitted pipeline state.
        """

        return {
            "fitted": self.state.fitted,
            "input_column_count": len(
                self.state.input_columns
            ),
            "numeric_column_count": len(
                self.state.numeric_columns
            ),
            "categorical_column_count": len(
                self.state.categorical_columns
            ),
            "retained_feature_count": len(
                self.state.feature_columns
            ),
            "constant_feature_count": len(
                self.state.constant_columns
            ),
            "duplicate_feature_count": len(
                self.state.duplicate_columns
            ),
            "metadata_feature_count": len(
                self.metadata
            ),
            "lag_periods": list(
                self.lags
            ),
            "rolling_windows": list(
                self.rolling_windows
            ),
            "rolling_statistics": list(
                self.rolling_statistics
            ),
            "ewm_spans": list(
                self.ewm_spans
            ),
            "roc_periods": list(
                self.roc_periods
            ),
            "z_score_windows": list(
                self.z_score_windows
            ),
            "anomaly_threshold": (
                self.anomaly_threshold
            ),
        }


# ============================================================================
# FUNCTIONAL API
# ============================================================================

def engineer_features(
    train_df: pd.DataFrame,
    validation_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    **kwargs,
) -> Dict[str, object]:
    """
    Convenience function for leakage-safe train/validation/test engineering.

    Parameters
    ----------
    train_df:
        Training data.

    validation_df:
        Validation data.

    test_df:
        Test data.

    **kwargs:
        Parameters passed to PredictiveMaintenanceFeatureEngineer.

    Returns
    -------
    dict
        Contains:
            engineer
            train
            validation
            test
            metadata
            leakage_report
    """

    engineer = (
        PredictiveMaintenanceFeatureEngineer(
            **kwargs
        )
    )

    train_features = (
        engineer.fit_transform(
            train_df
        )
    )

    validation_features = None

    if validation_df is not None:

        validation_features = (
            engineer.transform(
                validation_df
            )
        )

    test_features = None

    if test_df is not None:

        test_features = (
            engineer.transform(
                test_df
            )
        )

    leakage_report = (
        engineer.validate_no_target_leakage(
            train_features
        )
    )

    if leakage_report[
        "leakage_detected"
    ]:

        raise RuntimeError(
            "Feature leakage detected after engineering: "
            f"{leakage_report}"
        )

    return {
        "engineer": engineer,
        "train": train_features,
        "validation": validation_features,
        "test": test_features,
        "metadata": (
            engineer.get_feature_metadata()
        ),
        "leakage_report": leakage_report,
    }


# ============================================================================
# SIMPLE FEATURE COUNTER
# ============================================================================

def count_features_by_group(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Count engineered features by feature group.
    """

    if metadata.empty:

        return pd.DataFrame(
            columns=[
                "feature_group",
                "feature_count",
            ]
        )

    return (
        metadata.groupby(
            "feature_group"
        )
        .size()
        .reset_index(
            name="feature_count"
        )
        .sort_values(
            "feature_count",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================================
# STANDALONE TEST
# ============================================================================

def _create_demo_ai4i_data(
    n_rows: int = 100,
) -> pd.DataFrame:
    """
    Create a small synthetic AI4I-shaped dataset for module testing only.

    This is NOT the project dataset and is never used by the actual pipeline.
    """

    rng = np.random.default_rng(
        42
    )

    return pd.DataFrame(
        {
            "UID": np.arange(
                1,
                n_rows + 1,
            ),
            "Product ID": [
                f"L{index:05d}"
                for index in range(
                    1,
                    n_rows + 1,
                )
            ],
            "Type": rng.choice(
                [
                    "L",
                    "M",
                    "H",
                ],
                size=n_rows,
            ),
            "Air temperature [K]": (
                298.0
                + rng.normal(
                    0,
                    1.0,
                    n_rows,
                )
            ),
            "Process temperature [K]": (
                308.0
                + rng.normal(
                    0,
                    1.2,
                    n_rows,
                )
            ),
            "Rotational speed [rpm]": (
                1500.0
                + rng.normal(
                    0,
                    100.0,
                    n_rows,
                )
            ),
            "Torque [Nm]": (
                40.0
                + rng.normal(
                    0,
                    5.0,
                    n_rows,
                )
            ),
            "Tool wear [min]": np.clip(
                np.arange(
                    n_rows
                ) * 2.0
                + rng.normal(
                    0,
                    2.0,
                    n_rows,
                ),
                0,
                None,
            ),
            "Machine failure": rng.binomial(
                1,
                0.03,
                n_rows,
            ),
            "TWF": 0,
            "HDF": 0,
            "PWF": 0,
            "OSF": 0,
            "RNF": 0,
        }
    )


def main() -> None:
    """
    Standalone module test.

    This verifies that the feature-engineering implementation can:
        - fit
        - transform
        - generate physical features
        - generate time-series features
        - produce metadata
        - perform leakage validation
    """

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PREDICTIVE MAINTENANCE FEATURE ENGINEERING TEST"
    )

    print(
        "=" * 80
    )

    # ---------------------------------------------------------------------
    # AI4I-style test
    # ---------------------------------------------------------------------

    df = _create_demo_ai4i_data(
        100
    )

    engineer = (
        PredictiveMaintenanceFeatureEngineer()
    )

    engineered = (
        engineer.fit_transform(
            df
        )
    )

    print(
        "\nAI4I-style feature engineering:"
    )

    print(
        f"Input shape       : {df.shape}"
    )

    print(
        f"Output shape      : {engineered.shape}"
    )

    print(
        f"Features retained : "
        f"{len(engineer.state.feature_columns)}"
    )

    print(
        "\nFeature groups:"
    )

    for group, columns in (
        engineer.get_feature_groups().items()
    ):

        print(
            f"  {group:<25} {len(columns):>5}"
        )

    leakage_report = (
        engineer.validate_no_target_leakage(
            engineered
        )
    )

    print(
        "\nLeakage validation:"
    )

    print(
        leakage_report
    )

    # ---------------------------------------------------------------------
    # Genuine time-series test
    # ---------------------------------------------------------------------

    timestamps = pd.date_range(
        "2026-01-01",
        periods=100,
        freq="min",
    )

    ts_df = df.copy()

    ts_df[
        "timestamp"
    ] = timestamps

    ts_df[
        "machine_id"
    ] = np.where(
        np.arange(100) < 50,
        "M001",
        "M002",
    )

    ts_engineer = (
        PredictiveMaintenanceFeatureEngineer()
    )

    ts_features = (
        ts_engineer.fit_transform(
            ts_df
        )
    )

    print(
        "\nTrue time-series feature engineering:"
    )

    print(
        f"Input shape       : {ts_df.shape}"
    )

    print(
        f"Output shape      : {ts_features.shape}"
    )

    temporal_feature_count = sum(
        1
        for item in ts_engineer.metadata
        if item.feature_group
        in {
            FEATURE_GROUP_TIME,
            FEATURE_GROUP_LAG,
            FEATURE_GROUP_ROLLING,
            FEATURE_GROUP_EWM,
            FEATURE_GROUP_ROC,
            FEATURE_GROUP_ANOMALY,
        }
    )

    print(
        f"Temporal features : "
        f"{temporal_feature_count}"
    )

    print(
        "\nFeature engineering module test completed successfully."
    )


if __name__ == "__main__":
    main()