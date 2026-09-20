"""
src/realtime_features.py

Real-time / online feature generation for the Predictive Maintenance
Feature Engineering project.

Purpose
-------
This module provides production-oriented, stateful feature generation for
machine observations arriving one at a time.

It is designed to complement:

    src/feature_engineering.py

The batch feature-engineering module is responsible for generating features
from historical DataFrames.

This module is responsible for the online inference scenario:

    raw observation
            |
            v
    RealTimeFeatureEngineer
            |
            v
    leakage-safe engineered features
            |
            v
       ML prediction

Key design principles
---------------------
1. No future data is used.
2. Historical state is maintained independently for every machine.
3. Lag features use only observations already received.
4. Rolling statistics use only previous observations.
5. EWMA features are updated sequentially.
6. Rate-of-change uses elapsed time when a real timestamp exists.
7. Missing values are handled deterministically.
8. AI4I UID is NEVER treated as a timestamp.
9. Failure-mode labels are never used as predictive features.
10. The class can be persisted/restored for production deployments.
11. The implementation is deterministic and suitable for unit testing.
12. The feature names are stable and suitable for model-serving pipelines.

Important AI4I note
-------------------
The UCI AI4I 2020 dataset contains UID, Product ID, Type, sensor values,
Tool wear, Machine failure, and failure-mode columns.

UID is an identifier, not a genuine event timestamp.

Therefore:

    UID -> identifier only

and NOT:

    UID -> timestamp
    UID -> elapsed time
    UID -> chronological lag

For a real production deployment, the incoming observation should ideally
contain:

    machine_id
    timestamp
    sensor measurements

Example
-------
    engineer = RealTimeFeatureEngineer(
        machine_id_column="machine_id",
        timestamp_column="timestamp",
    )

    engineer.start_machine("MACHINE_001")

    features = engineer.update(
        {
            "machine_id": "MACHINE_001",
            "timestamp": "2026-01-01T10:00:00Z",
            "air_temperature_k": 298.5,
            "process_temperature_k": 308.7,
            "rotational_speed_rpm": 1500,
            "torque_nm": 42.0,
            "tool_wear_min": 120,
        }
    )

The returned dictionary contains the current observation's safe
real-time features.

The module also supports:

    - batch-like sequential processing
    - machine state reset
    - complete state reset
    - state export
    - state import
    - feature-name inspection
    - health/state inspection
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence
import copy
import json
import logging
import math

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


# ============================================================================
# Canonical AI4I columns
# ============================================================================

AI4I_TARGET_COLUMNS: tuple[str, ...] = (
    "machine_failure",
)

AI4I_FAILURE_MODE_COLUMNS: tuple[str, ...] = (
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
    "failure_type",
)

AI4I_IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "uid",
    "product_id",
)

AI4I_SENSOR_COLUMNS: tuple[str, ...] = (
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True)
class RealTimeFeatureConfig:
    """
    Configuration for online feature generation.

    All stateful temporal features are computed from observations that have
    already arrived.

    Parameters
    ----------
    lag_periods:
        Number of previous observations used for lag features.

    rolling_windows:
        Number of historical observations used for rolling statistics.

    ewm_spans:
        EWMA spans.

    rolling_min_periods:
        Minimum number of historical observations required for rolling
        statistics.

    zscore_min_history:
        Minimum historical observations required for a rolling z-score.

    epsilon:
        Numerical stability constant for ratios/divisions.

    max_history_per_machine:
        Maximum retained observations per machine.

        This is primarily a memory/safety control. It should be large enough
        to cover the largest configured rolling window and lag.

    add_time_features:
        Add calendar/time-derived features when a real timestamp exists.

    add_lag_features:
        Add lag features.

    add_rolling_features:
        Add rolling mean/std/min/max/median features.

    add_ewm_features:
        Add exponentially weighted moving averages.

    add_rate_of_change:
        Add rate-of-change features.

    add_anomaly_features:
        Add rolling z-score features.

    add_physical_features:
        Add physically meaningful AI4I features.

    add_interaction_features:
        Add sensor interaction features.

    allow_non_temporal_deltas:
        If True, rate-of-change features may fall back to differences per
        observation when no real timestamp is available.

        Default is False because a row difference is not necessarily a
        physical rate.

    timestamp_column:
        Name of the incoming timestamp field.

    machine_id_column:
        Name of the machine/asset identifier.

    sensor_columns:
        Sensors to track historically.

        If empty, the class automatically uses the standard AI4I sensors
        that are present in incoming observations.
    """

    lag_periods: tuple[int, ...] = (1, 3, 6)

    rolling_windows: tuple[int, ...] = (3, 6, 12)

    ewm_spans: tuple[int, ...] = (3, 6, 12)

    rolling_min_periods: int = 2

    zscore_min_history: int = 3

    epsilon: float = 1e-8

    max_history_per_machine: int = 128

    add_time_features: bool = True

    add_lag_features: bool = True

    add_rolling_features: bool = True

    add_ewm_features: bool = True

    add_rate_of_change: bool = True

    add_anomaly_features: bool = True

    add_physical_features: bool = True

    add_interaction_features: bool = True

    allow_non_temporal_deltas: bool = False

    timestamp_column: str = "timestamp"

    machine_id_column: str = "machine_id"

    sensor_columns: tuple[str, ...] = AI4I_SENSOR_COLUMNS

    def __post_init__(self) -> None:
        """
        Validate configuration.
        """
        if not self.lag_periods:
            raise ValueError(
                "lag_periods must contain at least one positive integer."
            )

        if not self.rolling_windows:
            raise ValueError(
                "rolling_windows must contain at least one positive integer."
            )

        if not self.ewm_spans:
            raise ValueError(
                "ewm_spans must contain at least one positive integer."
            )

        if any(
            not isinstance(value, int) or value <= 0
            for value in self.lag_periods
        ):
            raise ValueError(
                "All lag_periods must be positive integers."
            )

        if any(
            not isinstance(value, int) or value <= 0
            for value in self.rolling_windows
        ):
            raise ValueError(
                "All rolling_windows must be positive integers."
            )

        if any(
            not isinstance(value, int) or value <= 0
            for value in self.ewm_spans
        ):
            raise ValueError(
                "All ewm_spans must be positive integers."
            )

        if self.rolling_min_periods <= 0:
            raise ValueError(
                "rolling_min_periods must be positive."
            )

        if self.zscore_min_history <= 0:
            raise ValueError(
                "zscore_min_history must be positive."
            )

        if self.epsilon <= 0:
            raise ValueError(
                "epsilon must be greater than zero."
            )

        if self.max_history_per_machine <= 0:
            raise ValueError(
                "max_history_per_machine must be positive."
            )


# ============================================================================
# Internal state objects
# ============================================================================

@dataclass
class MachineHistory:
    """
    Stateful historical information for one machine.

    The deques retain only the amount of history required by the configured
    temporal features.

    All values represent observations that have already arrived.

    The current observation is NOT inserted into this history until all
    historical features for the current observation have been calculated.
    This ordering is critical for leakage prevention.
    """

    timestamps: deque = field(
        default_factory=lambda: deque(maxlen=128)
    )

    sensor_history: dict[str, deque] = field(
        default_factory=dict
    )

    ewm_values: dict[str, dict[int, float]] = field(
        default_factory=dict
    )

    observation_count: int = 0

    last_timestamp: Optional[pd.Timestamp] = None


# ============================================================================
# Main real-time feature engineer
# ============================================================================

class RealTimeFeatureEngineer:
    """
    Stateful, leakage-safe online feature generator.

    The class maintains independent state for each machine.

    Example
    -------
    >>> engineer = RealTimeFeatureEngineer()
    >>> features = engineer.update(
    ...     {
    ...         "machine_id": "M001",
    ...         "timestamp": "2026-01-01T10:00:00Z",
    ...         "air_temperature_k": 298.1,
    ...         "process_temperature_k": 308.5,
    ...         "rotational_speed_rpm": 1500,
    ...         "torque_nm": 40.0,
    ...         "tool_wear_min": 100.0,
    ...     }
    ... )
    """

    def __init__(
        self,
        config: Optional[RealTimeFeatureConfig] = None,
        *,
        machine_id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        sensor_columns: Optional[Sequence[str]] = None,
        feature_medians: Optional[Mapping[str, float]] = None,
    ) -> None:
        """
        Initialize the real-time feature generator.

        Parameters
        ----------
        config:
            RealTimeFeatureConfig instance.

        machine_id_column:
            Optional override for config.machine_id_column.

        timestamp_column:
            Optional override for config.timestamp_column.

        sensor_columns:
            Optional override for configured sensor columns.

        feature_medians:
            Optional training-set medians used for deterministic
            missing-value fallback.
        """
        if config is None:
            config = RealTimeFeatureConfig()

        if machine_id_column is not None:
            config = self._replace_config(
                config,
                machine_id_column=machine_id_column,
            )

        if timestamp_column is not None:
            config = self._replace_config(
                config,
                timestamp_column=timestamp_column,
            )

        if sensor_columns is not None:
            config = self._replace_config(
                config,
                sensor_columns=tuple(sensor_columns),
            )

        self.config = config

        self._states: dict[str, MachineHistory] = {}

        self._feature_medians: dict[str, float] = {
            str(key): float(value)
            for key, value in (
                feature_medians or {}
            ).items()
            if value is not None
            and np.isfinite(float(value))
        }

        self._feature_names: set[str] = set()

        self._total_observations: int = 0

        self._last_update_time: Optional[pd.Timestamp] = None

    # ------------------------------------------------------------------------
    # Configuration helper
    # ------------------------------------------------------------------------

    @staticmethod
    def _replace_config(
        config: RealTimeFeatureConfig,
        **updates: Any,
    ) -> RealTimeFeatureConfig:
        """
        Create a modified immutable configuration.
        """
        values = {
            field_name: getattr(
                config,
                field_name,
            )
            for field_name in config.__dataclass_fields__
        }

        values.update(updates)

        return RealTimeFeatureConfig(
            **values
        )

    # ------------------------------------------------------------------------
    # Machine state management
    # ------------------------------------------------------------------------

    def _create_machine_state(self) -> MachineHistory:
        """
        Create an empty state object with correctly sized deques.
        """
        state = MachineHistory(
            timestamps=deque(
                maxlen=self.config.max_history_per_machine
            ),
            sensor_history={},
            ewm_values={},
            observation_count=0,
            last_timestamp=None,
        )

        for sensor in self.config.sensor_columns:
            state.sensor_history[sensor] = deque(
                maxlen=self.config.max_history_per_machine
            )

            state.ewm_values[sensor] = {}

        return state

    def start_machine(
        self,
        machine_id: str,
    ) -> None:
        """
        Explicitly initialize state for a machine.

        Existing state is preserved.
        """
        machine_key = self._normalize_machine_id(
            machine_id
        )

        if machine_key not in self._states:
            self._states[machine_key] = (
                self._create_machine_state()
            )

    def reset_machine(
        self,
        machine_id: str,
    ) -> bool:
        """
        Delete the state of one machine.

        Returns
        -------
        bool
            True if a state existed and was removed.
        """
        machine_key = self._normalize_machine_id(
            machine_id
        )

        if machine_key in self._states:
            del self._states[machine_key]
            return True

        return False

    def reset_all(
        self,
    ) -> None:
        """
        Delete all online machine state.
        """
        self._states.clear()

        self._total_observations = 0

        self._last_update_time = None

    def machine_ids(
        self,
    ) -> list[str]:
        """
        Return currently tracked machine identifiers.
        """
        return sorted(
            self._states.keys()
        )

    def has_machine(
        self,
        machine_id: str,
    ) -> bool:
        """
        Check whether machine state exists.
        """
        machine_key = self._normalize_machine_id(
            machine_id
        )

        return machine_key in self._states

    # ------------------------------------------------------------------------
    # Input normalization
    # ------------------------------------------------------------------------

    @staticmethod
    def _normalize_machine_id(
        machine_id: Any,
    ) -> str:
        """
        Convert a machine identifier into a stable string key.
        """
        if machine_id is None:
            return "__DEFAULT_MACHINE__"

        if pd.isna(machine_id):
            return "__DEFAULT_MACHINE__"

        text = str(machine_id).strip()

        if not text:
            return "__DEFAULT_MACHINE__"

        return text

    def _parse_timestamp(
        self,
        value: Any,
    ) -> Optional[pd.Timestamp]:
        """
        Parse an incoming timestamp.

        Returns None for missing/invalid timestamps.

        No numeric UID is interpreted as a timestamp.
        """
        if value is None:
            return None

        if isinstance(
            value,
            pd.Timestamp,
        ):
            if value.tzinfo is None:
                return value.tz_localize("UTC")

            return value.tz_convert("UTC")

        if isinstance(
            value,
            datetime,
        ):
            timestamp = pd.Timestamp(value)

            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")

            return timestamp

        # Numeric values are deliberately rejected here.
        #
        # This prevents AI4I UID values from accidentally being interpreted
        # as UNIX timestamps.
        if isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            return None

        timestamp = pd.to_datetime(
            value,
            errors="coerce",
            utc=True,
        )

        if pd.isna(timestamp):
            return None

        return pd.Timestamp(timestamp)

    @staticmethod
    def _safe_float(
        value: Any,
    ) -> float:
        """
        Convert a sensor value to float.

        Invalid/missing values become NaN.
        """
        if value is None:
            return float("nan")

        try:
            numeric = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return float("nan")

        if not np.isfinite(numeric):
            return float("nan")

        return numeric

    # ------------------------------------------------------------------------
    # Safe mathematical operations
    # ------------------------------------------------------------------------

    def _safe_ratio(
        self,
        numerator: float,
        denominator: float,
    ) -> float:
        """
        Calculate a numerically safe ratio.
        """
        if not np.isfinite(numerator):
            return float("nan")

        if not np.isfinite(denominator):
            return float("nan")

        if abs(denominator) <= self.config.epsilon:
            return float("nan")

        return numerator / denominator

    @staticmethod
    def _safe_sqrt(
        value: float,
    ) -> float:
        """
        Safe square-root transformation.
        """
        if not np.isfinite(value):
            return float("nan")

        return math.sqrt(
            max(value, 0.0)
        )

    @staticmethod
    def _safe_log1p(
        value: float,
    ) -> float:
        """
        Safe log1p transformation for non-negative quantities.
        """
        if not np.isfinite(value):
            return float("nan")

        return math.log1p(
            max(value, 0.0)
        )

    # ------------------------------------------------------------------------
    # Sensor extraction
    # ------------------------------------------------------------------------

    def _extract_sensor_values(
        self,
        observation: Mapping[str, Any],
    ) -> dict[str, float]:
        """
        Extract configured numeric sensor values.
        """
        values: dict[str, float] = {}

        for sensor in self.config.sensor_columns:
            values[sensor] = self._safe_float(
                observation.get(sensor)
            )

        return values

    # ------------------------------------------------------------------------
    # Historical helper methods
    # ------------------------------------------------------------------------

    @staticmethod
    def _finite_history(
        history: deque,
    ) -> list[float]:
        """
        Return finite historical values only.
        """
        return [
            float(value)
            for value in history
            if value is not None
            and np.isfinite(value)
        ]

    def _lag_value(
        self,
        history: deque,
        lag: int,
    ) -> float:
        """
        Return a historical lag.

        history[-1] is the immediately previous observation.

        Current observation is NOT in history at this point.
        """
        if lag <= 0:
            raise ValueError(
                "lag must be positive."
            )

        if len(history) < lag:
            return float("nan")

        value = history[-lag]

        if value is None:
            return float("nan")

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return float("nan")

        return value if np.isfinite(value) else float("nan")

    def _rolling_values(
        self,
        history: deque,
        window: int,
    ) -> list[float]:
        """
        Return the latest historical window.
        """
        if window <= 0:
            raise ValueError(
                "window must be positive."
            )

        values = self._finite_history(
            history
        )

        if not values:
            return []

        return values[
            -window:
        ]

    def _rolling_mean(
        self,
        history: deque,
        window: int,
    ) -> float:
        """
        Historical rolling mean.
        """
        values = self._rolling_values(
            history,
            window,
        )

        if len(values) < self.config.rolling_min_periods:
            return float("nan")

        return float(
            np.mean(values)
        )

    def _rolling_std(
        self,
        history: deque,
        window: int,
    ) -> float:
        """
        Historical rolling standard deviation.

        ddof=1 is used to match pandas rolling.std() behavior.
        """
        values = self._rolling_values(
            history,
            window,
        )

        if len(values) < self.config.rolling_min_periods:
            return float("nan")

        if len(values) < 2:
            return float("nan")

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        return std

    def _rolling_min(
        self,
        history: deque,
        window: int,
    ) -> float:
        """
        Historical rolling minimum.
        """
        values = self._rolling_values(
            history,
            window,
        )

        if len(values) < self.config.rolling_min_periods:
            return float("nan")

        return float(
            np.min(values)
        )

    def _rolling_max(
        self,
        history: deque,
        window: int,
    ) -> float:
        """
        Historical rolling maximum.
        """
        values = self._rolling_values(
            history,
            window,
        )

        if len(values) < self.config.rolling_min_periods:
            return float("nan")

        return float(
            np.max(values)
        )

    def _rolling_median(
        self,
        history: deque,
        window: int,
    ) -> float:
        """
        Historical rolling median.
        """
        values = self._rolling_values(
            history,
            window,
        )

        if len(values) < self.config.rolling_min_periods:
            return float("nan")

        return float(
            np.median(values)
        )

    def _historical_zscore(
        self,
        current_value: float,
        history: deque,
        window: int,
    ) -> float:
        """
        Calculate the current observation's z-score against historical data.

        IMPORTANT:
            The current value is NOT used to calculate the historical mean
            or standard deviation.

        This makes the feature suitable for online prediction.
        """
        if not np.isfinite(current_value):
            return float("nan")

        values = self._rolling_values(
            history,
            window,
        )

        if len(values) < self.config.zscore_min_history:
            return float("nan")

        mean = float(
            np.mean(values)
        )

        std = float(
            np.std(
                values,
                ddof=1,
            )
        )

        if not np.isfinite(std):
            return float("nan")

        if std <= self.config.epsilon:
            return float("nan")

        return float(
            (current_value - mean)
            / std
        )

    # ------------------------------------------------------------------------
    # EWMA
    # ------------------------------------------------------------------------

    def _ewma_alpha(
        self,
        span: int,
    ) -> float:
        """
        Convert pandas-style EWMA span to alpha.
        """
        return 2.0 / (
            float(span) + 1.0
        )

    def _previous_ewma(
        self,
        state: MachineHistory,
        sensor: str,
        span: int,
    ) -> float:
        """
        Return the previously stored EWMA.
        """
        return state.ewm_values.get(
            sensor,
            {},
        ).get(
            span,
            float("nan"),
        )

    def _calculate_current_ewma(
        self,
        current_value: float,
        previous_ewma: float,
        span: int,
    ) -> float:
        """
        Calculate updated EWMA.

        If there is no previous EWMA, the first valid observation becomes
        the initial EWMA.
        """
        if not np.isfinite(current_value):
            return float("nan")

        if not np.isfinite(previous_ewma):
            return current_value

        alpha = self._ewma_alpha(
            span
        )

        return float(
            alpha * current_value
            + (1.0 - alpha) * previous_ewma
        )

    # ------------------------------------------------------------------------
    # Time features
    # ------------------------------------------------------------------------

    def _add_time_features(
        self,
        features: dict[str, Any],
        timestamp: Optional[pd.Timestamp],
        state: MachineHistory,
    ) -> None:
        """
        Add timestamp-derived features.

        These features are generated only when a genuine timestamp is
        supplied.
        """
        if not self.config.add_time_features:
            return

        if timestamp is None:
            return

        features["time_hour"] = float(
            timestamp.hour
        )

        features["time_minute"] = float(
            timestamp.minute
        )

        features["time_day_of_week"] = float(
            timestamp.dayofweek
        )

        features["time_day_of_month"] = float(
            timestamp.day
        )

        features["time_day_of_year"] = float(
            timestamp.dayofyear
        )

        features["time_month"] = float(
            timestamp.month
        )

        features["time_quarter"] = float(
            timestamp.quarter
        )

        features["time_week_of_year"] = float(
            timestamp.isocalendar().week
        )

        features["time_is_weekend"] = float(
            timestamp.dayofweek >= 5
        )

        if state.last_timestamp is not None:
            elapsed = (
                timestamp
                - state.last_timestamp
            ).total_seconds()

            if np.isfinite(elapsed):
                features[
                    "time_since_previous_observation_seconds"
                ] = float(
                    max(elapsed, 0.0)
                )

        if state.timestamps:
            first_timestamp = state.timestamps[0]

            if first_timestamp is not None:
                elapsed_from_start = (
                    timestamp
                    - first_timestamp
                ).total_seconds()

                if np.isfinite(
                    elapsed_from_start
                ):
                    features[
                        "elapsed_time_since_machine_start_seconds"
                    ] = float(
                        max(
                            elapsed_from_start,
                            0.0,
                        )
                    )

    # ------------------------------------------------------------------------
    # Physical feature engineering
    # ------------------------------------------------------------------------

    def _add_physical_features(
        self,
        features: dict[str, Any],
        sensors: Mapping[str, float],
    ) -> None:
        """
        Add physically meaningful predictive-maintenance features.

        AI4I-specific physical relationships:

            mechanical_power_w =
                torque_nm * rotational_speed_rpm * 2*pi / 60

            torque_speed_interaction =
                torque_nm * rotational_speed_rpm

            temperature_gap_k =
                process_temperature_k - air_temperature_k
        """
        if not self.config.add_physical_features:
            return

        air_temperature = sensors.get(
            "air_temperature_k",
            float("nan"),
        )

        process_temperature = sensors.get(
            "process_temperature_k",
            float("nan"),
        )

        rotational_speed = sensors.get(
            "rotational_speed_rpm",
            float("nan"),
        )

        torque = sensors.get(
            "torque_nm",
            float("nan"),
        )

        tool_wear = sensors.get(
            "tool_wear_min",
            float("nan"),
        )

        # ------------------------------------------------------------
        # Temperature relationship
        # ------------------------------------------------------------

        if (
            np.isfinite(process_temperature)
            and np.isfinite(air_temperature)
        ):
            temperature_gap = (
                process_temperature
                - air_temperature
            )

            features[
                "temperature_gap_k"
            ] = float(
                temperature_gap
            )

            features[
                "temperature_ratio"
            ] = self._safe_ratio(
                process_temperature,
                air_temperature,
            )

        # ------------------------------------------------------------
        # Mechanical power
        # ------------------------------------------------------------

        if (
            np.isfinite(rotational_speed)
            and np.isfinite(torque)
        ):
            torque_speed = (
                torque
                * rotational_speed
            )

            features[
                "torque_speed_interaction"
            ] = float(
                torque_speed
            )

            features[
                "rotational_speed_rad_per_s"
            ] = float(
                rotational_speed
                * 2.0
                * math.pi
                / 60.0
            )

            mechanical_power = (
                torque
                * rotational_speed
                * 2.0
                * math.pi
                / 60.0
            )

            features[
                "mechanical_power_w"
            ] = float(
                mechanical_power
            )

            features[
                "mechanical_power_kw"
            ] = float(
                mechanical_power
                / 1000.0
            )

            # 2860 W is the approximate nominal mechanical power around
            # which AI4I rotational speed is generated.
            #
            # This is a normalization feature, not a failure label.
            features[
                "mechanical_power_to_nominal_ratio"
            ] = self._safe_ratio(
                mechanical_power,
                2860.0,
            )

        # ------------------------------------------------------------
        # Torque/speed ratios
        # ------------------------------------------------------------

        if (
            np.isfinite(torque)
            and np.isfinite(rotational_speed)
        ):
            features[
                "torque_per_rpm"
            ] = self._safe_ratio(
                torque,
                rotational_speed,
            )

            features[
                "rpm_per_torque"
            ] = self._safe_ratio(
                rotational_speed,
                torque,
            )

        # ------------------------------------------------------------
        # Tool-wear transformations
        # ------------------------------------------------------------

        if np.isfinite(tool_wear):
            features[
                "tool_wear_squared"
            ] = float(
                tool_wear ** 2
            )

            features[
                "tool_wear_sqrt"
            ] = self._safe_sqrt(
                tool_wear
            )

            features[
                "tool_wear_log1p"
            ] = self._safe_log1p(
                tool_wear
            )

            # AI4I tool wear is expressed in minutes. 240 minutes is used
            # only as a domain normalization reference.
            features[
                "tool_wear_fraction_of_240_min"
            ] = self._safe_ratio(
                tool_wear,
                240.0,
            )

        # ------------------------------------------------------------
        # Normalized sensor quantities
        # ------------------------------------------------------------

        if np.isfinite(rotational_speed):
            features[
                "rotational_speed_k_rpm"
            ] = self._safe_ratio(
                rotational_speed,
                1000.0,
            )

        if np.isfinite(torque):
            features[
                "torque_k_nm"
            ] = self._safe_ratio(
                torque,
                10.0,
            )

    # ------------------------------------------------------------------------
    # Interaction features
    # ------------------------------------------------------------------------

    def _add_interaction_features(
        self,
        features: dict[str, Any],
        sensors: Mapping[str, float],
    ) -> None:
        """
        Add physically motivated interaction features.
        """
        if not self.config.add_interaction_features:
            return

        air_temperature = sensors.get(
            "air_temperature_k",
            float("nan"),
        )

        process_temperature = sensors.get(
            "process_temperature_k",
            float("nan"),
        )

        rotational_speed = sensors.get(
            "rotational_speed_rpm",
            float("nan"),
        )

        torque = sensors.get(
            "torque_nm",
            float("nan"),
        )

        tool_wear = sensors.get(
            "tool_wear_min",
            float("nan"),
        )

        temperature_gap = float("nan")

        if (
            np.isfinite(process_temperature)
            and np.isfinite(air_temperature)
        ):
            temperature_gap = (
                process_temperature
                - air_temperature
            )

        mechanical_power = float("nan")

        if (
            np.isfinite(torque)
            and np.isfinite(rotational_speed)
        ):
            mechanical_power = (
                torque
                * rotational_speed
                * 2.0
                * math.pi
                / 60.0
            )

        # Temperature-mechanical interactions
        if (
            np.isfinite(temperature_gap)
            and np.isfinite(torque)
        ):
            features[
                "temperature_gap_x_torque"
            ] = float(
                temperature_gap
                * torque
            )

        if (
            np.isfinite(temperature_gap)
            and np.isfinite(rotational_speed)
        ):
            features[
                "temperature_gap_x_speed"
            ] = float(
                temperature_gap
                * rotational_speed
            )

        if (
            np.isfinite(temperature_gap)
            and np.isfinite(tool_wear)
        ):
            features[
                "temperature_gap_x_tool_wear"
            ] = float(
                temperature_gap
                * tool_wear
            )

        # Mechanical-load interactions
        if (
            np.isfinite(mechanical_power)
            and np.isfinite(tool_wear)
        ):
            features[
                "mechanical_power_x_tool_wear"
            ] = float(
                mechanical_power
                * tool_wear
            )

        if (
            np.isfinite(torque)
            and np.isfinite(tool_wear)
        ):
            features[
                "torque_x_tool_wear"
            ] = float(
                torque
                * tool_wear
            )

        if (
            np.isfinite(rotational_speed)
            and np.isfinite(tool_wear)
        ):
            features[
                "speed_x_tool_wear"
            ] = float(
                rotational_speed
                * tool_wear
            )

    # ------------------------------------------------------------------------
    # Lag features
    # ------------------------------------------------------------------------

    def _add_lag_features(
        self,
        features: dict[str, Any],
        state: MachineHistory,
    ) -> None:
        """
        Add leakage-safe lag features.

        Only previous observations are available in state at this point.
        """
        if not self.config.add_lag_features:
            return

        for sensor in self.config.sensor_columns:
            history = state.sensor_history.get(
                sensor
            )

            if history is None:
                continue

            for lag in self.config.lag_periods:
                features[
                    f"{sensor}_lag_{lag}"
                ] = self._lag_value(
                    history,
                    lag,
                )

    # ------------------------------------------------------------------------
    # Rolling features
    # ------------------------------------------------------------------------

    def _add_rolling_features(
        self,
        features: dict[str, Any],
        state: MachineHistory,
    ) -> None:
        """
        Add leakage-safe rolling statistics.

        The current observation has not yet been inserted into state.
        Therefore all values come from the historical window.
        """
        if not self.config.add_rolling_features:
            return

        for sensor in self.config.sensor_columns:
            history = state.sensor_history.get(
                sensor
            )

            if history is None:
                continue

            for window in self.config.rolling_windows:
                prefix = (
                    f"{sensor}_rolling_{window}"
                )

                features[
                    f"{prefix}_mean"
                ] = self._rolling_mean(
                    history,
                    window,
                )

                features[
                    f"{prefix}_std"
                ] = self._rolling_std(
                    history,
                    window,
                )

                features[
                    f"{prefix}_min"
                ] = self._rolling_min(
                    history,
                    window,
                )

                features[
                    f"{prefix}_max"
                ] = self._rolling_max(
                    history,
                    window,
                )

                features[
                    f"{prefix}_median"
                ] = self._rolling_median(
                    history,
                    window,
                )

    # ------------------------------------------------------------------------
    # EWMA features
    # ------------------------------------------------------------------------

    def _add_ewm_features(
        self,
        features: dict[str, Any],
        state: MachineHistory,
        sensors: Mapping[str, float],
    ) -> None:
        """
        Calculate current EWMA values.

        The returned EWMA includes the current observation, which is valid
        because the feature is generated at the same moment as the prediction.

        The previous EWMA is used as state, so no future observations are
        incorporated.
        """
        if not self.config.add_ewm_features:
            return

        for sensor in self.config.sensor_columns:
            current_value = sensors.get(
                sensor,
                float("nan"),
            )

            for span in self.config.ewm_spans:
                previous = self._previous_ewma(
                    state,
                    sensor,
                    span,
                )

                current_ewma = (
                    self._calculate_current_ewma(
                        current_value,
                        previous,
                        span,
                    )
                )

                features[
                    f"{sensor}_ewm_{span}"
                ] = current_ewma

    # ------------------------------------------------------------------------
    # Rate-of-change features
    # ------------------------------------------------------------------------

    def _add_rate_of_change_features(
        self,
        features: dict[str, Any],
        state: MachineHistory,
        sensors: Mapping[str, float],
        timestamp: Optional[pd.Timestamp],
    ) -> None:
        """
        Add leakage-safe rate-of-change features.

        When a real timestamp exists:

            rate = delta_sensor / elapsed_seconds

        Without a real timestamp, rates are omitted unless
        allow_non_temporal_deltas=True.
        """
        if not self.config.add_rate_of_change:
            return

        for sensor in self.config.sensor_columns:
            history = state.sensor_history.get(
                sensor
            )

            if history is None:
                continue

            current_value = sensors.get(
                sensor,
                float("nan"),
            )

            if not np.isfinite(current_value):
                continue

            for lag in self.config.lag_periods:
                previous_value = self._lag_value(
                    history,
                    lag,
                )

                if not np.isfinite(
                    previous_value
                ):
                    features[
                        f"{sensor}_roc_{lag}"
                    ] = float("nan")

                    continue

                delta_value = (
                    current_value
                    - previous_value
                )

                elapsed_seconds: Optional[float] = None

                if (
                    timestamp is not None
                    and len(state.timestamps) >= lag
                ):
                    previous_timestamp = (
                        list(
                            state.timestamps
                        )[-lag]
                    )

                    if previous_timestamp is not None:
                        elapsed_seconds = (
                            timestamp
                            - previous_timestamp
                        ).total_seconds()

                if (
                    elapsed_seconds is not None
                    and np.isfinite(
                        elapsed_seconds
                    )
                    and elapsed_seconds > 0
                ):
                    rate = (
                        delta_value
                        / elapsed_seconds
                    )

                elif (
                    self.config.allow_non_temporal_deltas
                ):
                    rate = delta_value

                else:
                    rate = float("nan")

                features[
                    f"{sensor}_roc_{lag}"
                ] = float(rate)

    # ------------------------------------------------------------------------
    # Anomaly features
    # ------------------------------------------------------------------------

    def _add_anomaly_features(
        self,
        features: dict[str, Any],
        state: MachineHistory,
        sensors: Mapping[str, float],
    ) -> None:
        """
        Add historical rolling z-score features.

        The current measurement is compared with previous measurements only.
        """
        if not self.config.add_anomaly_features:
            return

        for sensor in self.config.sensor_columns:
            history = state.sensor_history.get(
                sensor
            )

            if history is None:
                continue

            current_value = sensors.get(
                sensor,
                float("nan"),
            )

            for window in self.config.rolling_windows:
                zscore = self._historical_zscore(
                    current_value,
                    history,
                    window,
                )

                features[
                    f"{sensor}_rolling_{window}_zscore"
                ] = zscore

                if np.isfinite(zscore):
                    features[
                        f"{sensor}_rolling_{window}_anomaly_score"
                    ] = float(
                        abs(zscore)
                    )
                else:
                    features[
                        f"{sensor}_rolling_{window}_anomaly_score"
                    ] = float("nan")

    # ------------------------------------------------------------------------
    # Online feature generation
    # ------------------------------------------------------------------------

    def update(
        self,
        observation: Mapping[str, Any],
        *,
        include_raw_features: bool = True,
        impute_missing: bool = False,
    ) -> dict[str, Any]:
        """
        Process one incoming machine observation.

        Parameters
        ----------
        observation:
            Mapping containing the current machine observation.

        include_raw_features:
            If True, safe raw sensor measurements are included in the
            returned dictionary.

        impute_missing:
            If True, missing generated features are replaced using the
            configured training medians when available.

            If no training median exists, numeric missing values are
            replaced with 0.0.

            For production ML serving, supplying training-derived medians
            is strongly recommended.

        Returns
        -------
        dict
            Leakage-safe engineered features.

        Critical processing order
        -------------------------
        1. Read current observation.
        2. Calculate features from historical state.
        3. Calculate current physical features.
        4. Calculate current EWMA.
        5. Return features.
        6. ONLY THEN append current observation to machine state.

        This guarantees that lag/rolling historical statistics cannot
        accidentally use the current observation as historical data.
        """
        if not isinstance(
            observation,
            Mapping,
        ):
            raise TypeError(
                "observation must be a mapping/dictionary-like object."
            )

        machine_id = self._normalize_machine_id(
            observation.get(
                self.config.machine_id_column
            )
        )

        if machine_id not in self._states:
            self.start_machine(
                machine_id
            )

        state = self._states[
            machine_id
        ]

        timestamp = self._parse_timestamp(
            observation.get(
                self.config.timestamp_column
            )
        )

        sensors = self._extract_sensor_values(
            observation
        )

        features: dict[str, Any] = {}

        # ------------------------------------------------------------
        # Safe contextual metadata
        # ------------------------------------------------------------

        features[
            self.config.machine_id_column
        ] = machine_id

        # Do NOT expose UID as a predictive feature.
        #
        # If a real timestamp exists, preserve it as metadata.
        if timestamp is not None:
            features[
                self.config.timestamp_column
            ] = timestamp.isoformat()

        # ------------------------------------------------------------
        # Raw sensor values
        # ------------------------------------------------------------

        if include_raw_features:
            for sensor, value in sensors.items():
                features[sensor] = value

        # ------------------------------------------------------------
        # Time features
        # ------------------------------------------------------------

        self._add_time_features(
            features,
            timestamp,
            state,
        )

        # ------------------------------------------------------------
        # Physical features
        # ------------------------------------------------------------

        self._add_physical_features(
            features,
            sensors,
        )

        # ------------------------------------------------------------
        # Interaction features
        # ------------------------------------------------------------

        self._add_interaction_features(
            features,
            sensors,
        )

        # ------------------------------------------------------------
        # Historical lag features
        # ------------------------------------------------------------

        self._add_lag_features(
            features,
            state,
        )

        # ------------------------------------------------------------
        # Historical rolling statistics
        # ------------------------------------------------------------

        self._add_rolling_features(
            features,
            state,
        )

        # ------------------------------------------------------------
        # EWMA
        # ------------------------------------------------------------

        self._add_ewm_features(
            features,
            state,
            sensors,
        )

        # ------------------------------------------------------------
        # Rate of change
        # ------------------------------------------------------------

        self._add_rate_of_change_features(
            features,
            state,
            sensors,
            timestamp,
        )

        # ------------------------------------------------------------
        # Historical anomaly statistics
        # ------------------------------------------------------------

        self._add_anomaly_features(
            features,
            state,
            sensors,
        )

        # ------------------------------------------------------------
        # Missing-value indicators for raw sensors
        # ------------------------------------------------------------

        for sensor, value in sensors.items():
            features[
                f"{sensor}_missing"
            ] = float(
                not np.isfinite(value)
            )

        # ------------------------------------------------------------
        # Record feature names
        # ------------------------------------------------------------

        self._feature_names.update(
            features.keys()
        )

        # ------------------------------------------------------------
        # Optional imputation
        # ------------------------------------------------------------

        if impute_missing:
            features = self._impute_feature_dictionary(
                features
            )

        # ------------------------------------------------------------
        # CRITICAL:
        # Update state AFTER feature calculation.
        # ------------------------------------------------------------

        self._append_observation_to_state(
            state=state,
            timestamp=timestamp,
            sensors=sensors,
        )

        self._total_observations += 1

        self._last_update_time = timestamp

        return features

    # ------------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------------

    def _append_observation_to_state(
        self,
        *,
        state: MachineHistory,
        timestamp: Optional[pd.Timestamp],
        sensors: Mapping[str, float],
    ) -> None:
        """
        Append the current observation to machine history.

        This method runs only AFTER current features have been calculated.
        """
        state.timestamps.append(
            timestamp
        )

        for sensor in self.config.sensor_columns:
            value = sensors.get(
                sensor,
                float("nan"),
            )

            state.sensor_history[
                sensor
            ].append(value)

            # Update EWMAs.
            if np.isfinite(value):
                if sensor not in state.ewm_values:
                    state.ewm_values[
                        sensor
                    ] = {}

                for span in self.config.ewm_spans:
                    previous = state.ewm_values[
                        sensor
                    ].get(
                        span,
                        float("nan"),
                    )

                    updated = (
                        self._calculate_current_ewma(
                            value,
                            previous,
                            span,
                        )
                    )

                    state.ewm_values[
                        sensor
                    ][span] = updated

        state.observation_count += 1

        if timestamp is not None:
            state.last_timestamp = timestamp

    # ------------------------------------------------------------------------
    # Imputation
    # ------------------------------------------------------------------------

    def _impute_feature_dictionary(
        self,
        features: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Impute numeric missing values.

        Priority:

            1. Training-derived median
            2. Zero fallback

        This method never calculates an imputation statistic from current
        or future online observations.
        """
        result = dict(
            features
        )

        for name, value in result.items():
            if isinstance(
                value,
                (
                    bool,
                    np.bool_,
                ),
            ):
                continue

            if isinstance(
                value,
                (
                    int,
                    float,
                    np.integer,
                    np.floating,
                ),
            ):
                numeric = self._safe_float(
                    value
                )

                if np.isfinite(numeric):
                    result[name] = numeric

                else:
                    fallback = (
                        self._feature_medians.get(
                            name,
                            0.0,
                        )
                    )

                    result[name] = float(
                        fallback
                    )

        return result

    def set_feature_medians(
        self,
        medians: Mapping[str, float],
    ) -> None:
        """
        Set training-derived feature medians.

        This should normally be called with statistics fitted exclusively
        on the training data.
        """
        self._feature_medians = {
            str(key): float(value)
            for key, value in medians.items()
            if value is not None
            and np.isfinite(
                float(value)
            )
        }

    # ------------------------------------------------------------------------
    # Sequential DataFrame processing
    # ------------------------------------------------------------------------

    def transform(
        self,
        observations: pd.DataFrame,
        *,
        reset_state: bool = False,
        include_raw_features: bool = True,
        impute_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Process a DataFrame sequentially as if observations arrived online.

        Parameters
        ----------
        observations:
            DataFrame containing observations.

        reset_state:
            If True, clear all existing machine states before processing.

        include_raw_features:
            Include current raw sensor measurements.

        impute_missing:
            Apply configured training medians/zero fallback.

        Returns
        -------
        pd.DataFrame
            One engineered row per input observation.

        Important
        ---------
        The input order is preserved.

        For a real time-series dataset, observations should already be
        ordered chronologically within each machine before calling this
        method.

        The method does not sort the input automatically because online
        systems receive observations in arrival order and silently sorting
        can conceal upstream data-quality problems.
        """
        if not isinstance(
            observations,
            pd.DataFrame,
        ):
            raise TypeError(
                "observations must be a pandas DataFrame."
            )

        if observations.empty:
            return pd.DataFrame()

        if reset_state:
            self.reset_all()

        results: list[dict[str, Any]] = []

        for _, row in observations.iterrows():
            result = self.update(
                row.to_dict(),
                include_raw_features=include_raw_features,
                impute_missing=impute_missing,
            )

            results.append(
                result
            )

        result_df = pd.DataFrame(
            results
        )

        # Ensure deterministic column order.
        result_df = self._order_output_columns(
            result_df
        )

        return result_df

    # ------------------------------------------------------------------------
    # Output feature ordering
    # ------------------------------------------------------------------------

    def _order_output_columns(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return a deterministic feature-column ordering.
        """
        if df.empty:
            return df

        priority: list[str] = []

        if (
            self.config.machine_id_column
            in df.columns
        ):
            priority.append(
                self.config.machine_id_column
            )

        if (
            self.config.timestamp_column
            in df.columns
        ):
            priority.append(
                self.config.timestamp_column
            )

        # Raw sensors next.
        for sensor in self.config.sensor_columns:
            if sensor in df.columns:
                priority.append(
                    sensor
                )

        remaining = [
            column
            for column in df.columns
            if column not in priority
        ]

        return df[
            priority + sorted(remaining)
        ]

    # ------------------------------------------------------------------------
    # Feature inspection
    # ------------------------------------------------------------------------

    def get_feature_names(
        self,
    ) -> list[str]:
        """
        Return all feature names observed/generated so far.
        """
        return sorted(
            self._feature_names
        )

    def get_feature_count(
        self,
    ) -> int:
        """
        Return number of feature names observed so far.
        """
        return len(
            self._feature_names
        )

    # ------------------------------------------------------------------------
    # Machine state inspection
    # ------------------------------------------------------------------------

    def get_machine_state_summary(
        self,
        machine_id: str,
    ) -> dict[str, Any]:
        """
        Return a safe diagnostic summary for one machine.
        """
        machine_key = self._normalize_machine_id(
            machine_id
        )

        state = self._states.get(
            machine_key
        )

        if state is None:
            return {
                "machine_id": machine_key,
                "exists": False,
                "observation_count": 0,
                "history_size": 0,
                "last_timestamp": None,
            }

        return {
            "machine_id": machine_key,
            "exists": True,
            "observation_count": int(
                state.observation_count
            ),
            "history_size": int(
                len(state.timestamps)
            ),
            "last_timestamp": (
                state.last_timestamp.isoformat()
                if state.last_timestamp is not None
                else None
            ),
            "tracked_sensors": list(
                self.config.sensor_columns
            ),
        }

    def get_state_summary(
        self,
    ) -> dict[str, Any]:
        """
        Return overall online-engine state.
        """
        return {
            "machine_count": len(
                self._states
            ),
            "machine_ids": self.machine_ids(),
            "total_observations": int(
                self._total_observations
            ),
            "feature_count": int(
                self.get_feature_count()
            ),
            "last_update_time": (
                self._last_update_time.isoformat()
                if self._last_update_time is not None
                else None
            ),
            "config": {
                "lag_periods": list(
                    self.config.lag_periods
                ),
                "rolling_windows": list(
                    self.config.rolling_windows
                ),
                "ewm_spans": list(
                    self.config.ewm_spans
                ),
                "timestamp_column": (
                    self.config.timestamp_column
                ),
                "machine_id_column": (
                    self.config.machine_id_column
                ),
                "sensor_columns": list(
                    self.config.sensor_columns
                ),
            },
        }

    # ------------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------------

    def export_state(
        self,
    ) -> dict[str, Any]:
        """
        Export online state into a JSON-serializable dictionary.

        This is useful for restarting a long-running prediction service
        without losing machine history.
        """
        machines: dict[str, Any] = {}

        for machine_id, state in self._states.items():
            machines[machine_id] = {
                "timestamps": [
                    timestamp.isoformat()
                    if timestamp is not None
                    else None
                    for timestamp in state.timestamps
                ],
                "sensor_history": {
                    sensor: [
                        (
                            float(value)
                            if value is not None
                            and np.isfinite(value)
                            else None
                        )
                        for value in history
                    ]
                    for sensor, history in (
                        state.sensor_history.items()
                    )
                },
                "ewm_values": {
                    sensor: {
                        str(span): (
                            float(value)
                            if value is not None
                            and np.isfinite(value)
                            else None
                        )
                        for span, value in spans.items()
                    }
                    for sensor, spans in (
                        state.ewm_values.items()
                    )
                },
                "observation_count": int(
                    state.observation_count
                ),
                "last_timestamp": (
                    state.last_timestamp.isoformat()
                    if state.last_timestamp is not None
                    else None
                ),
            }

        return {
            "version": 1,
            "config": {
                "lag_periods": list(
                    self.config.lag_periods
                ),
                "rolling_windows": list(
                    self.config.rolling_windows
                ),
                "ewm_spans": list(
                    self.config.ewm_spans
                ),
                "rolling_min_periods": (
                    self.config.rolling_min_periods
                ),
                "zscore_min_history": (
                    self.config.zscore_min_history
                ),
                "epsilon": self.config.epsilon,
                "max_history_per_machine": (
                    self.config.max_history_per_machine
                ),
                "timestamp_column": (
                    self.config.timestamp_column
                ),
                "machine_id_column": (
                    self.config.machine_id_column
                ),
                "sensor_columns": list(
                    self.config.sensor_columns
                ),
            },
            "feature_medians": dict(
                self._feature_medians
            ),
            "feature_names": sorted(
                self._feature_names
            ),
            "total_observations": int(
                self._total_observations
            ),
            "last_update_time": (
                self._last_update_time.isoformat()
                if self._last_update_time is not None
                else None
            ),
            "machines": machines,
        }

    def save_state(
        self,
        path: str | Path,
    ) -> Path:
        """
        Persist online state to a JSON file.
        """
        destination = Path(path)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        state = self.export_state()

        with destination.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                indent=2,
                ensure_ascii=False,
            )

        LOGGER.info(
            "Real-time feature state saved to %s",
            destination,
        )

        return destination

    def import_state(
        self,
        state: Mapping[str, Any],
        *,
        replace_existing: bool = True,
    ) -> None:
        """
        Restore online state from an exported dictionary.
        """
        if not isinstance(
            state,
            Mapping,
        ):
            raise TypeError(
                "state must be a mapping."
            )

        version = state.get(
            "version"
        )

        if version != 1:
            raise ValueError(
                f"Unsupported real-time state version: {version}"
            )

        if replace_existing:
            self.reset_all()

        medians = state.get(
            "feature_medians",
            {},
        )

        if isinstance(
            medians,
            Mapping,
        ):
            self.set_feature_medians(
                medians
            )

        feature_names = state.get(
            "feature_names",
            [],
        )

        self._feature_names = set(
            str(name)
            for name in feature_names
        )

        self._total_observations = int(
            state.get(
                "total_observations",
                0,
            )
        )

        last_update_time = state.get(
            "last_update_time"
        )

        if last_update_time:
            self._last_update_time = (
                self._parse_timestamp(
                    last_update_time
                )
            )

        machines = state.get(
            "machines",
            {},
        )

        if not isinstance(
            machines,
            Mapping,
        ):
            raise ValueError(
                "Invalid machine-state structure."
            )

        for machine_id, machine_data in machines.items():
            if not isinstance(
                machine_data,
                Mapping,
            ):
                continue

            state_object = (
                self._create_machine_state()
            )

            # --------------------------------------------------------
            # Timestamp history
            # --------------------------------------------------------

            timestamp_values = (
                machine_data.get(
                    "timestamps",
                    [],
                )
            )

            for timestamp_value in timestamp_values:
                state_object.timestamps.append(
                    self._parse_timestamp(
                        timestamp_value
                    )
                )

            # --------------------------------------------------------
            # Sensor history
            # --------------------------------------------------------

            sensor_history = machine_data.get(
                "sensor_history",
                {},
            )

            if isinstance(
                sensor_history,
                Mapping,
            ):
                for sensor, values in (
                    sensor_history.items()
                ):
                    if sensor not in state_object.sensor_history:
                        state_object.sensor_history[
                            sensor
                        ] = deque(
                            maxlen=(
                                self.config
                                .max_history_per_machine
                            )
                        )

                    for value in values:
                        if value is None:
                            state_object.sensor_history[
                                sensor
                            ].append(
                                float("nan")
                            )
                        else:
                            state_object.sensor_history[
                                sensor
                            ].append(
                                self._safe_float(
                                    value
                                )
                            )

            # --------------------------------------------------------
            # EWMA state
            # --------------------------------------------------------

            ewm_values = machine_data.get(
                "ewm_values",
                {},
            )

            if isinstance(
                ewm_values,
                Mapping,
            ):
                for sensor, span_values in (
                    ewm_values.items()
                ):
                    state_object.ewm_values[
                        sensor
                    ] = {}

                    if isinstance(
                        span_values,
                        Mapping,
                    ):
                        for span, value in (
                            span_values.items()
                        ):
                            numeric_value = (
                                self._safe_float(
                                    value
                                )
                            )

                            state_object.ewm_values[
                                sensor
                            ][int(span)] = (
                                numeric_value
                            )

            state_object.observation_count = int(
                machine_data.get(
                    "observation_count",
                    0,
                )
            )

            last_timestamp = machine_data.get(
                "last_timestamp"
            )

            state_object.last_timestamp = (
                self._parse_timestamp(
                    last_timestamp
                )
                if last_timestamp
                else None
            )

            self._states[
                self._normalize_machine_id(
                    machine_id
                )
            ] = state_object

    def load_state(
        self,
        path: str | Path,
        *,
        replace_existing: bool = True,
    ) -> None:
        """
        Restore state from a JSON file.
        """
        source = Path(path)

        if not source.exists():
            raise FileNotFoundError(
                f"Real-time state file does not exist: {source}"
            )

        with source.open(
            "r",
            encoding="utf-8",
        ) as file:
            state = json.load(file)

        self.import_state(
            state,
            replace_existing=replace_existing,
        )

        LOGGER.info(
            "Real-time feature state loaded from %s",
            source,
        )

    # ------------------------------------------------------------------------
    # Batch-to-online warm-start
    # ------------------------------------------------------------------------

    def warm_start(
        self,
        historical_data: pd.DataFrame,
        *,
        reset_state: bool = False,
    ) -> None:
        """
        Warm-start online state using historical observations.

        This is useful when a production system starts while a machine
        already has history.

        IMPORTANT:
            This method updates state but intentionally does not return
            model features.

        The historical data must be ordered chronologically within each
        machine before calling this method.

        Parameters
        ----------
        historical_data:
            Historical machine observations.

        reset_state:
            Clear existing state before warm-starting.
        """
        if not isinstance(
            historical_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "historical_data must be a pandas DataFrame."
            )

        if historical_data.empty:
            return

        if reset_state:
            self.reset_all()

        for _, row in historical_data.iterrows():
            # update() calculates features first and then updates state.
            # We intentionally discard the features here.
            self.update(
                row.to_dict(),
                include_raw_features=False,
                impute_missing=False,
            )

    # ------------------------------------------------------------------------
    # Validation utilities
    # ------------------------------------------------------------------------

    def validate_observation(
        self,
        observation: Mapping[str, Any],
    ) -> list[str]:
        """
        Validate an incoming observation without changing state.

        Returns a list of warnings/errors that can be logged by an upstream
        service.
        """
        if not isinstance(
            observation,
            Mapping,
        ):
            raise TypeError(
                "observation must be a mapping."
            )

        issues: list[str] = []

        machine_column = (
            self.config.machine_id_column
        )

        if machine_column not in observation:
            issues.append(
                f"Missing machine identifier column: "
                f"'{machine_column}'."
            )

        timestamp_column = (
            self.config.timestamp_column
        )

        if timestamp_column not in observation:
            if self.config.add_time_features:
                issues.append(
                    f"Timestamp column '{timestamp_column}' "
                    "is missing. Time-based features will be unavailable."
                )

        for sensor in self.config.sensor_columns:
            if sensor not in observation:
                issues.append(
                    f"Sensor '{sensor}' is missing."
                )

        # Explicitly identify forbidden leakage fields.
        for column in (
            *AI4I_TARGET_COLUMNS,
            *AI4I_FAILURE_MODE_COLUMNS,
        ):
            if column in observation:
                issues.append(
                    f"Leakage-prone target/failure column '{column}' "
                    "is present. It will not be used by the feature "
                    "engineering calculations."
                )

        return issues

    # ------------------------------------------------------------------------
    # Feature schema helper
    # ------------------------------------------------------------------------

    def expected_feature_names(
        self,
    ) -> list[str]:
        """
        Generate the expected feature schema without requiring observations.

        This provides a deterministic baseline for downstream model-serving
        validation.
        """
        names: set[str] = set()

        names.add(
            self.config.machine_id_column
        )

        if self.config.add_time_features:
            names.update(
                {
                    self.config.timestamp_column,
                    "time_hour",
                    "time_minute",
                    "time_day_of_week",
                    "time_day_of_month",
                    "time_day_of_year",
                    "time_month",
                    "time_quarter",
                    "time_week_of_year",
                    "time_is_weekend",
                    "time_since_previous_observation_seconds",
                    "elapsed_time_since_machine_start_seconds",
                }
            )

        # Raw sensors.
        names.update(
            self.config.sensor_columns
        )

        # Missing indicators.
        for sensor in self.config.sensor_columns:
            names.add(
                f"{sensor}_missing"
            )

        # Physical features.
        if self.config.add_physical_features:
            names.update(
                {
                    "temperature_gap_k",
                    "temperature_ratio",
                    "torque_speed_interaction",
                    "rotational_speed_rad_per_s",
                    "mechanical_power_w",
                    "mechanical_power_kw",
                    "mechanical_power_to_nominal_ratio",
                    "torque_per_rpm",
                    "rpm_per_torque",
                    "tool_wear_squared",
                    "tool_wear_sqrt",
                    "tool_wear_log1p",
                    "tool_wear_fraction_of_240_min",
                    "rotational_speed_k_rpm",
                    "torque_k_nm",
                }
            )

        # Interactions.
        if self.config.add_interaction_features:
            names.update(
                {
                    "temperature_gap_x_torque",
                    "temperature_gap_x_speed",
                    "temperature_gap_x_tool_wear",
                    "mechanical_power_x_tool_wear",
                    "torque_x_tool_wear",
                    "speed_x_tool_wear",
                }
            )

        # Lags.
        if self.config.add_lag_features:
            for sensor in self.config.sensor_columns:
                for lag in self.config.lag_periods:
                    names.add(
                        f"{sensor}_lag_{lag}"
                    )

        # Rolling statistics.
        if self.config.add_rolling_features:
            for sensor in self.config.sensor_columns:
                for window in self.config.rolling_windows:
                    prefix = (
                        f"{sensor}_rolling_{window}"
                    )

                    names.update(
                        {
                            f"{prefix}_mean",
                            f"{prefix}_std",
                            f"{prefix}_min",
                            f"{prefix}_max",
                            f"{prefix}_median",
                        }
                    )

        # EWMAs.
        if self.config.add_ewm_features:
            for sensor in self.config.sensor_columns:
                for span in self.config.ewm_spans:
                    names.add(
                        f"{sensor}_ewm_{span}"
                    )

        # Rates of change.
        if self.config.add_rate_of_change:
            for sensor in self.config.sensor_columns:
                for lag in self.config.lag_periods:
                    names.add(
                        f"{sensor}_roc_{lag}"
                    )

        # Anomaly features.
        if self.config.add_anomaly_features:
            for sensor in self.config.sensor_columns:
                for window in self.config.rolling_windows:
                    names.update(
                        {
                            (
                                f"{sensor}_rolling_"
                                f"{window}_zscore"
                            ),
                            (
                                f"{sensor}_rolling_"
                                f"{window}_anomaly_score"
                            ),
                        }
                    )

        return sorted(
            names
        )

    # ------------------------------------------------------------------------
    # Copying
    # ------------------------------------------------------------------------

    def clone(
        self,
        *,
        include_state: bool = False,
    ) -> "RealTimeFeatureEngineer":
        """
        Create an independent copy.

        By default the clone starts without machine state.
        """
        clone = RealTimeFeatureEngineer(
            config=self.config,
            feature_medians=self._feature_medians,
        )

        clone._feature_names = set(
            self._feature_names
        )

        if include_state:
            clone._states = copy.deepcopy(
                self._states
            )

            clone._total_observations = (
                self._total_observations
            )

            clone._last_update_time = (
                self._last_update_time
            )

        return clone


# ============================================================================
# Convenience functions
# ============================================================================

def create_ai4i_realtime_engineer(
    *,
    feature_medians: Optional[
        Mapping[str, float]
    ] = None,
    max_history_per_machine: int = 128,
) -> RealTimeFeatureEngineer:
    """
    Create a ready-to-use real-time engineer for AI4I-style sensor data.

    Since AI4I does not contain a real machine timestamp or machine ID,
    this configuration expects a production deployment to supply those
    fields separately.

    Example
    -------
    >>> engineer = create_ai4i_realtime_engineer()
    """
    config = RealTimeFeatureConfig(
        max_history_per_machine=(
            max_history_per_machine
        ),
        sensor_columns=AI4I_SENSOR_COLUMNS,
    )

    return RealTimeFeatureEngineer(
        config=config,
        feature_medians=feature_medians,
    )


def process_realtime_observation(
    engineer: RealTimeFeatureEngineer,
    observation: Mapping[str, Any],
    *,
    impute_missing: bool = False,
) -> dict[str, Any]:
    """
    Convenience wrapper for one online observation.
    """
    if not isinstance(
        engineer,
        RealTimeFeatureEngineer,
    ):
        raise TypeError(
            "engineer must be a RealTimeFeatureEngineer."
        )

    return engineer.update(
        observation,
        include_raw_features=True,
        impute_missing=impute_missing,
    )


def process_realtime_dataframe(
    engineer: RealTimeFeatureEngineer,
    observations: pd.DataFrame,
    *,
    reset_state: bool = False,
    impute_missing: bool = False,
) -> pd.DataFrame:
    """
    Convenience wrapper for sequential DataFrame processing.
    """
    if not isinstance(
        engineer,
        RealTimeFeatureEngineer,
    ):
        raise TypeError(
            "engineer must be a RealTimeFeatureEngineer."
        )

    return engineer.transform(
        observations,
        reset_state=reset_state,
        include_raw_features=True,
        impute_missing=impute_missing,
    )


# ============================================================================
# Module exports
# ============================================================================

__all__ = [
    "AI4I_TARGET_COLUMNS",
    "AI4I_FAILURE_MODE_COLUMNS",
    "AI4I_IDENTIFIER_COLUMNS",
    "AI4I_SENSOR_COLUMNS",
    "RealTimeFeatureConfig",
    "MachineHistory",
    "RealTimeFeatureEngineer",
    "create_ai4i_realtime_engineer",
    "process_realtime_observation",
    "process_realtime_dataframe",
]


# ============================================================================
# Demonstration / self-test
# ============================================================================

def _run_self_test() -> None:
    """
    Lightweight internal self-test.

    This is intentionally independent of the external AI4I CSV so that the
    module can be tested immediately after installation.
    """
    config = RealTimeFeatureConfig(
        lag_periods=(1, 2),
        rolling_windows=(3,),
        ewm_spans=(3,),
        rolling_min_periods=2,
        zscore_min_history=3,
        max_history_per_machine=16,
    )

    engineer = RealTimeFeatureEngineer(
        config=config
    )

    observations = [
        {
            "machine_id": "M001",
            "timestamp": "2026-01-01T10:00:00Z",
            "air_temperature_k": 298.0,
            "process_temperature_k": 308.0,
            "rotational_speed_rpm": 1500.0,
            "torque_nm": 40.0,
            "tool_wear_min": 100.0,
        },
        {
            "machine_id": "M001",
            "timestamp": "2026-01-01T10:01:00Z",
            "air_temperature_k": 298.2,
            "process_temperature_k": 308.3,
            "rotational_speed_rpm": 1510.0,
            "torque_nm": 41.0,
            "tool_wear_min": 101.0,
        },
        {
            "machine_id": "M001",
            "timestamp": "2026-01-01T10:02:00Z",
            "air_temperature_k": 298.4,
            "process_temperature_k": 308.6,
            "rotational_speed_rpm": 1520.0,
            "torque_nm": 42.0,
            "tool_wear_min": 102.0,
        },
        {
            "machine_id": "M001",
            "timestamp": "2026-01-01T10:03:00Z",
            "air_temperature_k": 298.6,
            "process_temperature_k": 308.8,
            "rotational_speed_rpm": 1530.0,
            "torque_nm": 43.0,
            "tool_wear_min": 103.0,
        },
    ]

    generated = []

    for observation in observations:
        features = engineer.update(
            observation,
            include_raw_features=True,
            impute_missing=False,
        )

        generated.append(
            features
        )

    assert len(generated) == 4

    # The second observation must have access to the first observation's
    # lag-1 value.
    second = generated[1]

    assert (
        second[
            "rotational_speed_rpm_lag_1"
        ]
        == 1500.0
    )

    # The fourth observation has enough history for a rolling mean with
    # window=3.
    fourth = generated[3]

    assert np.isfinite(
        fourth[
            "rotational_speed_rpm_rolling_3_mean"
        ]
    )

    # A physical power feature must be generated.
    assert np.isfinite(
        fourth[
            "mechanical_power_w"
        ]
    )

    # Rate of change must use the actual 60-second interval.
    assert np.isfinite(
        fourth[
            "rotational_speed_rpm_roc_1"
        ]
    )

    # Current observation must NOT appear as its own lag.
    assert (
        fourth[
            "rotational_speed_rpm_lag_1"
        ]
        == 1520.0
    )

    # Failure labels must not appear in the generated feature set.
    for forbidden_column in (
        *AI4I_TARGET_COLUMNS,
        *AI4I_FAILURE_MODE_COLUMNS,
    ):
        assert (
            forbidden_column
            not in fourth
        )

    # State should exist.
    summary = engineer.get_machine_state_summary(
        "M001"
    )

    assert summary["exists"] is True
    assert summary["observation_count"] == 4

    print(
        "RealTimeFeatureEngineer self-test passed."
    )


if __name__ == "__main__":
    _run_self_test()