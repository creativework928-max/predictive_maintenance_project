"""
Data loading and validation module.

Project:
    Predictive Maintenance Project

Task:
    Task 2 - Feature Engineering

Purpose:
    Robust data loading, normalization, validation, profiling, and
    diagnostics for the predictive maintenance pipeline.

Supported data sources:
    1. Local CSV file
    2. UCI Machine Learning Repository AI4I 2020 dataset

Supported operating modes:
    - auto
    - ai4i
    - timeseries

Important:
    The official AI4I 2020 dataset uses "UDI" as its identifier column.
    Internally, this project uses the canonical name "UID".

    Therefore:

        UDI -> UID

    is handled automatically during column normalization.

    UDI/UID is an identifier and is NEVER treated as a timestamp.

    For genuine industrial time-series data, the pipeline expects an actual
    timestamp column and, preferably, a machine_id column.

Design goals:
    - Robust
    - Reproducible
    - Explicit validation
    - Leakage-aware
    - Extensible
    - Production-oriented
"""

from __future__ import annotations

import io
import json
import logging
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION IMPORTS
# ============================================================================

try:
    from .config import (
        AI4I_THRESHOLDS,
        CATEGORICAL_COLUMNS,
        DATA_MODE,
        DATASET_DOI,
        DATASET_NAME,
        DATASET_SOURCE,
        DATASET_URL,
        EXCLUDED_FROM_FEATURE_GENERATION,
        FAILURE_MODE_COLUMNS,
        ID_COLUMNS,
        MACHINE_ID_COLUMN,
        NUMERIC_SENSOR_COLUMNS,
        PROJECT_ROOT,
        RAW_DATA_PATH,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
    )

except ImportError:
    from config import (
        AI4I_THRESHOLDS,
        CATEGORICAL_COLUMNS,
        DATA_MODE,
        DATASET_DOI,
        DATASET_NAME,
        DATASET_SOURCE,
        DATASET_URL,
        EXCLUDED_FROM_FEATURE_GENERATION,
        FAILURE_MODE_COLUMNS,
        ID_COLUMNS,
        MACHINE_ID_COLUMN,
        NUMERIC_SENSOR_COLUMNS,
        PROJECT_ROOT,
        RAW_DATA_PATH,
        TARGET_COLUMN,
        TIMESTAMP_COLUMN,
    )


# ============================================================================
# LOGGING
# ============================================================================

LOGGER = logging.getLogger(__name__)


# ============================================================================
# UCI CONFIGURATION
# ============================================================================

UCI_DATASET_ID = 601

UCI_CSV_URLS = [
    (
        "https://archive.ics.uci.edu/ml/"
        "machine-learning-databases/00601/"
        "ai4i2020.csv"
    ),
    (
        "https://archive.ics.uci.edu/static/public/"
        "601/ai4i2020.zip"
    ),
]


# ============================================================================
# AI4I EXPECTED SCHEMA
# ============================================================================

# IMPORTANT:
#
# The official AI4I 2020 dataset contains:
#
#     UDI
#
# not:
#
#     UID
#
# We intentionally use UID internally because the rest of this project
# already expects that canonical name.
#
# COLUMN_ALIASES below maps:
#
#     UDI -> UID
#
# before schema detection occurs.

AI4I_EXPECTED_COLUMNS = [
    "UID",
    "Product ID",
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


AI4I_NUMERIC_COLUMNS = [
    "UID",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
]


AI4I_CATEGORICAL_COLUMNS = [
    "Product ID",
    "Type",
]


# ============================================================================
# COLUMN ALIASES
# ============================================================================

COLUMN_ALIASES: Dict[str, str] = {

    # ------------------------------------------------------------------------
    # Identifiers
    # ------------------------------------------------------------------------

    # Official AI4I column.
    "udi": "UID",

    # Internal/project spelling.
    "uid": "UID",

    # Other identifier variations.
    "unique id": "UID",
    "unique_id": "UID",
    "unique identifier": "UID",
    "unique_identifier": "UID",

    # Product ID.
    "product id": "Product ID",
    "product_id": "Product ID",

    # ------------------------------------------------------------------------
    # Product type
    # ------------------------------------------------------------------------

    "type": "Type",
    "product type": "Type",
    "product_type": "Type",

    # ------------------------------------------------------------------------
    # Temperature
    # ------------------------------------------------------------------------

    "air temperature": "Air temperature [K]",
    "air_temperature": "Air temperature [K]",
    "air temperature [k]": "Air temperature [K]",
    "air_temperature_k": "Air temperature [K]",
    "air temp": "Air temperature [K]",
    "air_temp": "Air temperature [K]",

    "process temperature": "Process temperature [K]",
    "process_temperature": "Process temperature [K]",
    "process temperature [k]": "Process temperature [K]",
    "process_temperature_k": "Process temperature [K]",
    "process temp": "Process temperature [K]",
    "process_temp": "Process temperature [K]",

    # ------------------------------------------------------------------------
    # Mechanical sensors
    # ------------------------------------------------------------------------

    "rotational speed": "Rotational speed [rpm]",
    "rotational_speed": "Rotational speed [rpm]",
    "rotational speed [rpm]": "Rotational speed [rpm]",
    "rotational_speed_rpm": "Rotational speed [rpm]",
    "rpm": "Rotational speed [rpm]",

    "torque": "Torque [Nm]",
    "torque [nm]": "Torque [Nm]",
    "torque_nm": "Torque [Nm]",

    "tool wear": "Tool wear [min]",
    "tool_wear": "Tool wear [min]",
    "tool wear [min]": "Tool wear [min]",
    "tool_wear_min": "Tool wear [min]",

    # ------------------------------------------------------------------------
    # Target
    # ------------------------------------------------------------------------

    "machine failure": "Machine failure",
    "machine_failure": "Machine failure",
    "failure": "Machine failure",
    "failure flag": "Machine failure",
    "failure_flag": "Machine failure",

    # ------------------------------------------------------------------------
    # Failure modes
    # ------------------------------------------------------------------------

    "twf": "TWF",
    "hdf": "HDF",
    "pwf": "PWF",
    "osf": "OSF",
    "rnf": "RNF",

    # ------------------------------------------------------------------------
    # Real-world time-series fields
    # ------------------------------------------------------------------------

    "timestamp": "timestamp",
    "date": "timestamp",
    "datetime": "timestamp",
    "date_time": "timestamp",
    "time": "timestamp",

    # ------------------------------------------------------------------------
    # Machine identifier
    # ------------------------------------------------------------------------

    "machine id": "machine_id",
    "machine_id": "machine_id",
    "machineid": "machine_id",
    "machine identifier": "machine_id",
    "machine_identifier": "machine_id",

    "asset id": "machine_id",
    "asset_id": "machine_id",

    "equipment id": "machine_id",
    "equipment_id": "machine_id",
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DatasetProfile:
    """
    Structured summary of a loaded dataset.
    """

    dataset_name: str
    source: str
    mode: str

    rows: int
    columns: int

    column_names: List[str] = field(default_factory=list)

    numeric_columns: List[str] = field(default_factory=list)

    categorical_columns: List[str] = field(default_factory=list)

    datetime_columns: List[str] = field(default_factory=list)

    missing_values: int = 0

    duplicate_rows: int = 0

    target_column: Optional[str] = None

    target_distribution: Dict[str, int] = field(
        default_factory=dict
    )

    machine_id_column: Optional[str] = None

    timestamp_column: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """
        Convert profile to a serializable dictionary.
        """

        return {
            "dataset_name": self.dataset_name,
            "source": self.source,
            "mode": self.mode,
            "rows": self.rows,
            "columns": self.columns,
            "column_names": self.column_names,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "missing_values": self.missing_values,
            "duplicate_rows": self.duplicate_rows,
            "target_column": self.target_column,
            "target_distribution": self.target_distribution,
            "machine_id_column": self.machine_id_column,
            "timestamp_column": self.timestamp_column,
        }


@dataclass
class LoadedDataset:
    """
    Container for loaded and validated dataset information.
    """

    data: pd.DataFrame

    profile: DatasetProfile

    source_path: Optional[Path] = None

    source_url: Optional[str] = None

    is_ai4i: bool = False

    is_true_timeseries: bool = False


# ============================================================================
# COLUMN NORMALIZATION
# ============================================================================

def _clean_column_string(column: object) -> str:
    """
    Normalize an individual column name.

    Handles:
        - UTF-8 BOM
        - leading/trailing whitespace
        - repeated whitespace
    """

    text = str(column)

    # Remove UTF-8 BOM.
    #
    # Some CSV files can have:
    #
    #     \ufeffUDI
    #
    # rather than:
    #
    #     UDI
    #
    # Without removing this character, the first column may not match
    # the "udi" alias.
    text = text.replace("\ufeff", "")

    text = text.strip()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text


def _alias_key(column: object) -> str:
    """
    Create a normalized key used for alias matching.
    """

    text = _clean_column_string(
        column
    )

    text = text.lower()

    text = text.replace(
        "_",
        " ",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_column_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize dataset column names using COLUMN_ALIASES.

    The original DataFrame is not modified.
    """

    result = df.copy()

    normalized_columns = []

    for column in result.columns:

        cleaned = _clean_column_string(
            column
        )

        key = _alias_key(
            cleaned
        )

        normalized = COLUMN_ALIASES.get(
            key,
            cleaned,
        )

        normalized_columns.append(
            normalized
        )

    result.columns = normalized_columns

    # Ensure normalization did not create duplicate names.
    if result.columns.duplicated().any():

        duplicated = (
            result.columns[
                result.columns.duplicated(
                    keep=False
                )
            ]
            .tolist()
        )

        raise ValueError(
            "Column normalization created duplicate "
            "column names: "
            f"{sorted(set(duplicated))}"
        )

    return result


# ============================================================================
# FILE LOADING
# ============================================================================

def load_csv(
    path: Path,
    *,
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """
    Load a CSV file.

    utf-8-sig is intentionally used by default so that a UTF-8 BOM in
    the first column is automatically removed by pandas.
    """

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset file was not found: {path}"
        )

    if not path.is_file():

        raise ValueError(
            f"Dataset path is not a file: {path}"
        )

    LOGGER.info(
        "Loading CSV dataset: %s",
        path,
    )

    try:

        df = pd.read_csv(
            path,
            encoding=encoding,
        )

    except UnicodeDecodeError:

        LOGGER.warning(
            "UTF-8 decoding failed. "
            "Retrying with latin-1."
        )

        df = pd.read_csv(
            path,
            encoding="latin-1",
        )

    except Exception as exc:

        raise ValueError(
            f"Unable to read CSV file '{path}': {exc}"
        ) from exc

    if df.empty:

        raise ValueError(
            f"The dataset loaded from '{path}' is empty."
        )

    LOGGER.info(
        "Loaded dataset with %d rows and %d columns.",
        df.shape[0],
        df.shape[1],
    )

    return df


def download_file(
    url: str,
    *,
    timeout: int = 60,
) -> bytes:
    """
    Download bytes from a URL.
    """

    LOGGER.info(
        "Downloading dataset from: %s",
        url,
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "PredictiveMaintenanceFeatureEngineering/"
                "1.0"
            )
        },
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:

            return response.read()

    except Exception as exc:

        raise RuntimeError(
            f"Unable to download dataset from "
            f"'{url}': {exc}"
        ) from exc


def load_ai4i_from_uci_direct(
    *,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    Download AI4I directly from UCI.

    Tries both the direct CSV URL and the ZIP distribution.
    """

    errors: List[str] = []

    for url in UCI_CSV_URLS:

        try:

            content = download_file(
                url,
                timeout=timeout,
            )

            # ----------------------------------------------------------------
            # Direct CSV
            # ----------------------------------------------------------------

            if url.lower().endswith(
                ".csv"
            ):

                df = pd.read_csv(
                    io.BytesIO(content),
                    encoding="utf-8-sig",
                )

                if not df.empty:

                    LOGGER.info(
                        "Successfully downloaded "
                        "AI4I dataset from UCI."
                    )

                    return df

            # ----------------------------------------------------------------
            # ZIP archive
            # ----------------------------------------------------------------

            if url.lower().endswith(
                ".zip"
            ):

                import zipfile

                with zipfile.ZipFile(
                    io.BytesIO(content)
                ) as archive:

                    csv_candidates = [
                        name
                        for name in archive.namelist()
                        if name.lower().endswith(
                            ".csv"
                        )
                    ]

                    if not csv_candidates:

                        raise RuntimeError(
                            "UCI ZIP archive does not "
                            "contain a CSV file."
                        )

                    preferred = [
                        name
                        for name in csv_candidates
                        if "ai4i2020" in name.lower()
                    ]

                    selected = (
                        preferred[0]
                        if preferred
                        else csv_candidates[0]
                    )

                    with archive.open(
                        selected
                    ) as csv_file:

                        df = pd.read_csv(
                            csv_file,
                            encoding="utf-8-sig",
                        )

                    if not df.empty:

                        LOGGER.info(
                            "Successfully extracted "
                            "AI4I dataset from UCI archive."
                        )

                        return df

        except Exception as exc:

            errors.append(
                f"{url}: {exc}"
            )

            LOGGER.warning(
                "UCI download attempt failed: %s",
                exc,
            )

    raise RuntimeError(
        "Unable to download AI4I dataset from UCI.\n"
        "Attempted URLs:\n"
        + "\n".join(errors)
    )


def load_ai4i_with_ucimlrepo() -> pd.DataFrame:
    """
    Load AI4I using the ucimlrepo package.
    """

    try:

        from ucimlrepo import fetch_ucirepo

    except ImportError as exc:

        raise ImportError(
            "The 'ucimlrepo' package is not installed. "
            "Install it with: pip install ucimlrepo"
        ) from exc

    LOGGER.info(
        "Fetching UCI dataset ID %d using ucimlrepo.",
        UCI_DATASET_ID,
    )

    try:

        dataset = fetch_ucirepo(
            id=UCI_DATASET_ID
        )

    except Exception as exc:

        raise RuntimeError(
            f"ucimlrepo could not fetch UCI dataset "
            f"{UCI_DATASET_ID}: {exc}"
        ) from exc

    features = dataset.data.features

    targets = dataset.data.targets

    if features is None:

        raise RuntimeError(
            "UCI dataset did not provide feature data."
        )

    if (
        targets is not None
        and not targets.empty
    ):

        target_columns = [
            column
            for column in targets.columns
            if column not in features.columns
        ]

        if target_columns:

            features = pd.concat(
                [
                    features.reset_index(
                        drop=True
                    ),
                    targets[
                        target_columns
                    ].reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            )

    return features.copy()


def load_ai4i(
    *,
    local_path: Optional[Path] = None,
    prefer_local: bool = True,
    use_ucimlrepo: bool = True,
    timeout: int = 60,
) -> Tuple[
    pd.DataFrame,
    Optional[Path],
    Optional[str],
]:
    """
    Load the AI4I 2020 dataset.

    Priority:
        1. Local CSV
        2. ucimlrepo
        3. Direct UCI download
    """

    path = (
        Path(local_path)
        if local_path is not None
        else Path(RAW_DATA_PATH)
    )

    # ------------------------------------------------------------------------
    # Local file
    # ------------------------------------------------------------------------

    if (
        prefer_local
        and path.exists()
    ):

        LOGGER.info(
            "Using local AI4I dataset: %s",
            path,
        )

        df = load_csv(
            path
        )

        return (
            df,
            path,
            None,
        )

    # ------------------------------------------------------------------------
    # ucimlrepo
    # ------------------------------------------------------------------------

    if use_ucimlrepo:

        try:

            df = load_ai4i_with_ucimlrepo()

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            df.to_csv(
                path,
                index=False,
            )

            LOGGER.info(
                "Saved downloaded AI4I dataset to %s",
                path,
            )

            return (
                df,
                path,
                DATASET_URL,
            )

        except Exception as exc:

            LOGGER.warning(
                "ucimlrepo loading failed: %s",
                exc,
            )

    # ------------------------------------------------------------------------
    # Direct UCI download
    # ------------------------------------------------------------------------

    df = load_ai4i_from_uci_direct(
        timeout=timeout
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    LOGGER.info(
        "Saved UCI dataset to %s",
        path,
    )

    return (
        df,
        path,
        DATASET_URL,
    )


# ============================================================================
# DATA TYPE CONVERSION
# ============================================================================

def convert_numeric_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Convert specified columns to numeric.
    """

    result = df.copy()

    for column in columns:

        if column not in result.columns:
            continue

        original = result[column]

        converted = pd.to_numeric(
            original,
            errors="coerce",
        )

        if strict:

            invalid_mask = (
                original.notna()
                & converted.isna()
            )

            if invalid_mask.any():

                invalid_count = int(
                    invalid_mask.sum()
                )

                raise ValueError(
                    f"Column '{column}' contains "
                    f"{invalid_count} values that cannot "
                    "be converted to numeric."
                )

        result[column] = converted

    return result


def convert_timestamp_column(
    df: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """
    Convert a timestamp column to pandas datetime.
    """

    result = df.copy()

    if column not in result.columns:
        return result

    LOGGER.info(
        "Converting timestamp column '%s'.",
        column,
    )

    result[column] = pd.to_datetime(
        result[column],
        errors="coerce",
        utc=True,
    )

    return result


# ============================================================================
# TIMESTAMP DETECTION
# ============================================================================

def detect_timestamp_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Detect a genuine timestamp column.

    Detection is intentionally conservative.

    UID/UDI is NEVER treated as a timestamp.
    """

    # ------------------------------------------------------------------------
    # Explicit configured timestamp
    # ------------------------------------------------------------------------

    if TIMESTAMP_COLUMN in df.columns:

        parsed = pd.to_datetime(
            df[TIMESTAMP_COLUMN],
            errors="coerce",
            utc=True,
        )

        valid_ratio = (
            parsed.notna().mean()
            if len(parsed) > 0
            else 0.0
        )

        if valid_ratio >= 0.95:

            return TIMESTAMP_COLUMN

    # ------------------------------------------------------------------------
    # Search obvious timestamp/date fields.
    # ------------------------------------------------------------------------

    candidates = []

    for column in df.columns:

        # Never treat identifiers as timestamps.
        if column in {
            "UID",
            "UDI",
            "Product ID",
        }:

            continue

        name = str(
            column
        ).lower()

        if not any(
            token in name
            for token in [
                "time",
                "date",
                "datetime",
            ]
        ):

            continue

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

        valid_ratio = (
            parsed.notna().mean()
            if len(parsed) > 0
            else 0.0
        )

        if valid_ratio >= 0.95:

            candidates.append(
                (
                    column,
                    valid_ratio,
                )
            )

    if candidates:

        candidates.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return candidates[0][0]

    return None


# ============================================================================
# MACHINE-ID DETECTION
# ============================================================================

def detect_machine_id_column(
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Detect a machine/equipment/asset identifier.
    """

    if MACHINE_ID_COLUMN in df.columns:

        return MACHINE_ID_COLUMN

    candidates = []

    for column in df.columns:

        name = str(
            column
        ).lower()

        if not any(
            token in name
            for token in [
                "machine",
                "equipment",
                "asset",
            ]
        ):

            continue

        if any(
            token in name
            for token in [
                "id",
                "identifier",
                "number",
            ]
        ):

            candidates.append(
                column
            )

    if candidates:

        return candidates[0]

    return None


# ============================================================================
# AI4I DETECTION
# ============================================================================

def is_ai4i_dataset(
    df: pd.DataFrame,
) -> bool:
    """
    Determine whether a DataFrame resembles AI4I 2020.

    The DataFrame should already have been normalized.

    The official UDI column has already been mapped to UID at this point.
    """

    columns = set(
        df.columns
    )

    required = {
        "UID",
        "Product ID",
        "Type",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    }

    return required.issubset(
        columns
    )


# ============================================================================
# MODE DETECTION
# ============================================================================

def detect_data_mode(
    df: pd.DataFrame,
    requested_mode: str = DATA_MODE,
) -> str:
    """
    Determine the correct operating mode.

    Returns:
        "ai4i" or "timeseries"
    """

    requested_mode = (
        str(requested_mode)
        .lower()
        .strip()
    )

    if requested_mode not in {
        "auto",
        "ai4i",
        "timeseries",
    }:

        raise ValueError(
            "requested_mode must be 'auto', "
            "'ai4i', or 'timeseries'."
        )

    ai4i = is_ai4i_dataset(
        df
    )

    timestamp_column = (
        detect_timestamp_column(
            df
        )
    )

    # ------------------------------------------------------------------------
    # Explicit AI4I mode
    # ------------------------------------------------------------------------

    if requested_mode == "ai4i":

        if not ai4i:

            missing = [
                column
                for column in AI4I_EXPECTED_COLUMNS
                if column not in df.columns
            ]

            raise ValueError(
                "DATA_MODE='ai4i' was requested, but "
                "the loaded dataset does not match the "
                "expected AI4I schema.\n\n"
                "Missing expected columns:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in missing
                )
                + "\n\n"
                "Note: The official AI4I identifier is "
                "'UDI'. The loader automatically maps "
                "'UDI' to internal 'UID'."
            )

        return "ai4i"

    # ------------------------------------------------------------------------
    # Explicit time-series mode
    # ------------------------------------------------------------------------

    if requested_mode == "timeseries":

        if timestamp_column is None:

            raise ValueError(
                "DATA_MODE='timeseries' was requested, "
                "but no valid timestamp column could "
                "be detected."
            )

        return "timeseries"

    # ------------------------------------------------------------------------
    # Auto mode
    # ------------------------------------------------------------------------

    if ai4i:

        return "ai4i"

    if timestamp_column is not None:

        return "timeseries"

    raise ValueError(
        "Unable to determine dataset mode automatically.\n\n"
        "The dataset is neither recognized as AI4I nor does "
        "it contain a valid timestamp column.\n\n"
        "Detected columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in df.columns
        )
        + "\n\n"
        "For AI4I 2020, the official identifier is 'UDI'. "
        "The loader maps 'UDI' to the internal canonical "
        "name 'UID'."
    )


# ============================================================================
# SCHEMA VALIDATION
# ============================================================================

def validate_ai4i_schema(
    df: pd.DataFrame,
) -> None:
    """
    Validate the AI4I dataset schema.
    """

    missing = [
        column
        for column in AI4I_EXPECTED_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "AI4I dataset is missing required columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )

    # The official AI4I dataset contains 10,000 observations.
    # A subset is allowed, so this is a warning rather than an error.

    if len(df) != 10000:

        LOGGER.warning(
            "AI4I dataset contains %d rows; official "
            "reference dataset contains 10,000 observations.",
            len(df),
        )


def validate_general_schema(
    df: pd.DataFrame,
) -> None:
    """
    Validate basic requirements for a general dataset.
    """

    if df.empty:

        raise ValueError(
            "The dataset contains no observations."
        )

    if len(df.columns) < 2:

        raise ValueError(
            "The dataset must contain at least "
            "two columns."
        )

    if df.columns.duplicated().any():

        duplicates = (
            df.columns[
                df.columns.duplicated(
                    keep=False
                )
            ]
            .tolist()
        )

        raise ValueError(
            "Dataset contains duplicate column names: "
            f"{sorted(set(duplicates))}"
        )


# ============================================================================
# DUPLICATE AND MISSING DATA
# ============================================================================

def calculate_missing_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate a column-level missing-value summary.
    """

    total = len(df)

    rows = []

    for column in df.columns:

        missing = int(
            df[column].isna().sum()
        )

        missing_ratio = (
            missing / total
            if total > 0
            else 0.0
        )

        rows.append(
            {
                "feature": column,
                "missing_count": missing,
                "missing_ratio": missing_ratio,
                "dtype": str(
                    df[column].dtype
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            "missing_ratio",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


def calculate_duplicate_summary(
    df: pd.DataFrame,
) -> Dict[str, int]:
    """
    Calculate duplicate row information.
    """

    duplicate_mask = df.duplicated(
        keep=False
    )

    duplicate_rows = int(
        duplicate_mask.sum()
    )

    duplicate_groups = int(
        df.duplicated().sum()
    )

    return {
        "duplicate_rows": duplicate_rows,
        "duplicate_groups": duplicate_groups,
    }


# ============================================================================
# RANGE VALIDATION
# ============================================================================

def validate_ai4i_ranges(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform non-destructive AI4I physical-range diagnostics.

    Observations are NOT deleted.
    """

    checks = []

    range_definitions = {
        "Air temperature [K]": {
            "min": 250.0,
            "max": 350.0,
        },
        "Process temperature [K]": {
            "min": 250.0,
            "max": 400.0,
        },
        "Rotational speed [rpm]": {
            "min": 0.0,
            "max": 10000.0,
        },
        "Torque [Nm]": {
            "min": 0.0,
            "max": 500.0,
        },
        "Tool wear [min]": {
            "min": 0.0,
            "max": 1000.0,
        },
    }

    for column, bounds in range_definitions.items():

        if column not in df.columns:

            continue

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_mask = (
            (values < bounds["min"])
            | (values > bounds["max"])
        )

        invalid_mask = invalid_mask.fillna(
            False
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        checks.append(
            {
                "feature": column,
                "expected_min": bounds["min"],
                "expected_max": bounds["max"],
                "observed_min": (
                    float(values.min())
                    if values.notna().any()
                    else np.nan
                ),
                "observed_max": (
                    float(values.max())
                    if values.notna().any()
                    else np.nan
                ),
                "invalid_count": invalid_count,
                "valid": (
                    invalid_count == 0
                ),
            }
        )

    return pd.DataFrame(
        checks
    )


# ============================================================================
# TARGET VALIDATION
# ============================================================================

def validate_binary_target(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> None:
    """
    Validate a binary target column.
    """

    if target_column not in df.columns:

        LOGGER.warning(
            "Target column '%s' is not present.",
            target_column,
        )

        return

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    if target.isna().any():

        raise ValueError(
            f"Target column '{target_column}' contains "
            "non-numeric or missing values."
        )

    unique_values = set(
        target.unique()
    )

    if not unique_values.issubset(
        {0, 1}
    ):

        raise ValueError(
            f"Target column '{target_column}' must contain "
            f"only 0/1 values. Found: "
            f"{sorted(unique_values)}"
        )


# ============================================================================
# LEAKAGE VALIDATION
# ============================================================================

def validate_leakage_columns(
    df: pd.DataFrame,
) -> Dict[str, object]:
    """
    Report columns that should be excluded from predictive feature generation.

    This function does not delete any columns.
    """

    present_excluded = [
        column
        for column in EXCLUDED_FROM_FEATURE_GENERATION
        if column in df.columns
    ]

    return {
        "excluded_columns_present": (
            present_excluded
        ),
        "target_present": (
            TARGET_COLUMN in df.columns
        ),
        "failure_mode_columns_present": [
            column
            for column in FAILURE_MODE_COLUMNS
            if column in df.columns
        ],
    }


# ============================================================================
# DATASET PROFILING
# ============================================================================

def profile_dataset(
    df: pd.DataFrame,
    *,
    dataset_name: str = DATASET_NAME,
    source: str = DATASET_SOURCE,
    mode: Optional[str] = None,
) -> DatasetProfile:
    """
    Create a detailed dataset profile.
    """

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        df.select_dtypes(
            include=[
                "object",
                "category",
                "string",
            ]
        )
        .columns
        .tolist()
    )

    datetime_columns = (
        df.select_dtypes(
            include=[
                "datetime",
                "datetimetz",
            ]
        )
        .columns
        .tolist()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    target_distribution: Dict[
        str,
        int,
    ] = {}

    if TARGET_COLUMN in df.columns:

        target_values = (
            df[TARGET_COLUMN]
            .value_counts(
                dropna=False
            )
            .sort_index()
        )

        target_distribution = {
            str(index): int(value)
            for index, value
            in target_values.items()
        }

    machine_id_column = (
        detect_machine_id_column(
            df
        )
    )

    timestamp_column = (
        detect_timestamp_column(
            df
        )
    )

    return DatasetProfile(
        dataset_name=dataset_name,
        source=source,
        mode=mode or "unknown",
        rows=int(
            df.shape[0]
        ),
        columns=int(
            df.shape[1]
        ),
        column_names=df.columns.tolist(),
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        datetime_columns=datetime_columns,
        missing_values=int(
            df.isna().sum().sum()
        ),
        duplicate_rows=duplicate_rows,
        target_column=(
            TARGET_COLUMN
            if TARGET_COLUMN in df.columns
            else None
        ),
        target_distribution=(
            target_distribution
        ),
        machine_id_column=(
            machine_id_column
        ),
        timestamp_column=(
            timestamp_column
        ),
    )


# ============================================================================
# AI4I FAILURE LOGIC DIAGNOSTICS
# ============================================================================

def calculate_ai4i_failure_logic_diagnostics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate diagnostics corresponding to documented AI4I conditions.

    These diagnostics are NOT automatically used as predictive features.

    Failure conditions:
        HDF:
            Temperature difference < 8.6 K
            and rotational speed < 1380 rpm

        PWF:
            Power outside approximately 3500-9000 W

        TWF:
            Tool wear approximately 200-240 minutes

        OSF:
            Tool wear * torque exceeds product-specific threshold
    """

    if not is_ai4i_dataset(df):

        return pd.DataFrame()

    result = pd.DataFrame(
        index=df.index
    )

    air_temperature = pd.to_numeric(
        df["Air temperature [K]"],
        errors="coerce",
    )

    process_temperature = pd.to_numeric(
        df["Process temperature [K]"],
        errors="coerce",
    )

    rpm = pd.to_numeric(
        df["Rotational speed [rpm]"],
        errors="coerce",
    )

    torque = pd.to_numeric(
        df["Torque [Nm]"],
        errors="coerce",
    )

    tool_wear = pd.to_numeric(
        df["Tool wear [min]"],
        errors="coerce",
    )

    temperature_gap = (
        process_temperature
        - air_temperature
    )

    angular_velocity = (
        rpm
        * 2.0
        * np.pi
        / 60.0
    )

    power_w = (
        torque
        * angular_velocity
    )

    # ------------------------------------------------------------------------
    # HDF
    # ------------------------------------------------------------------------

    result[
        "diagnostic_hdf_condition"
    ] = (
        (
            temperature_gap
            < AI4I_THRESHOLDS[
                "heat_temperature_gap_failure_threshold_k"
            ]
        )
        &
        (
            rpm
            < AI4I_THRESHOLDS[
                "heat_rotational_speed_failure_threshold_rpm"
            ]
        )
    ).astype(
        "int8"
    )

    # ------------------------------------------------------------------------
    # PWF
    # ------------------------------------------------------------------------

    result[
        "diagnostic_power_w"
    ] = power_w

    result[
        "diagnostic_pwf_condition"
    ] = (
        (
            power_w
            < AI4I_THRESHOLDS[
                "power_failure_lower_w"
            ]
        )
        |
        (
            power_w
            > AI4I_THRESHOLDS[
                "power_failure_upper_w"
            ]
        )
    ).astype(
        "int8"
    )

    # ------------------------------------------------------------------------
    # TWF
    # ------------------------------------------------------------------------

    result[
        "diagnostic_twf_condition"
    ] = (
        (
            tool_wear
            >= AI4I_THRESHOLDS[
                "tool_wear_failure_lower_min"
            ]
        )
        &
        (
            tool_wear
            <= AI4I_THRESHOLDS[
                "tool_wear_failure_upper_min"
            ]
        )
    ).astype(
        "int8"
    )

    # ------------------------------------------------------------------------
    # OSF
    # ------------------------------------------------------------------------

    product_type = (
        df["Type"]
        .astype(str)
        .str.upper()
    )

    threshold = np.select(
        [
            product_type.eq("L"),
            product_type.eq("M"),
            product_type.eq("H"),
        ],
        [
            11000.0,
            12000.0,
            13000.0,
        ],
        default=np.nan,
    )

    wear_torque = (
        tool_wear
        * torque
    )

    result[
        "diagnostic_wear_torque"
    ] = wear_torque

    result[
        "diagnostic_osf_condition"
    ] = (
        wear_torque
        > threshold
    ).astype(
        "int8"
    )

    return result


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_loaded_dataframe(
    df: pd.DataFrame,
    *,
    mode: str,
) -> pd.DataFrame:
    """
    Perform safe structural cleaning.

    Operations:
        - normalize columns
        - convert known numeric fields
        - convert timestamps
        - preserve target
        - preserve failure labels
        - remove completely empty rows

    Statistical imputation is intentionally NOT performed here.
    """

    result = normalize_column_names(
        df
    )

    # ------------------------------------------------------------------------
    # Remove completely empty rows.
    # ------------------------------------------------------------------------

    all_missing = result.isna().all(
        axis=1
    )

    removed_rows = int(
        all_missing.sum()
    )

    if removed_rows > 0:

        LOGGER.warning(
            "Removing %d completely empty rows.",
            removed_rows,
        )

        result = result.loc[
            ~all_missing
        ].copy()

    # ------------------------------------------------------------------------
    # AI4I numeric conversion.
    # ------------------------------------------------------------------------

    if mode == "ai4i":

        result = convert_numeric_columns(
            result,
            AI4I_NUMERIC_COLUMNS,
            strict=True,
        )

    else:

        numeric_candidates = [
            column
            for column in result.columns
            if pd.api.types.is_numeric_dtype(
                result[column]
            )
        ]

        result = convert_numeric_columns(
            result,
            numeric_candidates,
            strict=False,
        )

    # ------------------------------------------------------------------------
    # Timestamp detection/conversion.
    # ------------------------------------------------------------------------

    timestamp_column = (
        detect_timestamp_column(
            result
        )
    )

    if timestamp_column is not None:

        result = convert_timestamp_column(
            result,
            timestamp_column,
        )

        invalid_timestamp = result[
            timestamp_column
        ].isna()

        invalid_count = int(
            invalid_timestamp.sum()
        )

        if invalid_count > 0:

            LOGGER.warning(
                "Found %d invalid timestamp rows.",
                invalid_count,
            )

    return result.reset_index(
        drop=True
    )


# ============================================================================
# TIME-SERIES SORTING
# ============================================================================

def sort_for_time_series(
    df: pd.DataFrame,
    *,
    timestamp_column: Optional[str] = None,
    machine_id_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Sort a true time-series dataset chronologically.

    For multi-machine data:
        machine_id -> timestamp
    """

    result = df.copy()

    if timestamp_column is None:

        return result

    sort_columns = []

    if (
        machine_id_column is not None
        and machine_id_column in result.columns
    ):

        sort_columns.append(
            machine_id_column
        )

    sort_columns.append(
        timestamp_column
    )

    LOGGER.info(
        "Sorting time-series data by: %s",
        sort_columns,
    )

    result = result.sort_values(
        sort_columns,
        kind="mergesort",
    )

    return result.reset_index(
        drop=True
    )


# ============================================================================
# MAIN DATASET LOADER
# ============================================================================

def load_dataset(
    *,
    path: Optional[Path] = None,
    mode: str = DATA_MODE,
    prefer_local: bool = True,
    use_ucimlrepo: bool = True,
    timeout: int = 60,
    sort_timeseries: bool = True,
) -> LoadedDataset:
    """
    Load, normalize, validate, and profile a dataset.

    Parameters
    ----------
    path:
        Optional local CSV path.

    mode:
        "auto", "ai4i", or "timeseries".

    prefer_local:
        Prefer local CSV when available.

    use_ucimlrepo:
        Use ucimlrepo for AI4I downloads.

    timeout:
        UCI direct-download timeout.

    sort_timeseries:
        Whether true time-series data should be sorted.

    Returns
    -------
    LoadedDataset
        Validated dataset and profile.
    """

    # ------------------------------------------------------------------------
    # Resolve path.
    # ------------------------------------------------------------------------

    path = (
        Path(path)
        if path is not None
        else Path(RAW_DATA_PATH)
    )

    source_path: Optional[Path] = None
    source_url: Optional[str] = None

    # ------------------------------------------------------------------------
    # Load raw data.
    # ------------------------------------------------------------------------

    if path.exists():

        raw_df = load_csv(
            path
        )

        source_path = path

    elif str(mode).lower().strip() in {
        "ai4i",
        "auto",
    }:

        raw_df, source_path, source_url = load_ai4i(
            local_path=path,
            prefer_local=prefer_local,
            use_ucimlrepo=use_ucimlrepo,
            timeout=timeout,
        )

    else:

        raise FileNotFoundError(
            f"Dataset file '{path}' was not found. "
            "For timeseries mode, provide a valid local "
            "dataset path."
        )

    # ------------------------------------------------------------------------
    # Normalize columns BEFORE mode detection.
    #
    # This is critical.
    #
    # Official AI4I:
    #
    #     UDI
    #
    # gets normalized to:
    #
    #     UID
    #
    # and only then is AI4I detection performed.
    # ------------------------------------------------------------------------

    normalized_df = normalize_column_names(
        raw_df
    )

    LOGGER.info(
        "Normalized dataset columns: %s",
        normalized_df.columns.tolist(),
    )

    # ------------------------------------------------------------------------
    # Determine mode.
    # ------------------------------------------------------------------------

    detected_mode = detect_data_mode(
        normalized_df,
        requested_mode=mode,
    )

    LOGGER.info(
        "Detected dataset mode: %s",
        detected_mode,
    )

    # ------------------------------------------------------------------------
    # General schema validation.
    # ------------------------------------------------------------------------

    validate_general_schema(
        normalized_df
    )

    # ------------------------------------------------------------------------
    # AI4I schema validation.
    # ------------------------------------------------------------------------

    if detected_mode == "ai4i":

        validate_ai4i_schema(
            normalized_df
        )

    # ------------------------------------------------------------------------
    # Structural cleaning.
    # ------------------------------------------------------------------------

    cleaned_df = clean_loaded_dataframe(
        normalized_df,
        mode=detected_mode,
    )

    # ------------------------------------------------------------------------
    # Detect timestamp and machine ID.
    # ------------------------------------------------------------------------

    timestamp_column = (
        detect_timestamp_column(
            cleaned_df
        )
    )

    machine_id_column = (
        detect_machine_id_column(
            cleaned_df
        )
    )

    # ------------------------------------------------------------------------
    # Timestamp conversion and sorting.
    # ------------------------------------------------------------------------

    if timestamp_column is not None:

        cleaned_df = convert_timestamp_column(
            cleaned_df,
            timestamp_column,
        )

        if sort_timeseries:

            cleaned_df = sort_for_time_series(
                cleaned_df,
                timestamp_column=timestamp_column,
                machine_id_column=machine_id_column,
            )

    # ------------------------------------------------------------------------
    # Target validation.
    # ------------------------------------------------------------------------

    if TARGET_COLUMN in cleaned_df.columns:

        validate_binary_target(
            cleaned_df,
            target_column=TARGET_COLUMN,
        )

    # ------------------------------------------------------------------------
    # AI4I range diagnostics.
    # ------------------------------------------------------------------------

    if detected_mode == "ai4i":

        range_report = (
            validate_ai4i_ranges(
                cleaned_df
            )
        )

        invalid_rows = (
            range_report[
                "invalid_count"
            ].sum()
            if not range_report.empty
            else 0
        )

        if invalid_rows > 0:

            LOGGER.warning(
                "AI4I physical-range diagnostics identified "
                "%d invalid-value occurrences.",
                int(invalid_rows),
            )

    # ------------------------------------------------------------------------
    # Duplicate diagnostics.
    # ------------------------------------------------------------------------

    duplicate_summary = (
        calculate_duplicate_summary(
            cleaned_df
        )
    )

    if duplicate_summary[
        "duplicate_rows"
    ] > 0:

        LOGGER.warning(
            "Dataset contains %d duplicated rows.",
            duplicate_summary[
                "duplicate_rows"
            ],
        )

    # ------------------------------------------------------------------------
    # Missing-value diagnostics.
    # ------------------------------------------------------------------------

    missing_summary = (
        calculate_missing_summary(
            cleaned_df
        )
    )

    missing_total = int(
        missing_summary[
            "missing_count"
        ].sum()
    )

    if missing_total > 0:

        LOGGER.warning(
            "Dataset contains %d missing values.",
            missing_total,
        )

    # ------------------------------------------------------------------------
    # Leakage diagnostics.
    # ------------------------------------------------------------------------

    leakage_report = (
        validate_leakage_columns(
            cleaned_df
        )
    )

    LOGGER.info(
        "Columns excluded from predictive feature "
        "generation: %s",
        leakage_report[
            "excluded_columns_present"
        ],
    )

    # ------------------------------------------------------------------------
    # Dataset profile.
    # ------------------------------------------------------------------------

    dataset_name = (
        DATASET_NAME
        if detected_mode == "ai4i"
        else (
            "Industrial Predictive "
            "Maintenance Dataset"
        )
    )

    source = (
        DATASET_SOURCE
        if detected_mode == "ai4i"
        else (
            "User-provided industrial dataset"
        )
    )

    profile = profile_dataset(
        cleaned_df,
        dataset_name=dataset_name,
        source=source,
        mode=detected_mode,
    )

    # ------------------------------------------------------------------------
    # Return loaded dataset.
    # ------------------------------------------------------------------------

    return LoadedDataset(
        data=cleaned_df,
        profile=profile,
        source_path=source_path,
        source_url=source_url,
        is_ai4i=(
            detected_mode == "ai4i"
        ),
        is_true_timeseries=(
            detected_mode == "timeseries"
        ),
    )


# ============================================================================
# DATASET EXPORT
# ============================================================================

def save_dataset(
    df: pd.DataFrame,
    path: Path,
) -> Path:
    """
    Save a DataFrame to CSV.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        path,
        index=False,
    )

    LOGGER.info(
        "Saved dataset to: %s",
        path,
    )

    return path


# ============================================================================
# PROFILE EXPORT
# ============================================================================

def save_profile(
    profile: DatasetProfile,
    path: Path,
) -> Path:
    """
    Save a dataset profile as JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            profile.to_dict(),
            file,
            indent=4,
            ensure_ascii=False,
        )

    LOGGER.info(
        "Saved dataset profile to: %s",
        path,
    )

    return path


# ============================================================================
# HUMAN-READABLE SUMMARY
# ============================================================================

def print_dataset_summary(
    loaded: LoadedDataset,
) -> None:
    """
    Print a professional console summary.
    """

    profile = loaded.profile

    print()
    print("=" * 80)
    print(
        "PREDICTIVE MAINTENANCE DATASET SUMMARY"
    )
    print("=" * 80)

    print(
        f"Dataset           : "
        f"{profile.dataset_name}"
    )

    print(
        f"Mode              : "
        f"{profile.mode}"
    )

    print(
        f"Rows              : "
        f"{profile.rows:,}"
    )

    print(
        f"Columns           : "
        f"{profile.columns:,}"
    )

    print(
        f"Missing values    : "
        f"{profile.missing_values:,}"
    )

    print(
        f"Duplicate rows    : "
        f"{profile.duplicate_rows:,}"
    )

    print(
        f"Target            : "
        f"{profile.target_column or 'Not available'}"
    )

    if profile.target_distribution:

        print(
            "Target distribution:"
        )

        for key, value in (
            profile.target_distribution.items()
        ):

            print(
                f"    {key}: {value:,}"
            )

    print(
        f"Timestamp column  : "
        f"{profile.timestamp_column or 'Not detected'}"
    )

    print(
        f"Machine ID        : "
        f"{profile.machine_id_column or 'Not detected'}"
    )

    print(
        f"AI4I dataset      : "
        f"{'Yes' if loaded.is_ai4i else 'No'}"
    )

    print(
        f"True time series  : "
        f"{'Yes' if loaded.is_true_timeseries else 'No'}"
    )

    print("=" * 80)


# ============================================================================
# MAIN VALIDATION
# ============================================================================

def main() -> None:
    """
    Run a standalone data-loader validation.

    This validates the data-loading layer before the complete Task 2
    feature-engineering pipeline is executed.
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
        "\nStarting Task 2 data-loader validation..."
    )

    # ------------------------------------------------------------------------
    # Load dataset.
    # ------------------------------------------------------------------------

    loaded = load_dataset(
        mode=DATA_MODE
    )

    # ------------------------------------------------------------------------
    # Print summary.
    # ------------------------------------------------------------------------

    print_dataset_summary(
        loaded
    )

    # ------------------------------------------------------------------------
    # Print actual normalized columns.
    # ------------------------------------------------------------------------

    print(
        "\nNormalized columns:"
    )

    for index, column in enumerate(
        loaded.data.columns,
        start=1,
    ):

        print(
            f"    {index:2d}. {column}"
        )

    # ------------------------------------------------------------------------
    # AI4I diagnostics.
    # ------------------------------------------------------------------------

    if loaded.is_ai4i:

        diagnostics = (
            calculate_ai4i_failure_logic_diagnostics(
                loaded.data
            )
        )

        if not diagnostics.empty:

            print(
                "\nAI4I physical-condition diagnostics:"
            )

            print(
                diagnostics.describe(
                    include="all"
                )
                .transpose()
                .to_string()
            )

    # ------------------------------------------------------------------------
    # Missing-value report.
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------------
    # Missing-value report
    # ---------------------------------------------------------------------

    missing_summary = (
        calculate_missing_summary(
            loaded.data
        )
    )

    print(
        "\nTop missing-value diagnostics:"
    )

    print(
        missing_summary.head(10).to_string(
            index=False
        )
    )

    print(
        "\nData loader completed successfully."
    )


if __name__ == "__main__":
    main()