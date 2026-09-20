"""
src/feature_quality.py

Feature Quality Assessment Module
=================================

Task 2: Feature Engineering — Predictive Maintenance Project

This module provides reusable utilities for evaluating the quality of
engineered features before they are used for machine-learning models.

The module checks:

1. Missing values
2. Missing-value percentage
3. Number of unique values
4. Constant features
5. Near-constant features
6. Duplicate columns
7. Numeric/non-numeric data types
8. Infinite values
9. Descriptive statistics
10. Variance
11. Skewness
12. Target correlation
13. High inter-feature correlation
14. Potential identifier-like columns
15. Potential leakage indicators
16. Feature usability recommendations

Important:
-----------
Feature-quality analysis must NOT introduce target leakage.

Correlation with the target is calculated for analysis/reporting only.
The target column itself is never modified.

For time-series datasets, feature generation and train/validation/test
splitting should happen according to chronological order before any
model fitting. This module only evaluates the resulting feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class FeatureQualityConfig:
    """
    Configuration for feature-quality analysis.

    Parameters
    ----------
    missing_threshold:
        Maximum acceptable missing percentage before a feature is
        flagged as having substantial missingness.

    near_constant_threshold:
        Fraction of observations occupied by the most common value.
        For example, 0.995 means that if one value occurs in at least
        99.5% of observations, the feature is considered near-constant.

    high_correlation_threshold:
        Absolute correlation above which two features are considered
        highly correlated.

    target_correlation_threshold:
        Absolute correlation above which a feature has a strong
        linear relationship with the target.

    unique_ratio_identifier_threshold:
        Ratio of unique values to number of rows above which a column
        may behave like an identifier.

    infinite_as_missing:
        Whether positive/negative infinity should be treated as
        invalid/missing values.
    """

    missing_threshold: float = 0.30
    near_constant_threshold: float = 0.995
    high_correlation_threshold: float = 0.95
    target_correlation_threshold: float = 0.80
    unique_ratio_identifier_threshold: float = 0.98
    infinite_as_missing: bool = True


# ---------------------------------------------------------------------
# Helper validation
# ---------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> None:
    """
    Validate that the supplied object is a non-empty DataFrame.

    Raises
    ------
    TypeError
        If df is not a pandas DataFrame.

    ValueError
        If df is empty.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, "
            f"received {type(df).__name__}."
        )

    if df.empty:
        raise ValueError(f"{name} is empty.")


def get_numeric_columns(
    df: pd.DataFrame,
    exclude: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Return numeric columns while optionally excluding selected columns.
    """

    exclude_set = set(exclude or [])

    return [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in exclude_set
    ]


# ---------------------------------------------------------------------
# Basic feature statistics
# ---------------------------------------------------------------------

def calculate_feature_statistics(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    config: Optional[FeatureQualityConfig] = None,
) -> pd.DataFrame:
    """
    Calculate a comprehensive quality profile for every column.

    Returns
    -------
    pandas.DataFrame
        One row per feature.
    """

    validate_dataframe(df)

    config = config or FeatureQualityConfig()

    if target_column is not None and target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    working_df = df.copy()

    if config.infinite_as_missing:
        numeric_columns = working_df.select_dtypes(
            include=[np.number]
        ).columns

        working_df[numeric_columns] = working_df[numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    rows = []

    target = None

    if target_column is not None:
        target = working_df[target_column]

        if not pd.api.types.is_numeric_dtype(target):
            # Convert categorical/binary target where possible.
            target_numeric = pd.to_numeric(target, errors="coerce")

            if target_numeric.notna().sum() > 0:
                target = target_numeric
            else:
                target = None

    n_rows = len(working_df)

    for column in working_df.columns:

        series = working_df[column]

        dtype = str(series.dtype)

        missing_count = int(series.isna().sum())
        missing_percentage = (
            missing_count / n_rows if n_rows else np.nan
        )

        non_missing = series.dropna()

        unique_count = int(series.nunique(dropna=True))

        unique_ratio = (
            unique_count / n_rows
            if n_rows > 0
            else np.nan
        )

        is_numeric = pd.api.types.is_numeric_dtype(series)

        is_constant = unique_count <= 1

        if len(non_missing) > 0:
            value_frequency = (
                non_missing.value_counts(normalize=True).iloc[0]
            )
        else:
            value_frequency = np.nan

        is_near_constant = (
            not pd.isna(value_frequency)
            and value_frequency >= config.near_constant_threshold
        )

        infinite_count = 0

        if is_numeric:
            values = series.to_numpy(dtype=float)

            infinite_count = int(
                np.isinf(values).sum()
            )

        mean_value = np.nan
        std_value = np.nan
        min_value = np.nan
        max_value = np.nan
        median_value = np.nan
        skewness = np.nan
        variance = np.nan

        if is_numeric and non_missing.shape[0] > 0:

            numeric_series = pd.to_numeric(
                series,
                errors="coerce",
            )

            mean_value = numeric_series.mean()
            std_value = numeric_series.std()
            min_value = numeric_series.min()
            max_value = numeric_series.max()
            median_value = numeric_series.median()
            skewness = numeric_series.skew()
            variance = numeric_series.var()

        target_correlation = np.nan

        if (
            target is not None
            and column != target_column
            and is_numeric
        ):
            comparison = pd.concat(
                [
                    pd.to_numeric(series, errors="coerce"),
                    pd.to_numeric(target, errors="coerce"),
                ],
                axis=1,
            ).dropna()

            if (
                len(comparison) >= 2
                and comparison.iloc[:, 0].nunique() > 1
                and comparison.iloc[:, 1].nunique() > 1
            ):
                target_correlation = comparison.iloc[:, 0].corr(
                    comparison.iloc[:, 1]
                )

        possible_identifier = (
            unique_ratio >= config.unique_ratio_identifier_threshold
            and unique_count > 10
        )

        rows.append(
            {
                "feature": column,
                "dtype": dtype,
                "is_numeric": is_numeric,
                "row_count": n_rows,
                "non_null_count": int(series.notna().sum()),
                "missing_count": missing_count,
                "missing_percentage": missing_percentage,
                "unique_count": unique_count,
                "unique_ratio": unique_ratio,
                "top_value_frequency": value_frequency,
                "is_constant": is_constant,
                "is_near_constant": is_near_constant,
                "infinite_count": infinite_count,
                "mean": mean_value,
                "std": std_value,
                "variance": variance,
                "min": min_value,
                "median": median_value,
                "max": max_value,
                "skewness": skewness,
                "target_correlation": target_correlation,
                "absolute_target_correlation": (
                    abs(target_correlation)
                    if not pd.isna(target_correlation)
                    else np.nan
                ),
                "possible_identifier": possible_identifier,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            by="feature",
            kind="stable",
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Missing values
# ---------------------------------------------------------------------

def find_missing_features(
    quality_report: pd.DataFrame,
    threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Return features whose missing-value percentage exceeds threshold.

    Parameters
    ----------
    quality_report:
        Output from calculate_feature_statistics().

    threshold:
        Fraction from 0 to 1.
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    required_columns = {
        "feature",
        "missing_percentage",
    }

    missing = required_columns - set(quality_report.columns)

    if missing:
        raise ValueError(
            f"quality_report is missing columns: {sorted(missing)}"
        )

    return quality_report[
        quality_report["missing_percentage"] > threshold
    ].copy()


# ---------------------------------------------------------------------
# Constant and near-constant features
# ---------------------------------------------------------------------

def find_constant_features(
    quality_report: pd.DataFrame,
) -> list[str]:
    """
    Return constant features.
    """

    if "is_constant" not in quality_report.columns:
        raise ValueError(
            "quality_report must contain 'is_constant'."
        )

    return quality_report.loc[
        quality_report["is_constant"],
        "feature",
    ].tolist()


def find_near_constant_features(
    quality_report: pd.DataFrame,
) -> list[str]:
    """
    Return near-constant features.
    """

    if "is_near_constant" not in quality_report.columns:
        raise ValueError(
            "quality_report must contain 'is_near_constant'."
        )

    return quality_report.loc[
        quality_report["is_near_constant"],
        "feature",
    ].tolist()


# ---------------------------------------------------------------------
# Duplicate columns
# ---------------------------------------------------------------------

def find_duplicate_features(
    df: pd.DataFrame,
) -> list[str]:
    """
    Identify duplicate columns.

    Two features are considered duplicates when they contain identical
    values for every row.

    The first occurrence is retained; later occurrences are returned.
    """

    validate_dataframe(df)

    duplicate_features: list[str] = []
    seen: dict[tuple, str] = {}

    for column in df.columns:

        # Convert to a stable hashable representation.
        values = tuple(
            df[column].astype(object).where(
                df[column].notna(),
                "__MISSING_VALUE__",
            )
        )

        if values in seen:
            duplicate_features.append(column)
        else:
            seen[values] = column

    return duplicate_features


def find_duplicate_feature_pairs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return pairs of duplicate feature columns.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - feature_1
        - feature_2
    """

    validate_dataframe(df)

    pairs = []

    columns = list(df.columns)

    for i in range(len(columns)):

        col_a = columns[i]

        for j in range(i + 1, len(columns)):

            col_b = columns[j]

            if df[col_a].equals(df[col_b]):
                pairs.append(
                    {
                        "feature_1": col_a,
                        "feature_2": col_b,
                    }
                )

    return pd.DataFrame(
        pairs,
        columns=["feature_1", "feature_2"],
    )


# ---------------------------------------------------------------------
# Infinite values
# ---------------------------------------------------------------------

def find_infinite_values(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the number of positive/negative infinity values per feature.
    """

    validate_dataframe(df)

    rows = []

    for column in df.select_dtypes(
        include=[np.number]
    ).columns:

        values = df[column].to_numpy(dtype=float)

        positive_infinity = int(
            np.isposinf(values).sum()
        )

        negative_infinity = int(
            np.isneginf(values).sum()
        )

        total_infinity = (
            positive_infinity + negative_infinity
        )

        if total_infinity > 0:
            rows.append(
                {
                    "feature": column,
                    "positive_infinity": positive_infinity,
                    "negative_infinity": negative_infinity,
                    "total_infinity": total_infinity,
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "feature",
            "positive_infinity",
            "negative_infinity",
            "total_infinity",
        ],
    )


# ---------------------------------------------------------------------
# High-correlation feature detection
# ---------------------------------------------------------------------

def calculate_correlation_matrix(
    df: pd.DataFrame,
    exclude_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation matrix for numeric features.

    Infinite values are converted to NaN before correlation.
    """

    validate_dataframe(df)

    exclude_set = set(exclude_columns or [])

    numeric_columns = [
        column
        for column in df.select_dtypes(include=[np.number]).columns
        if column not in exclude_set
    ]

    if not numeric_columns:
        return pd.DataFrame()

    numeric_df = df[numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return numeric_df.corr(method="pearson")


def find_highly_correlated_features(
    correlation_matrix: pd.DataFrame,
    threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Identify highly correlated feature pairs.

    Only the upper triangle is considered so each pair is reported once.

    Returns
    -------
    pandas.DataFrame
        Columns:
        - feature_1
        - feature_2
        - correlation
        - absolute_correlation
    """

    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    if correlation_matrix.empty:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    matrix = correlation_matrix.copy()

    rows = []

    columns = list(matrix.columns)

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            feature_1 = columns[i]
            feature_2 = columns[j]

            correlation = matrix.loc[
                feature_1,
                feature_2,
            ]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:

                rows.append(
                    {
                        "feature_1": feature_1,
                        "feature_2": feature_2,
                        "correlation": correlation,
                        "absolute_correlation": abs(correlation),
                    }
                )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False,
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Target correlation
# ---------------------------------------------------------------------

def calculate_target_correlations(
    df: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    """
    Calculate feature-to-target Pearson correlations.

    The target must be numeric or convertible to numeric.

    This function is for diagnostics only. It does not perform feature
    selection automatically because selection decisions should be made
    using the training data only.
    """

    validate_dataframe(df)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    rows = []

    for column in df.select_dtypes(include=[np.number]).columns:

        if column == target_column:
            continue

        feature = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        comparison = pd.concat(
            [feature, target],
            axis=1,
        ).dropna()

        if (
            len(comparison) < 2
            or comparison.iloc[:, 0].nunique() <= 1
            or comparison.iloc[:, 1].nunique() <= 1
        ):
            correlation = np.nan
        else:
            correlation = comparison.iloc[:, 0].corr(
                comparison.iloc[:, 1]
            )

        rows.append(
            {
                "feature": column,
                "target": target_column,
                "correlation": correlation,
                "absolute_correlation": (
                    abs(correlation)
                    if not pd.isna(correlation)
                    else np.nan
                ),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "absolute_correlation",
            ascending=False,
            na_position="last",
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Feature distribution diagnostics
# ---------------------------------------------------------------------

def calculate_numeric_distribution_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate detailed distribution statistics for numeric features.

    Statistics include:
    - mean
    - standard deviation
    - quartiles
    - IQR
    - skewness
    - minimum
    - maximum
    - zero count
    - negative count
    """

    validate_dataframe(df)

    rows = []

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns

    for column in numeric_columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)

        rows.append(
            {
                "feature": column,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "q1": float(q1),
                "median": float(series.median()),
                "q3": float(q3),
                "max": float(series.max()),
                "iqr": float(q3 - q1),
                "skewness": float(series.skew()),
                "zero_count": int((series == 0).sum()),
                "negative_count": int((series < 0).sum()),
                "positive_count": int((series > 0).sum()),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "feature",
            kind="stable",
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Outlier diagnostics
# ---------------------------------------------------------------------

def calculate_iqr_outlier_summary(
    df: pd.DataFrame,
    multiplier: float = 1.5,
) -> pd.DataFrame:
    """
    Estimate the number and percentage of IQR-based outliers.

    Important:
    ----------
    This is a diagnostic statistic, not an automatic instruction to
    remove observations. Equipment-failure datasets can contain
    legitimate extreme sensor readings that are highly informative.
    """

    validate_dataframe(df)

    if multiplier <= 0:
        raise ValueError(
            "multiplier must be greater than zero."
        )

    rows = []

    for column in df.select_dtypes(
        include=[np.number]
    ).columns:

        series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        valid = series.dropna()

        if valid.empty:
            continue

        q1 = valid.quantile(0.25)
        q3 = valid.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(
            outlier_mask.sum()
        )

        valid_count = int(
            series.notna().sum()
        )

        outlier_percentage = (
            outlier_count / valid_count
            if valid_count
            else np.nan
        )

        rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "valid_count": valid_count,
                "outlier_percentage": outlier_percentage,
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            "outlier_percentage",
            ascending=False,
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Identifier detection
# ---------------------------------------------------------------------

def find_identifier_like_features(
    quality_report: pd.DataFrame,
    unique_ratio_threshold: float = 0.98,
) -> list[str]:
    """
    Identify columns with almost one unique value per row.

    Such features may represent:
    - UID
    - transaction ID
    - row number
    - machine serial number
    - timestamp-like identifiers

    These should not automatically be deleted. In predictive
    maintenance, machine_id can be important for grouped/time-aware
    splitting and online processing.
    """

    if not 0 <= unique_ratio_threshold <= 1:
        raise ValueError(
            "unique_ratio_threshold must be between 0 and 1."
        )

    required = {
        "feature",
        "unique_ratio",
        "unique_count",
    }

    missing = required - set(quality_report.columns)

    if missing:
        raise ValueError(
            f"quality_report is missing columns: {sorted(missing)}"
        )

    result = quality_report[
        (
            quality_report["unique_ratio"]
            >= unique_ratio_threshold
        )
        & (
            quality_report["unique_count"]
            > 10
        )
    ]

    return result["feature"].tolist()


# ---------------------------------------------------------------------
# Potential leakage diagnostics
# ---------------------------------------------------------------------

def detect_potential_leakage_features(
    df: pd.DataFrame,
    target_column: str,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Perform conservative diagnostics for possible target leakage.

    The function does NOT claim that a feature is definitely leaked.

    It flags features when:

    1. Their name strongly resembles the target name.
    2. They are exact copies of the target.
    3. They have extremely high correlation with the target.

    Human review is required before removing a feature.
    """

    validate_dataframe(df)

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    if feature_names is None:
        feature_names = [
            column
            for column in df.columns
            if column != target_column
        ]

    target = pd.to_numeric(
        df[target_column],
        errors="coerce",
    )

    target_name_normalized = (
        target_column.lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )

    rows = []

    for feature in feature_names:

        if feature not in df.columns:
            continue

        feature_series = df[feature]

        name_normalized = (
            feature.lower()
            .replace("_", "")
            .replace("-", "")
            .replace(" ", "")
        )

        name_similarity = (
            target_name_normalized in name_normalized
            or name_normalized in target_name_normalized
        )

        exact_copy = False

        try:
            exact_copy = feature_series.equals(
                df[target_column]
            )
        except Exception:
            exact_copy = False

        correlation = np.nan

        numeric_feature = pd.to_numeric(
            feature_series,
            errors="coerce",
        )

        comparison = pd.concat(
            [numeric_feature, target],
            axis=1,
        ).dropna()

        if (
            len(comparison) >= 2
            and comparison.iloc[:, 0].nunique() > 1
            and comparison.iloc[:, 1].nunique() > 1
        ):
            correlation = comparison.iloc[:, 0].corr(
                comparison.iloc[:, 1]
            )

        extremely_high_correlation = (
            not pd.isna(correlation)
            and abs(correlation) >= 0.99
        )

        possible_leakage = (
            name_similarity
            or exact_copy
            or extremely_high_correlation
        )

        reason_parts = []

        if name_similarity:
            reason_parts.append(
                "feature name resembles target"
            )

        if exact_copy:
            reason_parts.append(
                "feature exactly matches target"
            )

        if extremely_high_correlation:
            reason_parts.append(
                "near-perfect target correlation"
            )

        rows.append(
            {
                "feature": feature,
                "target": target_column,
                "target_correlation": correlation,
                "name_similarity": name_similarity,
                "exact_target_copy": exact_copy,
                "extremely_high_target_correlation": (
                    extremely_high_correlation
                ),
                "potential_leakage": possible_leakage,
                "reason": "; ".join(reason_parts),
            }
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result = result.sort_values(
            [
                "potential_leakage",
                "exact_target_copy",
                "extremely_high_target_correlation",
            ],
            ascending=False,
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------
# Feature recommendation engine
# ---------------------------------------------------------------------

def generate_feature_recommendations(
    quality_report: pd.DataFrame,
    highly_correlated_features: Optional[pd.DataFrame] = None,
    leakage_report: Optional[pd.DataFrame] = None,
    missing_threshold: float = 0.30,
) -> pd.DataFrame:
    """
    Generate human-readable recommendations for feature handling.

    Possible recommendations:

    - KEEP
    - REVIEW
    - IMPUTE
    - REMOVE_CONSTANT
    - REMOVE_DUPLICATE
    - REVIEW_IDENTIFIER
    - REVIEW_LEAKAGE
    """

    if quality_report.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "recommendation",
                "reason",
            ]
        )

    recommendations = []

    high_corr_features: set[str] = set()

    if (
        highly_correlated_features is not None
        and not highly_correlated_features.empty
    ):
        high_corr_features.update(
            highly_correlated_features["feature_1"].tolist()
        )
        high_corr_features.update(
            highly_correlated_features["feature_2"].tolist()
        )

    leakage_features: set[str] = set()

    if (
        leakage_report is not None
        and not leakage_report.empty
        and "potential_leakage" in leakage_report.columns
    ):
        leakage_features.update(
            leakage_report.loc[
                leakage_report["potential_leakage"],
                "feature",
            ].tolist()
        )

    for _, row in quality_report.iterrows():

        feature = row["feature"]

        recommendation = "KEEP"
        reasons = []

        if row.get("is_constant", False):

            recommendation = "REMOVE_CONSTANT"

            reasons.append(
                "feature has only one unique value"
            )

        elif feature in leakage_features:

            recommendation = "REVIEW_LEAKAGE"

            reasons.append(
                "potential target leakage detected"
            )

        elif row.get("missing_percentage", 0) > missing_threshold:

            recommendation = "IMPUTE"

            reasons.append(
                "substantial missing values"
            )

        elif row.get("possible_identifier", False):

            recommendation = "REVIEW_IDENTIFIER"

            reasons.append(
                "feature has identifier-like uniqueness"
            )

        elif row.get("is_near_constant", False):

            recommendation = "REVIEW"

            reasons.append(
                "feature is near-constant"
            )

        elif feature in high_corr_features:

            recommendation = "REVIEW"

            reasons.append(
                "feature is highly correlated with another feature"
            )

        elif (
            not pd.isna(
                row.get("absolute_target_correlation", np.nan)
            )
            and row["absolute_target_correlation"] >= 0.80
        ):

            recommendation = "REVIEW"

            reasons.append(
                "feature has strong target correlation"
            )

        else:

            recommendation = "KEEP"

            reasons.append(
                "no major quality issue detected"
            )

        recommendations.append(
            {
                "feature": feature,
                "recommendation": recommendation,
                "reason": "; ".join(reasons),
            }
        )

    return pd.DataFrame(recommendations)


# ---------------------------------------------------------------------
# Complete quality audit
# ---------------------------------------------------------------------

def run_feature_quality_audit(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
    config: Optional[FeatureQualityConfig] = None,
) -> dict[str, pd.DataFrame]:
    """
    Run the complete feature-quality audit.

    Returns
    -------
    dict[str, pandas.DataFrame]

    Keys include:

    feature_statistics
    missing_features
    duplicate_feature_pairs
    infinite_values
    correlation_matrix
    highly_correlated_features
    target_correlations
    distribution_summary
    outlier_summary
    leakage_report
    recommendations
    """

    validate_dataframe(df)

    config = config or FeatureQualityConfig()

    feature_statistics = calculate_feature_statistics(
        df=df,
        target_column=target_column,
        config=config,
    )

    missing_features = find_missing_features(
        feature_statistics,
        threshold=config.missing_threshold,
    )

    duplicate_feature_pairs = find_duplicate_feature_pairs(df)

    infinite_values = find_infinite_values(df)

    correlation_matrix = calculate_correlation_matrix(
        df,
        exclude_columns=(
            [target_column]
            if target_column is not None
            else None
        ),
    )

    highly_correlated_features = (
        find_highly_correlated_features(
            correlation_matrix,
            threshold=config.high_correlation_threshold,
        )
    )

    if target_column is not None:

        target_correlations = calculate_target_correlations(
            df,
            target_column,
        )

        leakage_report = detect_potential_leakage_features(
            df,
            target_column,
        )

    else:

        target_correlations = pd.DataFrame(
            columns=[
                "feature",
                "target",
                "correlation",
                "absolute_correlation",
            ]
        )

        leakage_report = pd.DataFrame(
            columns=[
                "feature",
                "target",
                "target_correlation",
                "name_similarity",
                "exact_target_copy",
                "extremely_high_target_correlation",
                "potential_leakage",
                "reason",
            ]
        )

    distribution_summary = (
        calculate_numeric_distribution_summary(df)
    )

    outlier_summary = calculate_iqr_outlier_summary(df)

    recommendations = generate_feature_recommendations(
        quality_report=feature_statistics,
        highly_correlated_features=highly_correlated_features,
        leakage_report=leakage_report,
        missing_threshold=config.missing_threshold,
    )

    return {
        "feature_statistics": feature_statistics,
        "missing_features": missing_features,
        "duplicate_feature_pairs": duplicate_feature_pairs,
        "infinite_values": infinite_values,
        "correlation_matrix": correlation_matrix,
        "highly_correlated_features": highly_correlated_features,
        "target_correlations": target_correlations,
        "distribution_summary": distribution_summary,
        "outlier_summary": outlier_summary,
        "leakage_report": leakage_report,
        "recommendations": recommendations,
    }


# ---------------------------------------------------------------------
# Save audit results
# ---------------------------------------------------------------------

def save_quality_audit(
    audit_results: dict[str, pd.DataFrame],
    output_directory: str | Path,
    prefix: str = "feature_quality",
) -> dict[str, Path]:
    """
    Save all quality-audit DataFrames as CSV files.

    Parameters
    ----------
    audit_results:
        Dictionary returned by run_feature_quality_audit().

    output_directory:
        Destination directory.

    prefix:
        Filename prefix.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping between report names and generated files.
    """

    output_directory = Path(output_directory)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_files: dict[str, Path] = {}

    for name, data in audit_results.items():

        if not isinstance(data, pd.DataFrame):
            continue

        output_path = (
            output_directory
            / f"{prefix}_{name}.csv"
        )

        data.to_csv(
            output_path,
            index=True,
        )

        saved_files[name] = output_path

    return saved_files


# ---------------------------------------------------------------------
# Human-readable audit summary
# ---------------------------------------------------------------------

def create_quality_summary(
    audit_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Create a compact summary suitable for reports and dashboards.
    """

    feature_statistics = audit_results.get(
        "feature_statistics",
        pd.DataFrame(),
    )

    missing_features = audit_results.get(
        "missing_features",
        pd.DataFrame(),
    )

    duplicate_pairs = audit_results.get(
        "duplicate_feature_pairs",
        pd.DataFrame(),
    )

    infinite_values = audit_results.get(
        "infinite_values",
        pd.DataFrame(),
    )

    highly_correlated = audit_results.get(
        "highly_correlated_features",
        pd.DataFrame(),
    )

    leakage_report = audit_results.get(
        "leakage_report",
        pd.DataFrame(),
    )

    if feature_statistics.empty:
        total_features = 0
        numeric_features = 0
        constant_features = 0
        near_constant_features = 0
    else:
        total_features = len(feature_statistics)

        numeric_features = int(
            feature_statistics["is_numeric"].sum()
        )

        constant_features = int(
            feature_statistics["is_constant"].sum()
        )

        near_constant_features = int(
            feature_statistics["is_near_constant"].sum()
        )

    total_missing_features = len(
        missing_features
    )

    total_duplicate_pairs = len(
        duplicate_pairs
    )

    total_infinite_features = len(
        infinite_values
    )

    total_high_correlations = len(
        highly_correlated
    )

    total_potential_leakage = 0

    if (
        not leakage_report.empty
        and "potential_leakage" in leakage_report.columns
    ):
        total_potential_leakage = int(
            leakage_report["potential_leakage"].sum()
        )

    summary = pd.DataFrame(
        {
            "metric": [
                "Total features",
                "Numeric features",
                "Features with substantial missingness",
                "Constant features",
                "Near-constant features",
                "Duplicate feature pairs",
                "Features containing infinity",
                "Highly correlated feature pairs",
                "Potential leakage features",
            ],
            "value": [
                total_features,
                numeric_features,
                total_missing_features,
                constant_features,
                near_constant_features,
                total_duplicate_pairs,
                total_infinite_features,
                total_high_correlations,
                total_potential_leakage,
            ],
        }
    )

    return summary


# ---------------------------------------------------------------------
# Module test / standalone execution
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 72)
    print("Predictive Maintenance — Feature Quality Module")
    print("=" * 72)

    # Small synthetic demonstration dataset.
    demo_df = pd.DataFrame(
        {
            "temperature": [
                300.0,
                305.0,
                310.0,
                315.0,
                320.0,
            ],
            "rotational_speed": [
                1400,
                1450,
                1500,
                1550,
                1600,
            ],
            "constant_feature": [
                1,
                1,
                1,
                1,
                1,
            ],
            "target": [
                0,
                0,
                1,
                0,
                1,
            ],
        }
    )

    audit = run_feature_quality_audit(
        demo_df,
        target_column="target",
    )

    print("\nQuality summary:")
    print(
        create_quality_summary(audit).to_string(
            index=False
        )
    )

    print("\nFeature statistics:")
    print(
        audit["feature_statistics"].to_string(
            index=False
        )
    )

    print("\nRecommendations:")
    print(
        audit["recommendations"].to_string(
            index=False
        )
    )

    print("\nFeature quality module completed successfully.")
