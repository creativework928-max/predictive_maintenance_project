"""
src/visualization.py

Professional Visualization Module
==================================

Task 2: Feature Engineering — Predictive Maintenance Project

This module creates professional visualizations for:

1. Feature distributions
2. Correlation heatmaps
3. Rolling statistics
4. Sensor relationships
5. Missing-value profiles
6. Feature-quality summaries
7. Target-correlations
8. Anomaly/z-score distributions
9. Feature importance
10. Model-performance-oriented feature diagnostics

Design principles
-----------------
- Visualization logic is kept separate from feature engineering.
- Functions return matplotlib Figure/Axes objects where appropriate.
- Figures can be displayed interactively or saved to disk.
- No data transformation required for modeling is performed here.
- Visualizations should never alter the source DataFrame.
- Large datasets are sampled only for plotting when appropriate.
- Numeric and categorical features are handled separately.
- The module works with both AI4I reference data and genuine
  timestamped/machine-based predictive-maintenance datasets.

Expected canonical AI4I feature names
-------------------------------------
The project data loader is expected to normalize the original UCI
column names into names such as:

    uid
    product_id
    type
    air_temperature_k
    process_temperature_k
    rotational_speed_rpm
    torque_nm
    tool_wear_min
    machine_failure
    twf
    hdf
    pwf
    osf
    rnf

The plotting functions also accept arbitrary column names supplied
explicitly by the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Global visualization configuration
# ---------------------------------------------------------------------

DEFAULT_FIGURE_DPI = 160
DEFAULT_FIGURE_FORMAT = "png"

# Professional plotting defaults.
# We intentionally do not force a global matplotlib style so that
# projects embedding this module retain control of their environment.
DEFAULT_FIGURE_SIZE = (12, 7)

# A restrained professional palette.
PALETTE = {
    "primary": "#1F4E79",
    "secondary": "#2E75B6",
    "accent": "#70AD47",
    "warning": "#ED7D31",
    "danger": "#C00000",
    "neutral": "#7F8C8D",
    "dark": "#1F2937",
    "light": "#F3F6F9",
    "grid": "#D9E2F3",
}


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
) -> None:
    """
    Validate that the input is a non-empty pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{name} must be a pandas DataFrame, "
            f"received {type(df).__name__}."
        )

    if df.empty:
        raise ValueError(
            f"{name} is empty."
        )


def validate_columns(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """
    Validate that all requested columns exist.
    """

    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "The following required columns are missing: "
            + ", ".join(missing)
        )


def get_numeric_columns(
    df: pd.DataFrame,
    exclude: Optional[Iterable[str]] = None,
) -> list[str]:
    """
    Return numeric columns while excluding optional columns.
    """

    exclude_set = set(exclude or [])

    return [
        column
        for column in df.select_dtypes(
            include=[np.number]
        ).columns
        if column not in exclude_set
    ]


# ---------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------

def create_figure(
    figsize: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    title: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Create a professionally formatted matplotlib figure.
    """

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if title:
        ax.set_title(
            title,
            fontsize=16,
            fontweight="bold",
            pad=16,
            color=PALETTE["dark"],
        )

    ax.grid(
        axis="y",
        alpha=0.25,
        color=PALETTE["grid"],
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    return fig, ax


def finalize_figure(
    fig: plt.Figure,
    output_path: Optional[str | Path] = None,
    tight_layout: bool = True,
    transparent: bool = False,
) -> plt.Figure:
    """
    Finalize and optionally save a figure.
    """

    if tight_layout:
        fig.tight_layout()

    if output_path is not None:

        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            dpi=DEFAULT_FIGURE_DPI,
            bbox_inches="tight",
            facecolor="white",
            transparent=transparent,
        )

    return fig


def sanitize_filename(value: str) -> str:
    """
    Convert arbitrary text into a safe filename.
    """

    safe = str(value).strip()

    for character in [
        " ",
        "/",
        "\\",
        ":",
        ";",
        ",",
        "(",
        ")",
        "[",
        "]",
        "{",
        "}",
    ]:
        safe = safe.replace(
            character,
            "_",
        )

    while "__" in safe:
        safe = safe.replace(
            "__",
            "_",
        )

    return safe.strip("_").lower()


# ---------------------------------------------------------------------
# Distribution visualization
# ---------------------------------------------------------------------

def plot_feature_distribution(
    df: pd.DataFrame,
    feature: str,
    *,
    target_column: Optional[str] = None,
    bins: int = 40,
    figsize: tuple[float, float] = (11, 6),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the distribution of a numeric feature.

    If target_column is supplied and binary, distributions are shown
    separately for the target classes.

    Parameters
    ----------
    df:
        Input DataFrame.

    feature:
        Numeric feature to visualize.

    target_column:
        Optional binary target column.

    bins:
        Number of histogram bins.
    """

    validate_dataframe(df)
    validate_columns(df, [feature])

    if bins < 5:
        raise ValueError(
            "bins must be at least 5."
        )

    series = pd.to_numeric(
        df[feature],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if target_column is not None:

        validate_columns(
            df,
            [target_column],
        )

        target_values = df[target_column].dropna().unique()

        if len(target_values) <= 10:

            for index, target_value in enumerate(
                sorted(target_values)
            ):

                mask = (
                    df[target_column]
                    == target_value
                )

                values = series[mask].dropna()

                if values.empty:
                    continue

                ax.hist(
                    values,
                    bins=bins,
                    alpha=0.50,
                    density=True,
                    label=f"{target_column} = {target_value}",
                    edgecolor="white",
                    linewidth=0.5,
                )

            ax.legend(
                frameon=False,
            )

        else:

            values = series.dropna()

            ax.hist(
                values,
                bins=bins,
                alpha=0.75,
                edgecolor="white",
                linewidth=0.5,
                color=PALETTE["primary"],
            )

    else:

        values = series.dropna()

        ax.hist(
            values,
            bins=bins,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.5,
            color=PALETTE["primary"],
        )

    ax.set_title(
        f"Distribution of {feature}",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlabel(
        feature,
        fontsize=11,
    )

    ax.set_ylabel(
        "Density / Frequency",
        fontsize=11,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


def plot_feature_distributions_grid(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    *,
    max_features: int = 12,
    bins: int = 30,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (5, 3.5),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Create a multi-panel distribution overview for numeric features.
    """

    validate_dataframe(df)

    if features is None:

        features = get_numeric_columns(df)

    features = list(features)

    validate_columns(
        df,
        features,
    )

    features = features[:max_features]

    if not features:
        raise ValueError(
            "No features were supplied for distribution plotting."
        )

    ncols = max(1, ncols)

    nrows = math.ceil(
        len(features) / ncols
    )

    figsize = (
        figsize_per_panel[0] * ncols,
        figsize_per_panel[1] * nrows,
    )

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for index, feature in enumerate(features):

        ax = axes_flat[index]

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        ax.hist(
            values,
            bins=bins,
            alpha=0.80,
            edgecolor="white",
            linewidth=0.5,
            color=PALETTE["secondary"],
        )

        ax.set_title(
            feature,
            fontsize=11,
            fontweight="bold",
        )

        ax.set_xlabel("")
        ax.set_ylabel("Frequency")

        ax.grid(
            axis="y",
            alpha=0.20,
            color=PALETTE["grid"],
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for index in range(
        len(features),
        len(axes_flat),
    ):
        axes_flat[index].set_visible(False)

    fig.suptitle(
        "Engineered Feature Distributions",
        fontsize=17,
        fontweight="bold",
        color=PALETTE["dark"],
        y=1.01,
    )

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Boxplot visualization
# ---------------------------------------------------------------------

def plot_feature_boxplots(
    df: pd.DataFrame,
    features: Optional[Sequence[str]] = None,
    *,
    max_features: int = 15,
    figsize: tuple[float, float] = (13, 7),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot boxplots for selected numeric features.

    Features are standardized only for visualization so that features
    measured on very different physical scales can be compared.
    The original DataFrame is never modified.
    """

    validate_dataframe(df)

    if features is None:
        features = get_numeric_columns(df)

    features = list(features)[:max_features]

    validate_columns(
        df,
        features,
    )

    numeric_df = df[features].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Visualization-only standardization.
    means = numeric_df.mean()
    stds = numeric_df.std().replace(0, np.nan)

    standardized = (
        numeric_df - means
    ) / stds

    plot_data = [
        standardized[column].dropna().values
        for column in features
    ]

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    # Matplotlib 3.9+ uses `tick_labels`; older versions use `labels`.
    import inspect

    boxplot_label_keyword = (
        "tick_labels"
        if "tick_labels" in inspect.signature(ax.boxplot).parameters
        else "labels"
    )

    boxplot_kwargs = dict(
        patch_artist=True,
        boxprops=dict(
            facecolor="#D9EAF7",
            edgecolor=PALETTE["primary"],
        ),
        medianprops=dict(
            color=PALETTE["danger"],
            linewidth=2,
        ),
        whiskerprops=dict(
            color=PALETTE["primary"],
        ),
        capprops=dict(
            color=PALETTE["primary"],
        ),
        flierprops=dict(
            marker="o",
            markersize=3,
            markerfacecolor=PALETTE["warning"],
            markeredgecolor="none",
            alpha=0.35,
        ),
    )

    boxplot_kwargs[boxplot_label_keyword] = features

    ax.boxplot(
        plot_data,
        **boxplot_kwargs,
    )

    ax.axhline(
        0,
        linewidth=1,
        linestyle="--",
        color=PALETTE["neutral"],
        alpha=0.6,
    )

    ax.set_title(
        "Standardized Feature Spread and Outliers",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_ylabel(
        "Standardized value",
    )

    ax.tick_params(
        axis="x",
        rotation=45,
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------

def plot_correlation_heatmap(
    df_or_correlation: pd.DataFrame,
    *,
    features: Optional[Sequence[str]] = None,
    method: str = "pearson",
    figsize: tuple[float, float] = (13, 10),
    annotate: bool = False,
    annotation_decimals: int = 2,
    title: str = "Feature Correlation Heatmap",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot a professional correlation heatmap.

    The function accepts either:
    - a raw DataFrame, or
    - an already calculated square correlation matrix.
    """

    validate_dataframe(
        df_or_correlation,
        name="Input",
    )

    input_df = df_or_correlation.copy()

    # Detect whether this is already a correlation matrix.
    is_square = (
        input_df.shape[0] == input_df.shape[1]
        and list(input_df.index.astype(str))
        == list(input_df.columns.astype(str))
    )

    if is_square:

        correlation = input_df.apply(
            pd.to_numeric,
            errors="coerce",
        )

    else:

        if features is not None:
            validate_columns(
                input_df,
                features,
            )

            working = input_df[list(features)]
        else:
            working = input_df.select_dtypes(
                include=[np.number]
            )

        if working.empty:
            raise ValueError(
                "No numeric features available for correlation."
            )

        correlation = working.corr(
            method=method,
        )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    matrix = correlation.to_numpy(
        dtype=float
    )

    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
    )

    cbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

    cbar.set_label(
        "Correlation",
        rotation=270,
        labelpad=18,
    )

    labels = correlation.columns.astype(str)

    ax.set_xticks(
        np.arange(len(labels))
    )

    ax.set_yticks(
        np.arange(len(labels))
    )

    ax.set_xticklabels(
        labels,
        rotation=75,
        ha="right",
        fontsize=8,
    )

    ax.set_yticklabels(
        labels,
        fontsize=8,
    )

    if annotate:

        for i in range(matrix.shape[0]):

            for j in range(matrix.shape[1]):

                value = matrix[i, j]

                if np.isnan(value):
                    continue

                ax.text(
                    j,
                    i,
                    f"{value:.{annotation_decimals}f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Target-correlation visualization
# ---------------------------------------------------------------------

def plot_target_correlations(
    correlations: pd.DataFrame,
    *,
    feature_column: str = "feature",
    correlation_column: str = "correlation",
    top_n: int = 20,
    figsize: tuple[float, float] = (11, 8),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the strongest feature-target correlations.
    """

    validate_dataframe(
        correlations,
        name="Correlation DataFrame",
    )

    validate_columns(
        correlations,
        [
            feature_column,
            correlation_column,
        ],
    )

    working = correlations.copy()

    working[correlation_column] = pd.to_numeric(
        working[correlation_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[correlation_column]
    )

    working["absolute_correlation"] = (
        working[correlation_column].abs()
    )

    working = working.sort_values(
        "absolute_correlation",
        ascending=True,
    ).tail(top_n)

    fig_height = max(
        6,
        0.38 * len(working) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(figsize[0], fig_height),
        dpi=DEFAULT_FIGURE_DPI,
    )

    values = working[
        correlation_column
    ].to_numpy()

    labels = working[
        feature_column
    ].astype(str).to_numpy()

    y_positions = np.arange(
        len(working)
    )

    bars = ax.barh(
        y_positions,
        values,
        alpha=0.85,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.text(
            value
            + (0.015 if value >= 0 else -0.015),
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    ax.axvline(
        0,
        color=PALETTE["dark"],
        linewidth=1,
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlim(
        -1.05,
        1.05,
    )

    ax.set_xlabel(
        "Pearson correlation with target"
    )

    ax.set_title(
        "Strongest Feature–Target Relationships",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Missing-value visualization
# ---------------------------------------------------------------------

def plot_missing_value_profile(
    df: pd.DataFrame,
    *,
    top_n: int = 30,
    figsize: tuple[float, float] = (12, 8),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot missing-value percentages by feature.
    """

    validate_dataframe(df)

    missing_percentage = (
        df.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig_height = max(
        6,
        0.35 * len(missing_percentage) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(figsize[0], fig_height),
        dpi=DEFAULT_FIGURE_DPI,
    )

    bars = ax.barh(
        missing_percentage.index.astype(str),
        missing_percentage.values,
        color=PALETTE["warning"],
        alpha=0.85,
    )

    for bar, value in zip(
        bars,
        missing_percentage.values,
    ):

        ax.text(
            value + 0.4,
            bar.get_y()
            + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel(
        "Missing values (%)"
    )

    ax.set_title(
        "Missing-Value Profile",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlim(
        0,
        max(
            100,
            float(missing_percentage.max()) * 1.15
            if len(missing_percentage)
            else 100,
        ),
    )

    ax.grid(
        axis="x",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Rolling-statistics visualization
# ---------------------------------------------------------------------

def plot_rolling_statistics(
    df: pd.DataFrame,
    feature: str,
    *,
    window: int = 20,
    timestamp_column: Optional[str] = None,
    machine_id_column: Optional[str] = None,
    include_std: bool = True,
    include_min_max: bool = True,
    include_median: bool = True,
    figsize: tuple[float, float] = (13, 7),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Visualize rolling statistics for a sensor feature.

    This function is primarily intended for genuinely ordered data.

    If timestamp_column is supplied, timestamps are used on the x-axis.

    If machine_id_column is supplied, the function plots each machine
    separately when the number of machines is reasonably small.

    Important:
    ----------
    This function assumes that the DataFrame has already been ordered
    correctly by the upstream data-processing pipeline.
    """

    validate_dataframe(df)

    required = [feature]

    if timestamp_column:
        required.append(timestamp_column)

    if machine_id_column:
        required.append(machine_id_column)

    validate_columns(
        df,
        required,
    )

    if window < 2:
        raise ValueError(
            "window must be at least 2."
        )

    working = df.copy()

    working["_plot_value"] = pd.to_numeric(
        working[feature],
        errors="coerce",
    )

    if timestamp_column:

        working["_plot_timestamp"] = pd.to_datetime(
            working[timestamp_column],
            errors="coerce",
        )

        working = working.sort_values(
            "_plot_timestamp"
        )

        x_values = working["_plot_timestamp"]

    else:

        working["_plot_index"] = np.arange(
            len(working)
        )

        x_values = working["_plot_index"]

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if machine_id_column:

        machine_values = (
            working[machine_id_column]
            .dropna()
            .unique()
        )

        # Prevent unreadable figures for datasets containing thousands
        # of machines.
        machine_values = machine_values[:8]

        for machine in machine_values:

            mask = (
                working[machine_id_column]
                == machine
            )

            group = working.loc[
                mask
            ].copy()

            group["_value"] = group[
                "_plot_value"
            ]

            rolling = (
                group["_value"]
                .rolling(
                    window=window,
                    min_periods=max(
                        2,
                        window // 2,
                    ),
                )
            )

            group["_mean"] = rolling.mean()
            group["_std"] = rolling.std()
            group["_min"] = rolling.min()
            group["_max"] = rolling.max()
            group["_median"] = rolling.median()

            if timestamp_column:
                group_x = group[
                    "_plot_timestamp"
                ]
            else:
                group_x = group.index

            ax.plot(
                group_x,
                group["_mean"],
                linewidth=2,
                label=f"Machine {machine}",
            )

        ax.legend(
            frameon=False,
            ncol=2,
        )

    else:

        values = working[
            "_plot_value"
        ]

        rolling = values.rolling(
            window=window,
            min_periods=max(
                2,
                window // 2,
            ),
        )

        rolling_mean = rolling.mean()
        rolling_std = rolling.std()
        rolling_min = rolling.min()
        rolling_max = rolling.max()
        rolling_median = rolling.median()

        ax.plot(
            x_values,
            values,
            alpha=0.25,
            linewidth=1,
            label="Raw signal",
        )

        ax.plot(
            x_values,
            rolling_mean,
            linewidth=2.5,
            label=f"Rolling mean ({window})",
        )

        if include_median:

            ax.plot(
                x_values,
                rolling_median,
                linewidth=1.8,
                linestyle="--",
                label="Rolling median",
            )

        if include_std:

            upper = (
                rolling_mean
                + rolling_std
            )

            lower = (
                rolling_mean
                - rolling_std
            )

            ax.fill_between(
                x_values,
                lower,
                upper,
                alpha=0.15,
                label="± rolling std",
            )

        if include_min_max:

            ax.plot(
                x_values,
                rolling_min,
                linewidth=1,
                linestyle=":",
                label="Rolling minimum",
            )

            ax.plot(
                x_values,
                rolling_max,
                linewidth=1,
                linestyle=":",
                label="Rolling maximum",
            )

        ax.legend(
            frameon=False,
        )

    ax.set_title(
        f"Rolling Statistics — {feature}",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlabel(
        timestamp_column
        if timestamp_column
        else "Observation order"
    )

    ax.set_ylabel(
        feature
    )

    ax.grid(
        alpha=0.20,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Sensor relationship visualization
# ---------------------------------------------------------------------

def plot_sensor_relationship(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    *,
    target_column: Optional[str] = None,
    figsize: tuple[float, float] = (10, 7),
    sample_size: int = 10000,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the relationship between two sensor-derived features.

    A sample is used only when the DataFrame is larger than sample_size.
    """

    validate_dataframe(df)

    required = [
        x_feature,
        y_feature,
    ]

    if target_column:
        required.append(
            target_column
        )

    validate_columns(
        df,
        required,
    )

    working = df[required].copy()

    if len(working) > sample_size:

        working = working.sample(
            sample_size,
            random_state=42,
        )

    working[x_feature] = pd.to_numeric(
        working[x_feature],
        errors="coerce",
    )

    working[y_feature] = pd.to_numeric(
        working[y_feature],
        errors="coerce",
    )

    working = working.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=[
            x_feature,
            y_feature,
        ]
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if (
        target_column
        and working[target_column].nunique() <= 10
    ):

        for target_value in sorted(
            working[target_column]
            .dropna()
            .unique()
        ):

            subset = working[
                working[target_column]
                == target_value
            ]

            ax.scatter(
                subset[x_feature],
                subset[y_feature],
                s=24,
                alpha=0.55,
                label=f"{target_column} = {target_value}",
            )

        ax.legend(
            frameon=False,
        )

    else:

        ax.scatter(
            working[x_feature],
            working[y_feature],
            s=24,
            alpha=0.55,
            color=PALETTE["primary"],
        )

    ax.set_xlabel(
        x_feature
    )

    ax.set_ylabel(
        y_feature
    )

    ax.set_title(
        f"{x_feature} vs {y_feature}",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.grid(
        alpha=0.20,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Temperature-gap visualization
# ---------------------------------------------------------------------

def plot_temperature_gap(
    df: pd.DataFrame,
    *,
    air_temperature_column: str = "air_temperature_k",
    process_temperature_column: str = "process_temperature_k",
    target_column: Optional[str] = None,
    figsize: tuple[float, float] = (11, 6),
    bins: int = 40,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot the temperature gap:

        process_temperature - air_temperature

    This is a physically meaningful predictive-maintenance feature for
    the AI4I dataset.
    """

    validate_dataframe(df)

    validate_columns(
        df,
        [
            air_temperature_column,
            process_temperature_column,
        ],
    )

    air = pd.to_numeric(
        df[air_temperature_column],
        errors="coerce",
    )

    process = pd.to_numeric(
        df[process_temperature_column],
        errors="coerce",
    )

    gap = (
        process - air
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if (
        target_column is not None
        and target_column in df.columns
        and df[target_column].nunique() <= 10
    ):

        temp_df = pd.DataFrame(
            {
                "gap": process - air,
                "target": df[target_column],
            }
        ).dropna(
            subset=["gap"]
        )

        for target_value in sorted(
            temp_df["target"]
            .dropna()
            .unique()
        ):

            values = temp_df.loc[
                temp_df["target"] == target_value,
                "gap",
            ]

            ax.hist(
                values,
                bins=bins,
                alpha=0.50,
                density=True,
                label=f"Target = {target_value}",
                edgecolor="white",
            )

        ax.legend(
            frameon=False,
        )

    else:

        ax.hist(
            gap,
            bins=bins,
            alpha=0.80,
            edgecolor="white",
            color=PALETTE["primary"],
        )

    ax.axvline(
        8.6,
        linestyle="--",
        linewidth=2,
        color=PALETTE["danger"],
        label="8.6 K reference",
    )

    ax.set_title(
        "Temperature Gap Distribution",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlabel(
        "Process temperature − Air temperature [K]"
    )

    ax.set_ylabel(
        "Density / Frequency"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Power visualization
# ---------------------------------------------------------------------

def plot_mechanical_power(
    df: pd.DataFrame,
    *,
    power_column: str = "mechanical_power_w",
    target_column: Optional[str] = None,
    figsize: tuple[float, float] = (11, 6),
    bins: int = 50,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot mechanical power distribution.

    For AI4I, mechanical power is commonly derived from:

        P = torque * angular_velocity

    where:

        angular_velocity = 2*pi*rpm/60
    """

    validate_dataframe(df)

    validate_columns(
        df,
        [power_column],
    )

    power = pd.to_numeric(
        df[power_column],
        errors="coerce",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if (
        target_column
        and target_column in df.columns
        and df[target_column].nunique() <= 10
    ):

        for target_value in sorted(
            df[target_column]
            .dropna()
            .unique()
        ):

            values = power[
                df[target_column]
                == target_value
            ].dropna()

            if values.empty:
                continue

            ax.hist(
                values,
                bins=bins,
                alpha=0.50,
                density=True,
                label=f"Target = {target_value}",
                edgecolor="white",
            )

        ax.legend(
            frameon=False,
        )

    else:

        ax.hist(
            power.dropna(),
            bins=bins,
            alpha=0.80,
            edgecolor="white",
            color=PALETTE["primary"],
        )

    ax.axvline(
        3500,
        linestyle="--",
        linewidth=1.8,
        color=PALETTE["warning"],
        label="3.5 kW reference",
    )

    ax.axvline(
        9000,
        linestyle="--",
        linewidth=1.8,
        color=PALETTE["danger"],
        label="9.0 kW reference",
    )

    ax.set_title(
        "Mechanical Power Distribution",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlabel(
        "Mechanical power [W]"
    )

    ax.set_ylabel(
        "Density / Frequency"
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Anomaly / z-score visualization
# ---------------------------------------------------------------------

def plot_anomaly_score_distribution(
    df: pd.DataFrame,
    feature: str,
    *,
    z_score_column: Optional[str] = None,
    threshold: float = 3.0,
    figsize: tuple[float, float] = (11, 6),
    bins: int = 50,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot a feature's z-score/anomaly distribution.

    If z_score_column is not supplied, the z-score is calculated only
    for visualization.

    The calculated z-score does not modify the source DataFrame.
    """

    validate_dataframe(df)

    if z_score_column is not None:

        validate_columns(
            df,
            [z_score_column],
        )

        z_scores = pd.to_numeric(
            df[z_score_column],
            errors="coerce",
        )

    else:

        validate_columns(
            df,
            [feature],
        )

        values = pd.to_numeric(
            df[feature],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        )

        mean = values.mean()
        std = values.std()

        if pd.isna(std) or std == 0:

            z_scores = pd.Series(
                np.zeros(
                    len(values)
                ),
                index=values.index,
            )

        else:

            z_scores = (
                values - mean
            ) / std

    z_scores = z_scores.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    ax.hist(
        z_scores,
        bins=bins,
        alpha=0.80,
        edgecolor="white",
        color=PALETTE["secondary"],
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        color=PALETTE["danger"],
        label=f"+{threshold:g}",
    )

    ax.axvline(
        -threshold,
        linestyle="--",
        linewidth=2,
        color=PALETTE["danger"],
        label=f"-{threshold:g}",
    )

    anomaly_percentage = (
        (
            z_scores.abs()
            >= threshold
        ).mean()
        * 100
    )

    ax.set_title(
        f"Anomaly Score Distribution — {feature}",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlabel(
        "Z-score"
    )

    ax.set_ylabel(
        "Frequency"
    )

    ax.text(
        0.98,
        0.95,
        f"|z| ≥ {threshold:g}: "
        f"{anomaly_percentage:.2f}% of observations",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=PALETTE["grid"],
        ),
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        alpha=0.20,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Feature importance visualization
# ---------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    *,
    feature_column: str = "feature",
    importance_column: str = "importance",
    top_n: int = 20,
    figsize: tuple[float, float] = (11, 8),
    title: str = "Top Feature Importances",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot model-derived feature importance.

    The function is model-agnostic and accepts importance values
    generated by:
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - LightGBM
    - Extra Trees
    - permutation importance
    - other compatible estimators
    """

    validate_dataframe(
        importance_df,
        name="Importance DataFrame",
    )

    validate_columns(
        importance_df,
        [
            feature_column,
            importance_column,
        ],
    )

    working = importance_df[
        [
            feature_column,
            importance_column,
        ]
    ].copy()

    working[importance_column] = pd.to_numeric(
        working[importance_column],
        errors="coerce",
    )

    working = working.dropna(
        subset=[importance_column]
    )

    working = working.sort_values(
        importance_column,
        ascending=True,
    ).tail(top_n)

    fig_height = max(
        6,
        0.40 * len(working) + 2,
    )

    fig, ax = plt.subplots(
        figsize=(figsize[0], fig_height),
        dpi=DEFAULT_FIGURE_DPI,
    )

    y_positions = np.arange(
        len(working)
    )

    bars = ax.barh(
        y_positions,
        working[importance_column],
        alpha=0.85,
        color=PALETTE["primary"],
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        working[feature_column].astype(str)
    )

    ax.set_xlabel(
        "Importance"
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    for bar, value in zip(
        bars,
        working[importance_column],
    ):

        ax.text(
            bar.get_width(),
            bar.get_y()
            + bar.get_height() / 2,
            f" {value:.4f}",
            va="center",
            fontsize=9,
        )

    ax.grid(
        axis="x",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Feature count / quality overview
# ---------------------------------------------------------------------

def plot_feature_quality_overview(
    quality_report: pd.DataFrame,
    *,
    feature_column: str = "feature",
    missing_column: str = "missing_percentage",
    figsize: tuple[float, float] = (13, 8),
    top_n: int = 20,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Visualize feature-quality indicators.

    The plot combines:
    - missing percentage
    - unique-value count
    - constant/near-constant indicators

    It is intended as a report-level overview.
    """

    validate_dataframe(
        quality_report,
        name="Quality report",
    )

    validate_columns(
        quality_report,
        [
            feature_column,
            missing_column,
        ],
    )

    working = quality_report.copy()

    working[missing_column] = pd.to_numeric(
        working[missing_column],
        errors="coerce",
    )

    working = working.sort_values(
        missing_column,
        ascending=False,
    ).head(top_n)

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    labels = (
        working[feature_column]
        .astype(str)
        .tolist()
    )

    values = (
        working[missing_column]
        .fillna(0)
        .mul(100)
        .to_numpy()
    )

    y_positions = np.arange(
        len(labels)
    )

    ax.barh(
        y_positions,
        values,
        alpha=0.85,
        color=PALETTE["warning"],
    )

    ax.set_yticks(
        y_positions
    )

    ax.set_yticklabels(
        labels
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Missing values (%)"
    )

    ax.set_title(
        "Feature Quality — Missingness Overview",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    for y, value in zip(
        y_positions,
        values,
    ):

        ax.text(
            value + 0.2,
            y,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    ax.grid(
        axis="x",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Failure-rate visualization
# ---------------------------------------------------------------------

def plot_failure_rate(
    df: pd.DataFrame,
    target_column: str = "machine_failure",
    *,
    figsize: tuple[float, float] = (9, 6),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot target/failure class distribution.
    """

    validate_dataframe(df)

    validate_columns(
        df,
        [target_column],
    )

    counts = (
        df[target_column]
        .value_counts(dropna=False)
        .sort_index()
    )

    total = counts.sum()

    percentages = (
        counts / total * 100
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    bars = ax.bar(
        counts.index.astype(str),
        percentages.values,
        alpha=0.85,
        color=PALETTE["primary"],
    )

    for bar, count, percentage in zip(
        bars,
        counts.values,
        percentages.values,
    ):

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{percentage:.2f}%\n(n={count:,})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel(
        target_column
    )

    ax.set_ylabel(
        "Percentage of observations"
    )

    ax.set_title(
        "Machine Failure Class Distribution",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.grid(
        axis="y",
        alpha=0.25,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Sensor trend visualization
# ---------------------------------------------------------------------

def plot_sensor_trend(
    df: pd.DataFrame,
    feature: str,
    *,
    timestamp_column: Optional[str] = None,
    machine_id_column: Optional[str] = None,
    target_column: Optional[str] = None,
    sample_size: int = 10000,
    rolling_window: Optional[int] = None,
    figsize: tuple[float, float] = (13, 7),
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Plot a sensor/feature trend.

    This function does not assume that UID represents time.

    If timestamp_column is absent, observation order is used.
    """

    validate_dataframe(df)

    required = [feature]

    if timestamp_column:
        required.append(timestamp_column)

    if machine_id_column:
        required.append(machine_id_column)

    if target_column:
        required.append(target_column)

    validate_columns(
        df,
        required,
    )

    working = df[required].copy()

    working["_value"] = pd.to_numeric(
        working[feature],
        errors="coerce",
    )

    working = working.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna(
        subset=["_value"]
    )

    if len(working) > sample_size:

        working = working.iloc[
            np.linspace(
                0,
                len(working) - 1,
                sample_size,
            ).astype(int)
        ].copy()

    if timestamp_column:

        working["_timestamp"] = pd.to_datetime(
            working[timestamp_column],
            errors="coerce",
        )

        working = working.dropna(
            subset=["_timestamp"]
        ).sort_values(
            "_timestamp"
        )

        x = working["_timestamp"]

    else:

        working["_index"] = np.arange(
            len(working)
        )

        x = working["_index"]

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=DEFAULT_FIGURE_DPI,
    )

    if machine_id_column:

        machine_ids = (
            working[machine_id_column]
            .dropna()
            .unique()
        )

        machine_ids = machine_ids[:8]

        for machine_id in machine_ids:

            group = working[
                working[machine_id_column]
                == machine_id
            ]

            if timestamp_column:
                group_x = group["_timestamp"]
            else:
                group_x = group.index

            ax.plot(
                group_x,
                group["_value"],
                linewidth=1.5,
                alpha=0.80,
                label=f"Machine {machine_id}",
            )

        ax.legend(
            frameon=False,
            ncol=2,
        )

    else:

        ax.plot(
            x,
            working["_value"],
            linewidth=1.4,
            alpha=0.75,
            color=PALETTE["primary"],
            label="Observed",
        )

        if rolling_window is not None:

            if rolling_window < 2:
                raise ValueError(
                    "rolling_window must be at least 2."
                )

            rolling = (
                working["_value"]
                .rolling(
                    rolling_window,
                    min_periods=1,
                )
                .mean()
            )

            ax.plot(
                x,
                rolling,
                linewidth=2.5,
                color=PALETTE["danger"],
                label=(
                    f"Rolling mean "
                    f"({rolling_window})"
                ),
            )

        ax.legend(
            frameon=False,
        )

    ax.set_title(
        f"Feature Trend — {feature}",
        fontsize=16,
        fontweight="bold",
        pad=16,
        color=PALETTE["dark"],
    )

    ax.set_xlabel(
        timestamp_column
        if timestamp_column
        else "Observation order"
    )

    ax.set_ylabel(
        feature
    )

    ax.grid(
        alpha=0.20,
        color=PALETTE["grid"],
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return finalize_figure(
        fig,
        output_path,
    )


# ---------------------------------------------------------------------
# Batch report figure generation
# ---------------------------------------------------------------------

def generate_task2_visualizations(
    df: pd.DataFrame,
    *,
    output_directory: str | Path = "reports/figures",
    target_column: Optional[str] = "machine_failure",
    distribution_features: Optional[
        Sequence[str]
    ] = None,
    correlation_features: Optional[
        Sequence[str]
    ] = None,
    rolling_feature: Optional[str] = None,
    timestamp_column: Optional[str] = None,
    machine_id_column: Optional[str] = None,
    max_distribution_features: int = 12,
) -> dict[str, Path]:
    """
    Generate the core visualization set for Task 2.

    The function is deliberately tolerant of datasets that do not
    contain every AI4I-specific engineered feature.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from visualization names to generated file paths.
    """

    validate_dataframe(df)

    output_directory = Path(
        output_directory
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated: dict[str, Path] = {}

    # --------------------------------------------------------------
    # 1. Failure-rate plot
    # --------------------------------------------------------------

    if (
        target_column is not None
        and target_column in df.columns
    ):

        path = (
            output_directory
            / "01_failure_rate.png"
        )

        plot_failure_rate(
            df,
            target_column=target_column,
            output_path=path,
        )

        generated["failure_rate"] = path

    # --------------------------------------------------------------
    # 2. Distribution overview
    # --------------------------------------------------------------

    if distribution_features is None:

        distribution_features = (
            get_numeric_columns(
                df,
                exclude=(
                    [target_column]
                    if target_column
                    else None
                ),
            )
        )

    distribution_features = [
        feature
        for feature in distribution_features
        if feature in df.columns
    ][:max_distribution_features]

    if distribution_features:

        path = (
            output_directory
            / "02_feature_distributions.png"
        )

        plot_feature_distributions_grid(
            df,
            features=distribution_features,
            output_path=path,
        )

        generated[
            "feature_distributions"
        ] = path

    # --------------------------------------------------------------
    # 3. Correlation heatmap
    # --------------------------------------------------------------

    if correlation_features is None:

        correlation_features = (
            get_numeric_columns(
                df,
                exclude=(
                    [target_column]
                    if target_column
                    else None
                ),
            )
        )

    correlation_features = [
        feature
        for feature in correlation_features
        if feature in df.columns
    ]

    if correlation_features:

        path = (
            output_directory
            / "03_correlation_heatmap.png"
        )

        plot_correlation_heatmap(
            df,
            features=correlation_features,
            output_path=path,
        )

        generated[
            "correlation_heatmap"
        ] = path

    # --------------------------------------------------------------
    # 4. Temperature gap
    # --------------------------------------------------------------

    if {
        "air_temperature_k",
        "process_temperature_k",
    }.issubset(df.columns):

        path = (
            output_directory
            / "04_temperature_gap.png"
        )

        plot_temperature_gap(
            df,
            target_column=target_column,
            output_path=path,
        )

        generated[
            "temperature_gap"
        ] = path

    # --------------------------------------------------------------
    # 5. Mechanical power
    # --------------------------------------------------------------

    if "mechanical_power_w" in df.columns:

        path = (
            output_directory
            / "05_mechanical_power.png"
        )

        plot_mechanical_power(
            df,
            target_column=target_column,
            output_path=path,
        )

        generated[
            "mechanical_power"
        ] = path

    # --------------------------------------------------------------
    # 6. Rolling statistics
    # --------------------------------------------------------------

    if rolling_feature is None:

        candidates = [
            "air_temperature_k",
            "process_temperature_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
        ]

        for candidate in candidates:

            if candidate in df.columns:

                rolling_feature = candidate
                break

    if rolling_feature is not None:

        path = (
            output_directory
            / "06_rolling_statistics.png"
        )

        plot_rolling_statistics(
            df,
            rolling_feature,
            timestamp_column=timestamp_column,
            machine_id_column=machine_id_column,
            output_path=path,
        )

        generated[
            "rolling_statistics"
        ] = path

    # --------------------------------------------------------------
    # 7. Boxplots
    # --------------------------------------------------------------

    boxplot_features = [
        feature
        for feature in get_numeric_columns(
            df,
            exclude=(
                [target_column]
                if target_column
                else None
            ),
        )
        if feature in df.columns
    ][:15]

    if boxplot_features:

        path = (
            output_directory
            / "07_feature_boxplots.png"
        )

        plot_feature_boxplots(
            df,
            features=boxplot_features,
            output_path=path,
        )

        generated[
            "feature_boxplots"
        ] = path

    # --------------------------------------------------------------
    # 8. Missing-value profile
    # --------------------------------------------------------------

    path = (
        output_directory
        / "08_missing_value_profile.png"
    )

    plot_missing_value_profile(
        df,
        output_path=path,
    )

    generated[
        "missing_value_profile"
    ] = path

    # --------------------------------------------------------------
    # 9. Sensor relationship
    # --------------------------------------------------------------

    relationship_candidates = [
        (
            "torque_nm",
            "rotational_speed_rpm",
        ),
        (
            "air_temperature_k",
            "process_temperature_k",
        ),
    ]

    for x_feature, y_feature in relationship_candidates:

        if (
            x_feature in df.columns
            and y_feature in df.columns
        ):

            filename = (
                "09_"
                + sanitize_filename(x_feature)
                + "_vs_"
                + sanitize_filename(y_feature)
                + ".png"
            )

            path = (
                output_directory
                / filename
            )

            plot_sensor_relationship(
                df,
                x_feature,
                y_feature,
                target_column=target_column,
                output_path=path,
            )

            generated[
                f"{x_feature}_vs_{y_feature}"
            ] = path

            break

    return generated


# ---------------------------------------------------------------------
# Standalone demonstration
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 72)
    print(
        "Predictive Maintenance — Visualization Module"
    )
    print("=" * 72)

    # Small deterministic demonstration dataset.
    rng = np.random.default_rng(42)

    demo_df = pd.DataFrame(
        {
            "air_temperature_k": (
                298
                + np.cumsum(
                    rng.normal(
                        0,
                        0.25,
                        200,
                    )
                )
            ),
            "process_temperature_k": (
                308
                + np.cumsum(
                    rng.normal(
                        0,
                        0.30,
                        200,
                    )
                )
            ),
            "rotational_speed_rpm": (
                1500
                + rng.normal(
                    0,
                    80,
                    200,
                )
            ),
            "torque_nm": (
                40
                + rng.normal(
                    0,
                    5,
                    200,
                )
            ),
            "tool_wear_min": np.arange(200),
            "machine_failure": (
                rng.random(200) < 0.03
            ).astype(int),
        }
    )

    demo_df["temperature_gap_k"] = (
        demo_df["process_temperature_k"]
        - demo_df["air_temperature_k"]
    )

    demo_df["mechanical_power_w"] = (
        demo_df["torque_nm"]
        * demo_df["rotational_speed_rpm"]
        * 2
        * np.pi
        / 60
    )

    output_directory = Path(
        "reports/figures/demo"
    )

    generated = generate_task2_visualizations(
        demo_df,
        output_directory=output_directory,
        target_column="machine_failure",
        rolling_feature="torque_nm",
    )

    print(
        f"\nGenerated {len(generated)} visualizations:"
    )

    for name, path in generated.items():
        print(
            f"  - {name}: {path}"
        )

    print(
        "\nVisualization module completed successfully."
    )