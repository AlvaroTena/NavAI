import itertools
from typing import Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd

from mlnav.features.normalization import *
from mlnav.utils.logger import Logger
from mlnav.utils.plot import generate_subplots


def plot_features(
    df: pd.DataFrame,
    columns: List[str],
    constellation_plot: bool = True,
    box_plot: bool = True,
    density_plot: bool = True,
) -> Tuple[
    matplotlib.figure.Figure, matplotlib.figure.Figure, matplotlib.figure.Figure
]:
    """
    This function generates three figures: a bar plot of satellite constellation counts, a box plot of the given columns,
    and a density plot of the given columns. The function returns a tuple of the three figures.

    Parameters:
    -----------
    df : pd.DataFrame
        The input pandas DataFrame.
    columns : List[str]
        The list of column names to plot.

    Returns:
    --------
    Tuple[matplotlib.figure.Figure, matplotlib.figure.Figure, matplotlib.figure.Figure]
        A tuple containing three matplotlib figures: a bar plot, a box plot, and a density plot.
    """
    plot_counts = plot_cons(df) if constellation_plot else None
    plot_box = plot_boxplot(df, columns) if box_plot else None
    plot_kde = plot_density(df, columns) if density_plot else None

    return plot_counts, plot_box, plot_kde


def plot_boxplot(data: pd.DataFrame, columns):
    """
    This function generates a box plot of the given columns in a pandas DataFrame. The function returns a matplotlib figure object.

    Parameters:
    -----------
    data : pd.DataFrame
        The input pandas DataFrame.
    columns : List[str]
        The list of column names to plot.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the box plot of the given columns.
    """
    fig, axes = generate_subplots(k=len(columns), max_col=3, row_wise=True)

    for i, ax in enumerate(axes.flatten()):
        # Box and Whiskers
        ax.boxplot(data[columns[i]])

        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.tick_params(axis="both", which="minor", labelsize=6)

        ax.set_title(columns[i])

    # Get current figure
    return fig


def plot_density(data: pd.DataFrame, columns):
    """
    This function generates a density plot of the given columns in a pandas DataFrame. The function returns a matplotlib figure object.

    Parameters:
    -----------
    data : pd.DataFrame
        The input pandas DataFrame.
    columns : List[str]
        The list of column names to plot.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the density plot of the given columns.
    """
    fig, axes = generate_subplots(k=len(columns), max_col=3, row_wise=True)

    for i, ax in enumerate(axes.flatten()):
        # Histograms
        density, bins, _ = ax.hist(data[columns[i]], bins=20, alpha=0, density=True)

        ax.plot(bins[:-1], density)
        ax.tick_params(axis="both", which="major", labelsize=6)
        ax.tick_params(axis="both", which="minor", labelsize=6)

        ax.set_title(columns[i])

    # Get current figure
    return fig


def plot_cons(data: pd.DataFrame) -> matplotlib.figure.Figure:
    """
    This function generates a bar plot of the counts of satellite constellations in a pandas DataFrame. The function returns a
    matplotlib figure object.

    Parameters:
    -----------
    data : pd.DataFrame
        The input pandas DataFrame.

    Returns:
    --------
    matplotlib.figure.Figure
        A matplotlib figure object containing the bar plot of satellite constellation counts.
    """
    plt.rcParams.update({"font.size": 24})
    fig, ax = plt.subplots(figsize=(30, 10))

    # Concatenate 'cons' and 'sat_id' with leading zeros for sat_id
    data["cons_sat_id"] = data["cons"].astype(str) + data["sat_id"].astype(
        str
    ).str.zfill(2)

    # Define all possible combinations but only keep those present in the data
    all_combinations = [
        f"{cons}{sat_id:02d}"
        for cons, sat_id in list(itertools.product(["G", "E"], range(1, 41)))
        + list(itertools.product(["B"], range(1, 64)))
    ]
    valid_combinations = [
        combo for combo in all_combinations if combo in data["cons_sat_id"].values
    ]

    # Create a Series with counts of each valid combination
    plot_data = data["cons_sat_id"].value_counts().reindex(valid_combinations).fillna(0)

    # Prepare labels and colors for the plot
    bar_labels = []
    bar_colors = []
    for cons_sat_id in plot_data.index:
        cons = cons_sat_id[0]
        if cons == "E":
            bar_colors.append("tab:orange")
            label = "GALILEO"
        elif cons == "G":
            bar_colors.append("tab:blue")
            label = "GPS"
        elif cons == "B":
            bar_colors.append("tab:green")
            label = "BDS"

        if label not in bar_labels:
            bar_labels.append(label)
        else:
            bar_labels.append("_" + label)

    ax.bar(plot_data.index, plot_data, color=bar_colors)
    ax.tick_params(axis="x", labelrotation=45, labelsize=16)
    ax.tick_params(axis="y", labelsize=16)

    # Create a unique list for the legend
    legend_labels = {
        label.strip("_"): color for label, color in zip(bar_labels, bar_colors)
    }
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=legend_labels[label])
            for label in legend_labels
        ],
        title="Constellation",
        labels=list(legend_labels.keys()),
        title_fontsize=26,
        fontsize=20,
    )

    return fig
