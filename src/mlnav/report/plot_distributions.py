from typing import List

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import StratifiedShuffleSplit

from mlnav.utils.logger import Logger
from mlnav.utils.plot import generate_subplots


def plot_pairplots(
    data: pd.DataFrame,
    labels: np.ndarray,
    sample_size: int,
    drop_noise: bool = True,
    plot_cluster_pairplots=True,
    plot_pairplots2d=True,
    plot_pairplots3d=True,
):
    plot_columns = ["elevation", "residual", "snr", "code_rate_cons", "delta_cmc"]
    if "predictions" in data:
        plot_columns += ["predictions"]
    data = data[plot_columns]

    cluster_distribution = plot_cluster_distribution(labels)

    if drop_noise:
        noise_index = list(
            set(
                list(data[data.residual < -1000].index)
                + list(data[data.residual > 1000].index)
                + list(data[data.code_rate_cons > 5000].index)
                + list(data[data.delta_cmc > 100].index)
                + list(data[data.delta_cmc < -100].index)
            )
        )
        Logger.getLogger().debug(
            f"{len(noise_index)}/{len(data)} rows will be dropped as noise."
        )
        data.drop(noise_index, inplace=True)

    if sample_size != -1:
        plot_indices, _ = next(
            StratifiedShuffleSplit(
                n_splits=1,
                train_size=sample_size if len(data) > sample_size else 0.8,
                random_state=42,
            ).split(data, data["predictions"])
        )
        data = data.iloc[plot_indices]

    variables = [data[column] for column in data.columns if column != "predictions"]
    labels = data["predictions"] if "predictions" in data else labels

    # Plot pairplots by cluster
    cluster_pairplots = []
    if plot_cluster_pairplots:
        for cluster in range(int(max(labels)) + 1):
            Logger.getLogger().debug(f"Plotting Cluster {cluster}")
            cluster_pairplots.append(plot_pairplots_2d(variables, labels, cluster))

    # Plot pairplots in 2D
    pairplots2d = plot_pairplots_2d(variables, labels, -1) if plot_pairplots2d else None

    # Plot pairplots in 3D
    pairplots3d = plot_pairplots_3d(variables, labels) if plot_pairplots3d else None

    return cluster_distribution, cluster_pairplots, pairplots2d, pairplots3d


def plot_cluster_distribution(labels):
    # Calcula la distribución de los clusters
    unique, counts = np.unique(labels, return_counts=True)

    cmap = ListedColormap(plt.cm.tab10.colors)
    colors = cmap.colors[: len(unique)]

    # Crea el gráfico de tarta
    fig, ax = plt.subplots()
    ax.pie(
        counts,
        labels=[f"Cluster {int(label)}" for label in unique],
        startangle=90,
        autopct="%1.1f%%",
        colors=colors,
        textprops={"fontsize": 14},
    )

    # Igualar aspecto para asegurar que el gráfico de tarta sea circular
    ax.axis("equal")

    # Título del gráfico
    plt.title("Cluster Distribution", fontsize=16)

    plt.tight_layout()

    return fig


def plot_pairplots_2d(variables: List[pd.Series], labels: pd.Series, cluster: int):
    Logger.getLogger().debug("Plotting 2D distributions...")

    # Set the number of clusters to print
    if cluster != -1:
        n_clusters = 1
    else:
        n_clusters = max(labels) + 1

    colors = ListedColormap(plt.cm.tab10.colors)
    # Get the number of variables
    n_variables = len(variables)

    # Create a grid of subplots with dimensions depending on the number of variables
    fig, axes = plt.subplots(
        nrows=n_variables,
        ncols=n_variables,
        figsize=(n_variables * 2.5, n_variables * 2.5),
    )

    # Adjust the alpha for dense scatter plots
    scatter_alpha = 0.8 if n_clusters == 1 else 0.5

    legend_elements = []

    # Fill in the grid with histograms on the diagonal and scatter plots off the diagonal
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax = axes[i, j]
            # Histograms
            if i == j:
                for label, color in zip(range(int(n_clusters)), colors.colors):
                    if cluster != -1:
                        color = colors.colors[cluster]
                        idx = labels == cluster
                    else:
                        idx = labels == label

                    sns.kdeplot(
                        variables[i][idx], ax=ax, fill=True, color=color, alpha=0.5
                    )

                    if j == 0:
                        legend_elements.append(
                            mpatches.Patch(
                                color=color,
                                label=f"Cluster {label if cluster == -1 else cluster}",
                            )
                        )

                ax.set_ylabel("", fontsize=12)

            # Scatter plot
            else:
                for label, color in zip(range(int(n_clusters)), colors.colors):
                    if cluster != -1:
                        idx = labels == cluster
                        color = colors.colors[cluster]
                    else:
                        idx = labels == label

                    ax.scatter(
                        variables[j][idx],
                        variables[i][idx],
                        s=1,
                        alpha=scatter_alpha,
                        color=color,
                    )

            # Display variable names only on the left and bottom sides
            if i == n_variables - 1:
                ax.set_xlabel(variables[j].name, fontsize=12)
            if j == 0:
                ax.set_ylabel(variables[i].name, fontsize=12)

    # Add a legend in the center of the right side
    plt.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
        fontsize=12,
    )

    plt.tight_layout()
    # if path != "":
    #     plt.savefig(path, dpi=500, bbox_inches="tight")
    # plt.show()
    return fig


def plot_pairplots_3d(variables: list, labels: np.array):
    Logger.getLogger().debug("Plotting 3D distributtions...")

    n_clusters = max(labels) + 1
    colors = ListedColormap(plt.cm.tab10.colors).colors

    # Get the number of variables
    n_variables = len(variables)

    # Calculate the number of rows and columns for the subplots
    ncols = n_variables
    nrows = n_variables

    fig = plt.figure(figsize=(4 * n_variables, 4 * n_variables))

    # Plotting
    for i in range(n_variables):
        for j in range(n_variables):
            if i == j:  # Histograms
                ax = fig.add_subplot(nrows, ncols, i * n_variables + j + 1)
                for label in range(int(n_clusters)):
                    idx = labels == label
                    sns.kdeplot(
                        variables[i][idx],
                        ax=ax,
                        fill=True,
                        color=colors[label],
                        alpha=0.5,
                    )
                ax.set_ylabel("")

            else:  # 3D Scatter plot
                ax = fig.add_subplot(
                    nrows, ncols, i * n_variables + j + 1, projection="3d"
                )
                for label in range(int(n_clusters)):
                    idx = labels == label
                    ax.scatter(
                        variables[j][idx],
                        labels[idx],
                        variables[i][idx],
                        s=10,
                        alpha=0.6,
                        color=colors[label],
                        label=f"Cluster {label}",
                    )

                ax.set_xlabel(variables[j].name)
                ax.set_zlabel(variables[i].name)
                if i == n_variables - 1:
                    ax.set_ylabel("Cluster Label")

    # Creating legend with color patches
    legend_elements = [
        mpatches.Patch(facecolor=colors[i], label=f"Cluster {i}")
        for i in range(int(n_clusters))
    ]
    fig.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(0.85, 0.85))
    return fig
