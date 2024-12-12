import math
from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt


def choose_subplot_dimensions(k: int, max_col: int) -> Tuple[int, int]:
    """
    This function returns the number of rows and columns needed to create a grid of subplots for a given number of plots and maximum number of columns allowed.

    Parameters:
    -----------
    k : int
        The total number of plots.
    max_col : int
        The maximum number of columns allowed in the grid.

    Returns:
    --------
    Tuple[int, int]
        A tuple containing the number of rows and columns needed to create a grid of subplots.
    """
    return math.ceil(k / max_col), max_col


def generate_subplots(
    k: int, max_col: int = 5, row_wise: bool = False, diag_2d: bool = False
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    This function generates a grid of subplots in 2D or 3D with the diagonal in 2D, depending on the value of the 'diag_2d' parameter. The function returns a tuple of the generated figure and axes.

    Parameters:
    -----------
    k : int
        The total number of subplots.
    max_col : int, optional
        The maximum number of columns allowed in the grid. Default value is 5.
    row_wise : bool, optional
        Determines whether the subplots will be arranged row-wise (True) or column-wise (False). Default value is False.
    diag_2d : bool, optional
        If set to True, the diagonal subplot will be in 2D and the rest of the grid will be in 3D. If set to False, the entire grid will be in 2D. Default value is False.

    Returns:
    --------
    Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes | List[matplotlib.axes.Axes]]
        A tuple containing the generated figure and the corresponding axes. If there is only one plot, returns a single Axes object. Otherwise, returns a list of Axes objects.
    """
    nrow, ncol = choose_subplot_dimensions(k, max_col)

    # Choose your share X and share Y parameters as you wish:
    figure, axes = plt.subplots(
        nrow,
        ncol,
        sharex=False,
        sharey=False,
        figsize=(3 * nrow, 3 * ncol),
        subplot_kw={"projection": "3d" if diag_2d else None},
    )

    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Check if it's an array. If there's only one plot, it's just an Axes obj
    if not isinstance(axes, np.ndarray):
        if diag_2d:
            axes = axes.flatten()
        else:
            axes = [axes]
        return figure, axes
    else:
        # Choose the traversal you'd like: 'F' is col-wise, 'C' is row-wise
        axes = axes.flatten(order=("C" if row_wise else "F"))

        # Delete any unused axes from the figure, so that they don't show
        # blank x- and y-axis lines
        for idx, ax in enumerate(axes[k:]):
            figure.delaxes(ax)

            # Turn ticks on for the last ax in each column, wherever it lands
            idx_to_turn_on_ticks = idx + k - ncol if row_wise else idx + k - 1
            for tk in axes[idx_to_turn_on_ticks].get_xticklabels():
                tk.set_visible(True)

        axes = axes[:k]
        if diag_2d:
            for i in range(nrow):
                for j in range(ncol):
                    if i == j:
                        axes[i * ncol + j].remove()
                        axes[i * ncol + j] = figure.add_subplot(
                            nrow, ncol, i * ncol + j + 1
                        )

        return figure, axes
