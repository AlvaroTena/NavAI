import numpy as np
import plotly.graph_objects as go


def generate_times_plot(envs_times, time_key):
    """
    Generate a Plotly figure visualizing computing time data with mean and standard deviation.

    This function creates a line plot using Plotly to display the mean times and
    their corresponding standard deviation bounds for a given time key from the
    provided environment computing times data.

    Parameters:
        envs_times (dict): A dictionary containing computing time data with keys for 'mean'
                        and 'std' values.
        time_key (str): The key to access specific computing time data within the envs_times
                        dictionary.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the computing time
                                    data visualization.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=np.arange(len(envs_times[time_key]["mean"])),
            y=envs_times[time_key]["mean"],
            mode="lines",
            line=dict(color="blue"),
            name=f"{time_key}",
            legendgroup=f"{time_key}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(envs_times[time_key]["std"])),
            y=np.array(envs_times[time_key]["mean"])
            + np.array(envs_times[time_key]["std"]),
            mode="lines",
            line=dict(color="lightblue"),
            name="Upper Bound",
            showlegend=False,
            legendgroup=f"{time_key}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(envs_times[time_key]["std"])),
            y=np.array(envs_times[time_key]["mean"])
            - np.array(envs_times[time_key]["std"]),
            mode="lines",
            line=dict(color="lightblue"),
            name="Lower Bound",
            fill="tonexty",
            showlegend=False,
            legendgroup=f"{time_key}",
        )
    )

    fig.update_layout(
        title=f"{time_key} (mean ± std)",
        xaxis_title="Step",
        yaxis_title="Time (s)",
        showlegend=True,
    )

    return fig
