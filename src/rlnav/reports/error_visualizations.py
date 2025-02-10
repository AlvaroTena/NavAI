import numpy as np
import plotly.graph_objects as go


def generate_HV_errors_plot(baseline_errors, agent_errors):
    """
    Generate a Plotly figure visualizing horizontal and vertical errors for both
    baseline and agent data.

    This function creates a line plot comparing the horizontal and vertical errors
    of a baseline model against those of an agent model. The plot includes mean
    error lines for the agent, with shaded areas representing the standard
    deviation. The x-axis represents the step number, while the y-axis represents
    the error in meters.

    Parameters
    ----------
    baseline_errors : dict
        A dictionary containing baseline error data with keys "Epoch", "HorizontalError"
        and "VerticalError".
    agent_errors : dict
        A dictionary containing agent error data with keys "Epoch", "HorizontalError" and
        "VerticalError", each containing "mean" and "std" sub-keys.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object representing the error comparison plot.
    """
    fig = go.Figure()

    ### Baseline
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=baseline_errors["HorizontalError"],
            mode="lines",
            line=dict(color="teal", dash="dash"),
            name="Baseline Horizontal Error",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=baseline_errors["VerticalError"],
            mode="lines",
            line=dict(color="darkred", dash="dash"),
            name="Baseline Vertical Error",
        )
    )

    ### Agent
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=agent_errors["HorizontalError"]["mean"],
            mode="lines",
            line=dict(color="mediumorchid"),
            name="Agent Horizontal Error",
            legendgroup="Agent Horizontal Error",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["HorizontalError"]["mean"])
            + np.array(agent_errors["HorizontalError"]["std"]),
            mode="lines",
            line=dict(color="mediumorchid", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent Horizontal Error",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["HorizontalError"]["mean"])
            - np.array(agent_errors["HorizontalError"]["std"]),
            mode="lines",
            line=dict(color="mediumorchid", width=0),
            fill="tonexty",
            fillcolor="rgba(186,85,211,0.2)",
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent Horizontal Error",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=agent_errors["VerticalError"]["mean"],
            mode="lines",
            line=dict(color="crimson"),
            name="Agent Vertical Error",
            legendgroup="Agent Vertical Error",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["VerticalError"]["mean"])
            + np.array(agent_errors["VerticalError"]["std"]),
            mode="lines",
            line=dict(color="crimson", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent Vertical Error",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["VerticalError"]["mean"])
            - np.array(agent_errors["VerticalError"]["std"]),
            mode="lines",
            line=dict(color="crimson", width=0),
            fill="tonexty",
            fillcolor="rgba(220,20,60,0.2)",
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent Vertical Error",
        )
    )

    fig.update_layout(
        title="Agent vs Baseline H-V Errors",
        xaxis_title="Step",
        yaxis_title="Error (m)",
        showlegend=True,
    )

    return fig


def generate_NEU_errors_plot(baseline_errors, agent_errors):
    """
    Generate a Plotly figure visualizing NEU (North, East, Up) errors for both
    baseline and agent data.

    This function creates a line plot with shaded areas representing the standard
    deviation for the agent's errors. The plot includes separate traces for North,
    East, and Up errors, comparing baseline and agent performance.

    Parameters:
        baseline_errors (dict): A dictionary containing baseline error data with
            keys "NorthError", "EastError", and "UpError".
        agent_errors (dict): A dictionary containing agent error data with keys
            "NorthError", "EastError", and "UpError", each having "mean" and "std"
            sub-keys.

    Returns:
        plotly.graph_objects.Figure: A Plotly figure object representing the NEU
        errors plot.
    """
    fig = go.Figure()

    ### Baseline
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=baseline_errors["NorthError"],
            mode="lines",
            line=dict(color="forestgreen", dash="dash"),
            name="Baseline NorthError",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=baseline_errors["EastError"],
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="Baseline EastError",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=baseline_errors["UpError"],
            mode="lines",
            line=dict(color="darkblue", dash="dash"),
            name="Baseline UpError",
        )
    )

    ### Agent
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=agent_errors["NorthError"]["mean"],
            mode="lines",
            line=dict(color="limegreen"),
            name="Agent NorthError",
            legendgroup="Agent NorthError",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["NorthError"]["mean"])
            + np.array(agent_errors["NorthError"]["std"]),
            mode="lines",
            line=dict(color="limegreen", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent NorthError",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["NorthError"]["mean"])
            - np.array(agent_errors["NorthError"]["std"]),
            mode="lines",
            line=dict(color="limegreen", width=0),
            fill="tonexty",
            fillcolor="rgba(50,205,50,0.2)",
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent NorthError",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=agent_errors["EastError"]["mean"],
            mode="lines",
            line=dict(color="gold"),
            name="Agent EastError",
            legendgroup="Agent EastError",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["EastError"]["mean"])
            + np.array(agent_errors["EastError"]["std"]),
            mode="lines",
            line=dict(color="gold", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent EastError",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["EastError"]["mean"])
            - np.array(agent_errors["EastError"]["std"]),
            mode="lines",
            line=dict(color="gold", width=0),
            fill="tonexty",
            fillcolor="rgba(255,215,0,0.2)",
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent EastError",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=agent_errors["UpError"]["mean"],
            mode="lines",
            line=dict(color="royalblue"),
            name="Agent UpError",
            legendgroup="Agent UpError",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["UpError"]["mean"])
            + np.array(agent_errors["UpError"]["std"]),
            mode="lines",
            line=dict(color="royalblue", width=0),
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent UpError",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=baseline_errors["Epoch"],
            y=np.array(agent_errors["UpError"]["mean"])
            - np.array(agent_errors["UpError"]["std"]),
            mode="lines",
            line=dict(color="royalblue", width=0),
            fill="tonexty",
            fillcolor="rgba(65,105,225,0.2)",
            showlegend=False,
            hoverinfo="skip",
            legendgroup="Agent UpError",
        )
    )

    fig.update_layout(
        title="Agent vs Baseline NEU Errors",
        xaxis_title="Step",
        yaxis_title="Error (m)",
        showlegend=True,
    )

    return fig
