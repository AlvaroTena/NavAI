import numpy as np
import plotly.graph_objects as go


def generate_HV_errors_plot(baseline_errors, agent_errors, combined=True):
    """
    Generates a comparative plot of horizontal and vertical errors between a baseline and an agent.

    This function creates a Plotly figure (go.Figure) that visualizes the horizontal and vertical errors over time
    for both the baseline and the agent. For the baseline, horizontal and vertical error curves are plotted.
    For the agent, the corresponding error curves are plotted. When `combined` is True, uncertainty bands (max and
    min values) around the agent's mean error are also displayed for each axis.

    Parameters:
        baseline_errors (dict): A dictionary containing the baseline error data. Expected keys:
            - "Epoch": A list or array of time steps.
            - "HorizontalError": Values representing the horizontal error.
            - "VerticalError": Values representing the vertical error.
        agent_errors (dict): A dictionary indexed by epochs containing the agent's error data. Each entry should
            include keys "HorizontalError" and "VerticalError". If `combined` is True, these values should be dictionaries
            containing the keys "mean", "max", and "min"; otherwise, numerical values are expected.
        combined (bool, optional): Indicates whether the agent errors are provided in a combined format (with mean,
            max, and min values). Defaults to True.

    Returns:
        go.Figure: A Plotly figure object containing the generated plot with error curves for both the baseline
        and the agent, including uncertainty bands if `combined` is True.

    Example:
        >>> fig = generate_HV_errors_plot(baseline_errors, agent_errors, combined=True)
        >>> fig.show()
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
    agent_epoch = sorted(agent_errors.keys())

    if combined:
        horizontal_error = [
            agent_errors[epoch]["HorizontalError"]["mean"]
            for epoch in agent_epoch
            if "HorizontalError" in agent_errors[epoch]
        ]
    else:
        horizontal_error = np.array(
            [
                agent_errors[epoch]["HorizontalError"]
                for epoch in agent_epoch
                if "HorizontalError" in agent_errors[epoch]
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=agent_epoch,
            y=horizontal_error,
            mode="lines",
            line=dict(color="mediumorchid"),
            name="Agent Horizontal Error",
            legendgroup="Agent Horizontal Error",
        )
    )
    if combined:
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=[
                    agent_errors[epoch]["HorizontalError"]["max"]
                    for epoch in agent_epoch
                ],
                mode="lines",
                line=dict(color="mediumorchid", width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent Horizontal Error",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=[
                    agent_errors[epoch]["HorizontalError"]["min"]
                    for epoch in agent_epoch
                ],
                mode="lines",
                line=dict(color="mediumorchid", width=0),
                fill="tonexty",
                fillcolor="rgba(186,85,211,0.2)",
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent Horizontal Error",
            )
        )

    if combined:
        vertical_error = [
            agent_errors[epoch]["VerticalError"]["mean"]
            for epoch in agent_epoch
            if "VerticalError" in agent_errors[epoch]
        ]
    else:
        vertical_error = np.array(
            [
                agent_errors[epoch]["VerticalError"]
                for epoch in agent_epoch
                if "VerticalError" in agent_errors[epoch]
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=agent_epoch,
            y=vertical_error,
            mode="lines",
            line=dict(color="crimson"),
            name="Agent Vertical Error",
            legendgroup="Agent Vertical Error",
        )
    )
    if combined:
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=[
                    agent_errors[epoch]["VerticalError"]["max"] for epoch in agent_epoch
                ],
                mode="lines",
                line=dict(color="crimson", width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent Vertical Error",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=[
                    agent_errors[epoch]["VerticalError"]["min"] for epoch in agent_epoch
                ],
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
        xaxis_title="Time Step",
        yaxis_title="Error (m)",
        showlegend=True,
    )

    return fig


def generate_NEU_errors_plot(baseline_errors, agent_errors, combined=True):
    """
    Generates a comparative plot of North, East, and Up errors between a baseline and an agent.

    This function creates a Plotly figure (go.Figure) that visualizes the North, East, and Up errors over time
    for both the baseline and the agent. For the baseline, it plots separate line traces for NorthError,
    EastError, and UpError. For the agent, it plots the corresponding error curves. When `combined` is True,
    it also displays uncertainty bands based on the standard deviation around the mean error for each error type.

    Parameters:
        baseline_errors (dict): A dictionary containing the baseline error data. Expected keys:
            - "Epoch": A list or array of time steps.
            - "NorthError": Values representing the North error.
            - "EastError": Values representing the East error.
            - "UpError": Values representing the Up error.
        agent_errors (dict): A dictionary indexed by epochs containing the agent's error data. Each entry should
            include keys "NorthError", "EastError", and "UpError". If `combined` is True, these values should be
            dictionaries containing the keys "mean" and "std"; otherwise, numerical values are expected.
        combined (bool, optional): Indicates whether the agent errors are provided in a combined format (with mean and
            standard deviation values). Defaults to True.

    Returns:
        go.Figure: A Plotly figure object containing the generated plot with error curves for both the baseline and
        the agent, including uncertainty bands if `combined` is True.

    Example:
        >>> fig = generate_NEU_errors_plot(baseline_errors, agent_errors, combined=True)
        >>> fig.show()
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
    agent_epoch = sorted(agent_errors.keys())

    if combined:
        north_error = np.array(
            [
                agent_errors[epoch]["NorthError"]["mean"]
                for epoch in agent_epoch
                if "NorthError" in agent_errors[epoch]
            ]
        )
        north_error_std = np.array(
            [
                agent_errors[epoch]["NorthError"]["std"]
                for epoch in agent_epoch
                if "NorthError" in agent_errors[epoch]
            ]
        )
    else:
        north_error = np.array(
            [
                agent_errors[epoch]["NorthError"]
                for epoch in agent_epoch
                if "NorthError" in agent_errors[epoch]
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=agent_epoch,
            y=north_error,
            mode="lines",
            line=dict(color="limegreen"),
            name="Agent NorthError",
            legendgroup="Agent NorthError",
        )
    )
    if combined:
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=north_error + north_error_std,
                mode="lines",
                line=dict(color="limegreen", width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent NorthError",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=north_error - north_error_std,
                mode="lines",
                line=dict(color="limegreen", width=0),
                fill="tonexty",
                fillcolor="rgba(50,205,50,0.2)",
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent NorthError",
            )
        )

    if combined:
        east_error = np.array(
            [
                agent_errors[epoch]["EastError"]["mean"]
                for epoch in agent_epoch
                if "EastError" in agent_errors[epoch]
            ]
        )
        east_error_std = np.array(
            [
                agent_errors[epoch]["EastError"]["std"]
                for epoch in agent_epoch
                if "EastError" in agent_errors[epoch]
            ]
        )
    else:
        east_error = np.array(
            [
                agent_errors[epoch]["EastError"]
                for epoch in agent_epoch
                if "EastError" in agent_errors[epoch]
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=agent_epoch,
            y=east_error,
            mode="lines",
            line=dict(color="gold"),
            name="Agent EastError",
            legendgroup="Agent EastError",
        )
    )
    if combined:
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=east_error + east_error_std,
                mode="lines",
                line=dict(color="gold", width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent EastError",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=east_error - east_error_std,
                mode="lines",
                line=dict(color="gold", width=0),
                fill="tonexty",
                fillcolor="rgba(255,215,0,0.2)",
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent EastError",
            )
        )

    if combined:
        up_error = np.array(
            [
                agent_errors[epoch]["UpError"]["mean"]
                for epoch in agent_epoch
                if "UpError" in agent_errors[epoch]
            ]
        )
        up_error_std = np.array(
            [
                agent_errors[epoch]["UpError"]["std"]
                for epoch in agent_epoch
                if "UpError" in agent_errors[epoch]
            ]
        )
    else:
        up_error = np.array(
            [
                agent_errors[epoch]["UpError"]
                for epoch in agent_epoch
                if "UpError" in agent_errors[epoch]
            ]
        )

    fig.add_trace(
        go.Scatter(
            x=agent_epoch,
            y=up_error,
            mode="lines",
            line=dict(color="royalblue"),
            name="Agent UpError",
            legendgroup="Agent UpError",
        )
    )
    if combined:
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=up_error + up_error_std,
                mode="lines",
                line=dict(color="royalblue", width=0),
                showlegend=False,
                hoverinfo="skip",
                legendgroup="Agent UpError",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=agent_epoch,
                y=up_error - up_error_std,
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
        xaxis_title="Time Step",
        yaxis_title="Error (m)",
        showlegend=True,
    )

    return fig
