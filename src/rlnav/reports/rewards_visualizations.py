import numpy as np
import plotly.graph_objects as go


def generate_rewards_plot(rewards):
    """
    Generate a Plotly figure visualizing rewards data.

    This function creates a line plot with shaded areas representing the mean and
    standard deviation of instant, running, and cumulative rewards over time.
    Each reward type is plotted with a distinct color for clarity.

    Parameters
    ----------
    rewards : dict
        A dictionary containing 'instant_rewards', 'running_rewards', and
        'cumulative_rewards', each with 'mean' and 'std' keys representing
        the respective reward statistics.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure object containing the rewards visualization.
    """
    fig = go.Figure()

    ### Instant Rewards
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["instant_rewards"]["mean"])),
            y=rewards["instant_rewards"]["mean"],
            mode="lines",
            line=dict(color="RoyalBlue"),
            name="Reward",
            legendgroup="Reward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["instant_rewards"]["std"])),
            y=np.array(rewards["instant_rewards"]["mean"])
            + np.array(rewards["instant_rewards"]["std"]),
            mode="lines",
            line=dict(color="lightblue", width=0),
            fillcolor="rgba(65,105,225,0.2)",
            name="Upper Bound",
            showlegend=False,
            legendgroup="Reward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["instant_rewards"]["std"])),
            y=np.array(rewards["instant_rewards"]["mean"])
            - np.array(rewards["instant_rewards"]["std"]),
            mode="lines",
            line=dict(color="lightblue", width=0),
            fillcolor="rgba(65,105,225,0.2)",
            name="Lower Bound",
            fill="tonexty",
            showlegend=False,
            legendgroup="Reward",
        )
    )

    ### Running Rewards
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["running_rewards"]["mean"])),
            y=rewards["running_rewards"]["mean"],
            mode="lines",
            line=dict(color="ForestGreen"),
            name="Running Reward",
            legendgroup="Running Reward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["running_rewards"]["std"])),
            y=np.array(rewards["running_rewards"]["mean"])
            + np.array(rewards["running_rewards"]["std"]),
            mode="lines",
            line=dict(color="lightgreen", width=0),
            fillcolor="rgba(34,139,34,0.2)",
            name="Upper Bound",
            showlegend=False,
            legendgroup="Running Reward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["running_rewards"]["std"])),
            y=np.array(rewards["running_rewards"]["mean"])
            - np.array(rewards["running_rewards"]["std"]),
            mode="lines",
            line=dict(color="lightgreen", width=0),
            fillcolor="rgba(34,139,34,0.2)",
            name="Lower Bound",
            fill="tonexty",
            showlegend=False,
            legendgroup="Running Reward",
        )
    )

    ### Cummulative Rewards
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["cumulative_rewards"]["mean"])),
            y=rewards["cumulative_rewards"]["mean"],
            mode="lines",
            line=dict(color="DarkOrange"),
            name="Cummulative Reward",
            legendgroup="Cummulative Reward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["cumulative_rewards"]["std"])),
            y=np.array(rewards["cumulative_rewards"]["mean"])
            + np.array(rewards["cumulative_rewards"]["std"]),
            mode="lines",
            line=dict(color="orange", width=0),
            fillcolor="rgba(255,140,0,0.2)",
            name="Upper Bound",
            showlegend=False,
            legendgroup="Cummulative Reward",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(rewards["cumulative_rewards"]["std"])),
            y=np.array(rewards["cumulative_rewards"]["mean"])
            - np.array(rewards["cumulative_rewards"]["std"]),
            mode="lines",
            line=dict(color="orange", width=0),
            fillcolor="rgba(255,140,0,0.2)",
            name="Lower Bound",
            fill="tonexty",
            showlegend=False,
            legendgroup="Cummulative Reward",
        )
    )

    fig.update_layout(
        title="Rewards (mean ± std)",
        xaxis_title="Step",
        yaxis_title="Reward",
    )

    return fig
