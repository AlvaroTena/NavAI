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

    objective_names = ["North log-ratio", "East log-ratio", "Up log-ratio"]
    color_schemes = {
        "instant_rewards": {
            "colors": ["seagreen", "orange", "dodgerblue"],
            "shades": [
                "rgba(46,139,87,0.2)",
                "rgba(255,165,0,0.2)",
                "rgba(30,144,255,0.2)",
            ],
            "name_prefix": "Instant Reward",
        },
        "running_rewards": {
            "colors": ["mediumseagreen", "darkgoldenrod", "steelblue"],
            "shades": [
                "rgba(60,179,113,0.2)",
                "rgba(184,134,11,0.2)",
                "rgba(70,130,180,0.2)",
            ],
            "name_prefix": "Running Reward",
        },
        "cumulative_rewards": {
            "colors": ["limegreen", "gold", "royalblue"],
            "shades": [
                "rgba(50,205,50,0.2)",
                "rgba(255,215,0,0.2)",
                "rgba(65,105,225,0.2)",
            ],
            "name_prefix": "Cumulative Reward",
        },
    }

    objective_trace_indices = {0: [], 1: [], 2: []}
    trace_counter = 0

    for reward_type, scheme in color_schemes.items():
        mean_data = np.array(rewards[reward_type]["mean"])
        std_data = np.array(rewards[reward_type]["std"])
        T = mean_data.shape[0]
        x = np.arange(T)

        for i in range(mean_data.shape[1]):
            line_color = scheme["colors"][i]
            shade_color = scheme["shades"][i]
            trace_name = f"{scheme['name_prefix']} - {objective_names[i]}"

            # Mean trace
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean_data[:, i],
                    mode="lines",
                    line=dict(color=line_color),
                    name=trace_name,
                    legendgroup=trace_name,
                )
            )
            objective_trace_indices[i].append(trace_counter)
            trace_counter += 1

            # Upper bound trace
            upper_bound = mean_data[:, i] + std_data[:, i]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=upper_bound,
                    mode="lines",
                    line=dict(color=shade_color, width=0),
                    fill=None,
                    showlegend=False,
                    legendgroup=trace_name,
                )
            )
            objective_trace_indices[i].append(trace_counter)
            trace_counter += 1

            # Lower bound trace
            lower_bound = mean_data[:, i] - std_data[:, i]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=lower_bound,
                    mode="lines",
                    line=dict(color=shade_color, width=0),
                    fill="tonexty",
                    showlegend=False,
                    legendgroup=trace_name,
                )
            )
            objective_trace_indices[i].append(trace_counter)
            trace_counter += 1

    def build_visibility_vector(visible_indices):
        vis = [False] * len(fig.data)
        for idx in visible_indices:
            vis[idx] = True
        return vis

    def show_all():
        return [True] * len(fig.data)

    buttons_menu = [
        dict(
            label="Show All",
            method="update",
            args=[{"visible": show_all()}],
        ),
        dict(
            label="North Only",
            method="update",
            args=[{"visible": build_visibility_vector(objective_trace_indices[0])}],
        ),
        dict(
            label="East Only",
            method="update",
            args=[{"visible": build_visibility_vector(objective_trace_indices[1])}],
        ),
        dict(
            label="Up Only",
            method="update",
            args=[{"visible": build_visibility_vector(objective_trace_indices[2])}],
        ),
        dict(
            label="North & East",
            method="update",
            args=[
                {
                    "visible": build_visibility_vector(
                        objective_trace_indices[0] + objective_trace_indices[1]
                    )
                }
            ],
        ),
        dict(
            label="North & Up",
            method="update",
            args=[
                {
                    "visible": build_visibility_vector(
                        objective_trace_indices[0] + objective_trace_indices[2]
                    )
                }
            ],
        ),
        dict(
            label="East & Up",
            method="update",
            args=[
                {
                    "visible": build_visibility_vector(
                        objective_trace_indices[1] + objective_trace_indices[2]
                    )
                }
            ],
        ),
    ]

    fig.update_layout(
        title="Rewards (mean ± std) per Objective Dimension",
        xaxis_title="Step",
        yaxis_title="Reward",
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons_menu,
                direction="down",
                showactive=True,
                x=1.25,
                xanchor="left",
                y=1,
                yanchor="top",
            ),
        ],
    )

    return fig
