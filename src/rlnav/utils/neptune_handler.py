import os

from neptune import Run
from rlnav.reports.error_visualizations import (
    generate_HV_errors_plot,
    generate_NEU_errors_plot,
)
from rlnav.reports.rewards_visualizations import generate_rewards_plot
from rlnav.reports.times_visualization import generate_times_plot


def log_computation_times(envs_times: dict, output_path: str, npt_run: Run) -> None:
    """
    Log computation time data by generating plots and saving them to file or uploading to Neptune.

    Parameters:
        envs_times (dict): Dictionary containing computation time data.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    for time_key in envs_times:
        fig_time = generate_times_plot(envs_times, time_key)
        if npt_run._mode == "debug":
            os.makedirs(
                os.path.join(output_path, "Training_Tracing/times/"),
                exist_ok=True,
            )
            fig_time.write_html(
                os.path.join(
                    output_path,
                    "Training_Tracing/times/",
                    f"times_{time_key}.html",
                )
            )

        else:
            npt_run[f"monitoring/times/{time_key}"].upload(
                fig_time, include_plotlyjs="cdn"
            )


def log_rewards(
    agent_stats: dict, output_path: str, scenario: str, generation: int, npt_run: Run
) -> None:
    """
    Log rewards data by generating plots and saving them to file or uploading to Neptune.

    Parameters:
        agent_stats (dict): Dictionary containing rewards data.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        scenario (str): The scenario name for Neptune logging path.
        generation (int): The AI generation number for Neptune logging path.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    fig_rewards = generate_rewards_plot(
        {k: v for k, v in agent_stats.items() if k != "agent_errors"}
    )
    if npt_run._mode == "debug":
        fig_rewards.write_html(
            os.path.join(
                output_path,
                "rewards.html",
            )
        )

    else:
        npt_run["training/train/rewards"].upload(fig_rewards, include_plotlyjs="cdn")
        npt_run[f"training/{scenario}/AI_gen{generation}/rewards"].upload(
            fig_rewards, include_plotlyjs="cdn"
        )


def log_hv_per_env(
    envs_generation: list,
    envs_errors: dict,
    baseline_errors: dict,
    next_gen_errors: dict,
    scenario: str,
    output_path: str,
    npt_run: Run,
) -> None:
    """
    Logs horizontal and vertical error plots for each environment across different generations.

    Parameters:
        envs_generation (list): List of generation numbers for each environment.
        envs_errors (dict): Dictionary mapping environment names to their error data.
        baseline_errors (dict): Dictionary containing baseline error data for comparison.
        next_gen_errors (dict): Dictionary containing error data for the next generation.
        scenario (str): The scenario name used for Neptune logging paths.
        output_path (str): Path where HTML plots will be saved in debug mode.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    for gen, (env, errors) in zip(envs_generation, envs_errors.items()):
        if errors.empty:
            continue

        fig_HV = generate_HV_errors_plot(
            baseline_errors=baseline_errors,
            agent_errors=errors.to_dict("index"),
            combined=False,
        )

        if npt_run._mode == "debug":
            os.makedirs(
                (
                    new_path := os.path.join(
                        "/".join(output_path.split("/")[:-1]),
                        f"AI_gen{gen}",
                        f"{env}",
                    )
                ),
                exist_ok=True,
            )
            fig_HV.write_html(
                os.path.join(
                    new_path,
                    "HV_errors.html",
                )
            )
        else:
            npt_run[f"training/train/{env}/HV"].upload(fig_HV, include_plotlyjs="cdn")
            npt_run[f"training/{scenario}/AI_gen{gen}/{env}/HV"].upload(
                fig_HV, include_plotlyjs="cdn"
            )

        del fig_HV

        if next_gen_errors and env in next_gen_errors:
            fig_HV = generate_HV_errors_plot(
                baseline_errors=baseline_errors,
                agent_errors=next_gen_errors[env].to_dict("index"),
                combined=False,
            )

            if npt_run._mode == "debug":
                os.makedirs(
                    (
                        new_path := os.path.join(
                            "/".join(output_path.split("/")[:-1]),
                            f"AI_gen{gen+1}",
                            f"{env}",
                        )
                    ),
                    exist_ok=True,
                )
                fig_HV.write_html(
                    os.path.join(
                        new_path,
                        "HV_errors.html",
                    )
                )

            else:
                npt_run[f"training/train/{env}/HV"].upload(
                    fig_HV, include_plotlyjs="cdn"
                )
                npt_run[f"training/{scenario}/AI_gen{gen+1}/{env}/HV"].upload(
                    fig_HV, include_plotlyjs="cdn"
                )


def log_hv_global(
    agent_errors: dict,
    baseline_errors: dict,
    output_path: str,
    scenario: str,
    generation: int,
    npt_run: Run,
) -> None:
    """
    Logs horizontal and vertical error data by generating plots and uploading them to Neptune.

    Parameters:
        agent_errors (dict): Dictionary containing the agent's error data.
        baseline_errors (dict): Dictionary containing the baseline error data.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        scenario (str): The scenario name for Neptune logging.
        generation (int): The AI generation number for Neptune logging.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    fig_HV = generate_HV_errors_plot(
        baseline_errors=baseline_errors,
        agent_errors=agent_errors,
    )

    if npt_run._mode == "debug":
        fig_HV.write_html(
            os.path.join(
                output_path,
                "HV_errors.html",
            )
        )

    else:
        npt_run[f"training/train/HV"].upload(fig_HV, include_plotlyjs="cdn")
        npt_run[f"training/{scenario}/AI_gen{generation}/HV"].upload(
            fig_HV, include_plotlyjs="cdn"
        )


def log_hv(
    envs_errors: dict,
    next_gen_errors: dict,
    agent_errors: dict,
    baseline_errors: dict,
    output_path: str,
    scenario: str,
    envs_generation: dict,
    npt_run: Run,
) -> None:
    """
    Logs horizontal and vertical error data by generating plots for both individual environments and global metrics.

    Parameters:
        envs_errors (dict): Dictionary mapping environment names to their error data.
        next_gen_errors (dict): Dictionary containing error data for the next generation.
        agent_errors (dict): Dictionary containing the agent's aggregated error data.
        baseline_errors (dict): Dictionary containing baseline error data for comparison.
        output_path (str): Path where HTML plots will be saved in debug mode.
        scenario (str): The scenario name used for Neptune logging paths.
        envs_generation (dict): Dictionary mapping environments to their generation numbers.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    log_hv_per_env(
        envs_generation,
        envs_errors,
        baseline_errors,
        next_gen_errors,
        scenario,
        output_path,
        npt_run,
    )
    log_hv_global(
        agent_errors,
        baseline_errors,
        output_path,
        scenario,
        envs_generation[0],
        npt_run,
    )


def log_neu_per_env(
    envs_generation: list,
    envs_errors: dict,
    baseline_errors: dict,
    next_gen_errors: dict,
    scenario: str,
    output_path: str,
    npt_run: Run,
) -> None:
    """
    Logs North-East-Up (NEU) errors for each environment by generating comparative plots and saving them to file or uploading to Neptune.

    Parameters:
        envs_generation (list): List of generation numbers for each environment.
        envs_errors (dict): Dictionary containing error data for each environment.
        baseline_errors (dict): Dictionary containing baseline error data.
        next_gen_errors (dict): Dictionary containing error data for the next generation.
        scenario (str): The scenario name used for Neptune logging path.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    for gen, (env, errors) in zip(envs_generation, envs_errors.items()):
        fig_NEU = generate_NEU_errors_plot(
            baseline_errors=baseline_errors,
            agent_errors=errors.to_dict("index"),
            combined=False,
        )

        if npt_run._mode == "debug":
            os.makedirs(
                (
                    new_path := os.path.join(
                        "/".join(output_path.split("/")[:-1]),
                        f"AI_gen{gen}",
                        f"{env}",
                    )
                ),
                exist_ok=True,
            )
            fig_NEU.write_html(
                os.path.join(
                    new_path,
                    "NEU_errors.html",
                )
            )

        else:
            npt_run[f"training/{scenario}/AI_gen{gen}/{env}/NEU"].upload(
                fig_NEU, include_plotlyjs="cdn"
            )

        if next_gen_errors and env in next_gen_errors:
            fig_NEU = generate_NEU_errors_plot(
                baseline_errors=baseline_errors,
                agent_errors=next_gen_errors[env].to_dict("index"),
                combined=False,
            )

            if npt_run._mode == "debug":
                os.makedirs(
                    (
                        new_path := os.path.join(
                            "/".join(output_path.split("/")[:-1]),
                            f"AI_gen{gen+1}",
                            f"{env}",
                        )
                    ),
                    exist_ok=True,
                )
                fig_NEU.write_html(
                    os.path.join(
                        new_path,
                        "NEU_errors.html",
                    )
                )

            else:
                npt_run[f"training/train/{env}/NEU"].upload(
                    fig_NEU, include_plotlyjs="cdn"
                )
                npt_run[f"training/{scenario}/AI_gen{gen+1}/{env}/NEU"].upload(
                    fig_NEU, include_plotlyjs="cdn"
                )


def log_neu_global(
    agent_errors: dict,
    baseline_errors: dict,
    output_path: str,
    scenario: str,
    generation: int,
    npt_run: Run,
) -> None:
    """
    Logs North-East-Up global errors by generating comparative plots and uploading them to Neptune.

    Parameters:
        agent_errors (dict): Dictionary containing the agent's error data.
        baseline_errors (dict): Dictionary containing the baseline error data.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        scenario (str): The scenario name for Neptune logging.
        generation (int): The AI generation number for Neptune logging.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    fig_NEU = generate_NEU_errors_plot(
        baseline_errors=baseline_errors,
        agent_errors=agent_errors,
    )

    if npt_run._mode == "debug":
        fig_NEU.write_html(
            os.path.join(
                output_path,
                "NEU_errors.html",
            )
        )

    else:
        npt_run[f"training/train/NEU"].upload(fig_NEU, include_plotlyjs="cdn")
        npt_run[f"training/{scenario}/AI_gen{generation}/NEU"].upload(
            fig_NEU, include_plotlyjs="cdn"
        )


def log_neu(
    envs_errors: dict,
    next_gen_errors: dict,
    agent_errors: dict,
    baseline_errors: dict,
    output_path: str,
    scenario: str,
    envs_generation: dict,
    npt_run: Run,
) -> None:
    """
    Logs North-East-Up (NEU) errors by generating both per-environment and global comparative plots.

    Parameters:
        envs_errors (dict): Dictionary containing error data for each environment.
        next_gen_errors (dict): Dictionary containing error data for the next generation.
        agent_errors (dict): Dictionary containing the agent's global error data.
        baseline_errors (dict): Dictionary containing baseline error data.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        scenario (str): The scenario name used for Neptune logging path.
        envs_generation (dict): Dictionary of generation numbers for each environment.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    log_neu_per_env(
        envs_generation,
        envs_errors,
        baseline_errors,
        next_gen_errors,
        scenario,
        output_path,
        npt_run,
    )
    log_neu_global(
        agent_errors,
        baseline_errors,
        output_path,
        scenario,
        envs_generation[0],
        npt_run,
    )


def log_stats(
    agent_stats: dict,
    envs_errors: dict,
    next_gen_errors: dict,
    baseline_errors: dict,
    output_path: str,
    scenario: str,
    envs_generation: dict,
    npt_run: Run,
) -> None:
    """
    Logs comprehensive agent statistics by generating and saving plots for rewards, HV errors, and NEU errors.

    Parameters:
        agent_stats (dict): Dictionary containing agent performance statistics including rewards and errors.
        envs_errors (dict): Dictionary mapping environment names to their error data.
        next_gen_errors (dict): Dictionary containing error data for the next generation.
        baseline_errors (dict): Dictionary containing baseline error data for comparison.
        output_path (str): Path where the HTML plots will be saved if in debug mode.
        scenario (str): The scenario name used for Neptune logging paths.
        envs_generation (dict): Dictionary mapping environments to their generation numbers.
        npt_run (Run): Neptune run object for logging data.

    Returns:
        None
    """
    log_rewards(
        agent_stats,
        output_path,
        scenario,
        envs_generation[0],
        npt_run,
    )

    agent_errors = agent_stats["agent_errors"]
    log_hv(
        envs_errors,
        next_gen_errors,
        agent_errors,
        baseline_errors,
        output_path,
        scenario,
        envs_generation,
        npt_run,
    )
    log_neu(
        envs_errors,
        next_gen_errors,
        agent_errors,
        baseline_errors,
        output_path,
        scenario,
        envs_generation,
        npt_run,
    )
