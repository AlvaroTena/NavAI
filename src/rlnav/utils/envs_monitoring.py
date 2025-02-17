from typing import Any, Dict

import numpy as np
import pandas as pd


def extract_computation_times(parallel_envs, envs_times):
    """
    Extracts and computes the mean and standard deviation of computation times from parallel environments.

    This function retrieves computation times from a list of parallel environments, calculates the mean
    and standard deviation for each time key, and updates the provided `envs_times` dictionary with these
    statistics.

    Parameters:
        parallel_envs (list): A list of parallel environment objects that support the `call` method to
                            retrieve computation times.
        envs_times (dict): A dictionary to be updated with the mean and standard deviation of computation
                        times for each time key.

    Returns:
        dict: The updated `envs_times` dictionary containing the mean and standard deviation for each
            computation time key.
    """
    envs_times_aux = {
        f"env_{i}": sub_env.call("get_times")()
        for i, sub_env in enumerate(parallel_envs)
    }

    mean_std = {}
    for time_key in envs_times_aux[next(iter(envs_times_aux))]:
        mean_std[time_key] = {}
        values = [
            times_dict[time_key]
            for times_dict in envs_times_aux.values()
            if times_dict[time_key]  # Check not empty
        ]

        mean_std[time_key]["mean"] = np.mean(values, axis=0).tolist() if values else []
        mean_std[time_key]["std"] = np.std(values, axis=0).tolist() if values else []

    for time_key, stats in mean_std.items():
        if time_key not in envs_times:
            envs_times[time_key] = {"mean": [], "std": []}

        envs_times[time_key]["mean"].extend(stats["mean"])
        envs_times[time_key]["std"].extend(stats["std"])

    return envs_times


def extract_agent_stats(parallel_envs, agent_stats):
    """
    Extracts and aggregates statistical metrics from multiple parallel environments.

    This function retrieves logging data from each environment, computes mean, standard deviation,
    and other statistics for various error metrics, and updates the agent_stats dictionary accordingly.
    AI error metrics are handled separately and stored under a dedicated key.

    Parameters:
        parallel_envs (list): Environments to extract logging data from.
        agent_stats (dict): Dictionary to update with aggregated metrics.

    Returns:
        tuple: A tuple containing:
            - The updated agent_stats dictionary with aggregated statistics.
            - A dictionary mapping each environment to its raw ai_errors log data.
    """
    # Retrieve logging data from each environment
    log_data = {
        f"env_{i}": sub_env.call("get_logging_data")()
        for i, sub_env in enumerate(parallel_envs)
    }

    mean_std = {}
    dfs = [pd.DataFrame(env_log["ai_errors"]) for env_log in log_data.values()]
    all_data = pd.concat(dfs, ignore_index=True)

    aggregated_stats = all_data.groupby("Epoch").agg(
        {
            "NorthError": [np.mean, np.std, np.min, np.max],
            "EastError": [np.mean, np.std, np.min, np.max],
            "UpError": [np.mean, np.std, np.min, np.max],
            "HorizontalError": [np.mean, np.std, np.min, np.max],
            "VerticalError": [np.mean, np.std, np.min, np.max],
        }
    )

    stats_dict = aggregated_stats.to_dict("index")
    mean_std["AI_Errors"] = {
        epoch: {
            error: {
                stat: value for (err, stat), value in subdict.items() if err == error
            }
            for error in {err for (err, _) in subdict.keys()}
        }
        for epoch, subdict in stats_dict.items()
    }

    for log_key in log_data[next(iter(log_data))]:
        if log_key != "ai_errors":
            mean_std[log_key] = {}
            values = [log_dict[log_key] for log_dict in log_data.values()]

            mean_std[log_key]["mean"] = np.mean(values, axis=0).tolist()
            mean_std[log_key]["std"] = np.std(values, axis=0).tolist()

    for metric_key, stats in mean_std.items():
        if metric_key != "AI_Errors" and metric_key not in agent_stats:
            agent_stats[metric_key] = {"mean": [], "std": []}
        elif metric_key == "AI_Errors" and "agent_errors" not in agent_stats:
            agent_stats["agent_errors"] = {}

        if metric_key != "AI_Errors":
            agent_stats[metric_key]["mean"].extend(stats["mean"])
            agent_stats[metric_key]["std"].extend(stats["std"])

        else:
            agent_stats["agent_errors"] = merge_agent_errors(
                agent_stats.get("agent_errors", {}), mean_std["AI_Errors"]
            )

    return agent_stats, {k: pd.DataFrame(v["ai_errors"]) for k, v in log_data.items()}


def merge_agent_errors(
    old_errors: Dict[Any, Dict[str, Dict[str, Any]]],
    new_errors: Dict[Any, Dict[str, Dict[str, Any]]],
) -> Dict[Any, Dict[str, Dict[str, Any]]]:
    """
    Merge two dictionaries of agent errors by epoch.

    Each dictionary has epochs as keys and values that are dictionaries
    mapping error types to their statistiques (e.g., "max", "min", etc.),
    which are lists of numbers. For epochs present in both dictionaries:
      - "mean": element-wise average.
      - "std": combined standard deviation computed as:
               sqrt(((old_std² + new_std²) / 2) + ((old_mean - new_mean)² / 4))
      - "max" stats are combined element-wise using np.maximum.
      - "min" stats are combined element-wise using np.minimum.

    If an epoch from new_errors is absent in old_errors, it is added directly.

    Parameters:
        old_errors (dict): Existing error metrics organized by epoch.
        new_errors (dict): New error metrics to merge, organized by epoch.

    Returns:
        dict: The merged error metrics dictionary.
    """
    merged_errors = old_errors.copy()

    for epoch, epoch_errors in new_errors.items():
        if epoch in merged_errors:
            for error_key, new_stat_dict in epoch_errors.items():
                # Ensure the error_key exists in merged_errors for the current epoch
                if error_key not in merged_errors[epoch]:
                    merged_errors[epoch][error_key] = new_stat_dict
                    continue

                old_stat_dict = merged_errors[epoch][error_key]

                if "mean" in new_stat_dict and "mean" in old_stat_dict:
                    old_mean = np.array(old_stat_dict["mean"])
                    new_mean = np.array(new_stat_dict["mean"])
                    combined_mean = ((old_mean + new_mean) / 2.0).tolist()
                    merged_errors[epoch][error_key]["mean"] = combined_mean

                if (
                    "std" in new_stat_dict
                    and "std" in old_stat_dict
                    and "mean" in new_stat_dict
                    and "mean" in old_stat_dict
                ):
                    old_std = np.array(old_stat_dict["std"])
                    new_std = np.array(new_stat_dict["std"])

                    combined_var = ((old_std**2 + new_std**2) / 2.0) + (
                        ((old_mean - new_mean) ** 2) / 4.0
                    )
                    combined_std = np.sqrt(combined_var).tolist()
                    merged_errors[epoch][error_key]["std"] = combined_std

                if "max" in new_stat_dict and "max" in old_stat_dict:
                    merged_errors[epoch][error_key]["max"] = np.maximum(
                        np.array(old_stat_dict["max"]), np.array(new_stat_dict["max"])
                    ).tolist()
                if "min" in new_stat_dict and "min" in old_stat_dict:
                    merged_errors[epoch][error_key]["min"] = np.minimum(
                        np.array(old_stat_dict["min"]), np.array(new_stat_dict["min"])
                    ).tolist()
        else:
            # If the epoch doesn't exist in merged_errors, add it directly.
            merged_errors[epoch] = {
                error_key: value for error_key, value in epoch_errors.items()
            }

    return merged_errors
