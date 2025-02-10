import numpy as np


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
    envs_times_aux = {}
    for i, sub_env in enumerate(parallel_envs):
        sub_env_times = sub_env.call("get_times")
        envs_times_aux.update({f"env_{i}": sub_env_times()})

    mean_std = {}
    for time_key in envs_times_aux[next(iter(envs_times_aux))]:
        mean_std[time_key] = {}
        values = [times_dict[time_key] for times_dict in envs_times_aux.values()]
        mean_std[time_key]["mean"] = np.mean(values, axis=0).tolist()
        mean_std[time_key]["std"] = np.std(values, axis=0).tolist()

    for time_key, stats in mean_std.items():
        if time_key not in envs_times:
            envs_times[time_key] = {"mean": [], "std": []}

        envs_times[time_key]["mean"].extend(stats["mean"])
        envs_times[time_key]["std"].extend(stats["std"])

    return envs_times


def extract_agent_stats(parallel_envs, agent_stats):
    """
    Extracts and aggregates statistical data from multiple parallel environments.

    This function retrieves logging data from each environment in the `parallel_envs`
    list, calculates the mean and standard deviation for each metric, and updates
    the `agent_stats` dictionary with these aggregated statistics. Special handling
    is applied for "ai_errors" metrics, which are stored under the "agent_errors"
    key in `agent_stats`.

    Parameters:
        parallel_envs (list): A list of environments from which to extract logging data.
        agent_stats (dict): A dictionary to be updated with aggregated statistics.

    Returns:
        dict: The updated `agent_stats` dictionary containing mean and standard
        deviation for each metric across all environments.
    """
    log_data = {}
    for i, sub_env in enumerate(parallel_envs):
        sub_env_data = sub_env.call("get_logging_data")
        log_data.update({f"env_{i}": sub_env_data()})

    mean_std = {}
    for log_key in log_data[next(iter(log_data))]:
        if log_key == "ai_errors":
            error_dict = {}
            for error_key in log_data[next(iter(log_data))][log_key]:
                error_dict[error_key] = {}
                values = [
                    log_dict[log_key][error_key] for log_dict in log_data.values()
                ]
                error_dict[error_key]["mean"] = np.mean(values, axis=0).tolist()
                error_dict[error_key]["std"] = np.std(values, axis=0).tolist()

            mean_std.update({"AI_Errors": error_dict})

        else:
            mean_std[log_key] = {}
            values = [log_dict[log_key] for log_dict in log_data.values()]

            mean_std[log_key]["mean"] = np.mean(values, axis=0).tolist()
            mean_std[log_key]["std"] = np.std(values, axis=0).tolist()

    for metric_key, stats in mean_std.items():
        if metric_key != "AI_Errors" and metric_key not in agent_stats:
            agent_stats[metric_key] = {"mean": [], "std": []}
        elif metric_key == "AI_Errors" and "agent_errors" not in agent_stats:
            agent_stats["agent_errors"] = {}
            for error_key in stats:
                if error_key not in agent_stats["agent_errors"]:
                    agent_stats["agent_errors"][error_key] = {
                        "mean": [],
                        "std": [],
                    }

        if metric_key != "AI_Errors":
            agent_stats[metric_key]["mean"].extend(stats["mean"])
            agent_stats[metric_key]["std"].extend(stats["std"])

        else:
            for error_key, error_stats in stats.items():
                agent_stats["agent_errors"][error_key]["mean"].extend(
                    error_stats["mean"]
                )
                agent_stats["agent_errors"][error_key]["std"].extend(error_stats["std"])

    return agent_stats
