import argparse
import logging
import os
import shutil
import signal
import sys
import time
import warnings
from typing import Any

import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging as absl_logging
from neptune_tensorboard import enable_tensorboard_logging
from tf_agents.agents import tf_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from navutils.config import load_config
from navutils.logger import Logger
from navutils.user_interrupt import UserInterruptException, signal_handler
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO
from rlnav.agent.ppo_agent import create_ppo_agent
from rlnav.data.experience import pad_batch
from rlnav.drivers.dynamic_step_driver_opt import DynamicStepDriverOpt
from rlnav.env.cascade_parallel_py_environment import CascadeParallelPyEnvironment
from rlnav.env.pe_env import PE_Env, get_transformers_min_max
from rlnav.env.wrapper import WrapperDataAttributeError
from rlnav.managers.reward_mgr import RewardManager
from rlnav.managers.wrapper_mgr import WrapperManager
from rlnav.recorder.training_recorder import TrainingRecorder
from rlnav.reports.error_visualizations import (
    generate_HV_errors_plot,
    generate_NEU_errors_plot,
)
from rlnav.reports.rewards_visualizations import generate_rewards_plot
from rlnav.reports.times_visualization import generate_times_plot
from rlnav.types.running_metric import RunningMetric
from rlnav.utils.envs_monitoring import extract_agent_stats, extract_computation_times

warnings.simplefilter(action="ignore", category=FutureWarning)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

eps = np.finfo(np.float32).eps.item()

gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def remove_empty_lists(d):
    if isinstance(d, dict):
        return {
            k: remove_empty_lists(v)
            for k, v in d.items()
            if (isinstance(v, dict) and remove_empty_lists(v))
            or (isinstance(v, list) and v)
            or not isinstance(v, (dict, list))
        }
    return d


class EnvCreator:
    def __init__(
        self,
        config,
        configMgr,
        wrapper_data,
        rewardMgr,
        scenario,
        generation,
        min_values,
        max_values,
        output_path,
        shared_queue,
    ):
        self.config = config
        self.configMgr = configMgr
        self.wrapper_data = wrapper_data
        self.rewardMgr = rewardMgr
        self.scenario = scenario
        self.generation = generation
        self.min_values = min_values
        self.max_values = max_values
        self.output_path = output_path
        self.shared_queue = shared_queue

    def __call__(self):
        """
        Este método permite que la clase actúe como una función al ser llamada.
        """
        absl_logging.use_python_logging(quiet=True)
        Logger.reconfigure_child(log_queue=self.shared_queue)
        return PE_Env(
            configMgr=self.configMgr,
            wrapper_data=self.wrapper_data,
            rewardMgr=self.rewardMgr,
            scenario=self.scenario,
            generation=self.generation,
            min_values=self.min_values,
            max_values=self.max_values,
            transformers_path=self.config.transformed_data.path,
            window_size=(
                self.config.training.rnn.window_size
                if self.config.training.rnn.enable
                else 1
            ),
            output_path=self.output_path,
        )


def create_parallel_environment(
    config: Any,
    wrapperMgr: WrapperManager,
    rewardMgr: RewardManager,
    min_values: list,
    max_values: list,
    output_path: str,
    num_parallel_environments: int,
) -> Any:
    """
    Creates a parallel environment using CascadeParallelPyEnvironment.

    The function creates multiple environment instances using the provided
    configuration, wrapper manager, and reward manager. Each instance is configured
    with specific parameters and a unique output path. Finally, the parallel
    environment is adapted to be TensorFlow compatible.

    Args:
        config (Any): Overall configuration settings.
        wrapperMgr (WrapperManager): Manager object for wrappers and scenario generation.
        rewardMgr (RewardManager): Manager object for reward handling.
        min_values (list): Minimum boundary values for the environment.
        max_values (list): Maximum boundary values for the environment.
        output_path (str): Base directory path for saving outputs.
        num_parallel_environments (int): Number of parallel environments to create.

    Returns:
        Any: A TensorFlow-compatible parallel environment.

    Raises:
        Exception: Propagates any exception that occurs during environment creation.
    """
    try:
        env_creators = [
            EnvCreator(
                config=config,
                configMgr=wrapperMgr.configMgr,
                wrapper_data=wrapperMgr.wrapper_data,
                rewardMgr=rewardMgr,
                scenario=wrapperMgr.scenario,
                generation=wrapperMgr.scenario_generation[wrapperMgr.scenario],
                min_values=min_values,
                max_values=max_values,
                output_path=os.path.join(output_path, f"env_{i}"),
                shared_queue=Logger.get_queue(),
            )
            for i in range(num_parallel_environments)
        ]

        # Create the parallel environment
        parallel_env = CascadeParallelPyEnvironment(env_creators)

        # Convert to a TensorFlow-compatible environment
        return tf_py_environment.TFPyEnvironment(parallel_env)

    except Exception as e:
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.MAIN,
            f"Error initializing parallel environments: {e}",
        )
        raise


def run_training_loop(config_path, output_path, parsing_rate, npt_run: neptune.Run):
    enable_tensorboard_logging(npt_run)
    config = load_config(config_path)

    baseRewardMgr = RewardManager()
    wrapperMgr = WrapperManager(
        config.scenarios.path,
        (config.scenarios.skip_first, config.scenarios.n_scenarios),
        config.scenarios.priority,
        config.scenarios.subscenarios,
        output_path,
        baseRewardMgr,
        npt_run,
        num_generations=config.scenarios.n_generations,
    )

    min_values, max_values = get_transformers_min_max(
        os.path.join(config.transformed_data.path, "transforms")
    )

    env = PE_Env(
        configMgr=wrapperMgr.configMgr,
        wrapper_data=wrapperMgr.wrapper_data,
        rewardMgr=baseRewardMgr,
        generation=wrapperMgr.scenario_generation[wrapperMgr.scenario],
        min_values=min_values,
        max_values=max_values,
        transformers_path=config.transformed_data.path,
        window_size=(
            config.training.rnn.window_size if config.training.rnn.enable else 1
        ),
        output_path=output_path,
    )
    agent = create_ppo_agent(
        tf_py_environment.TFPyEnvironment(env), npt_run, rnn=config.training.rnn.enable
    )

    with TrainingRecorder(
        output_path=os.path.join(output_path, "Training_Tracing")
    ) as training_recorder:
        envs_times, agent_stats = {}, {}

        try:
            while wrapperMgr.next_scenario(parsing_rate):
                scenario = wrapperMgr.scenario
                generation = wrapperMgr.scenario_generation[scenario]

                train_env = create_parallel_environment(
                    config,
                    wrapperMgr,
                    baseRewardMgr,
                    min_values,
                    max_values,
                    baseRewardMgr.output_path,
                    config.training.num_parallel_environments,
                )

                policy_dir = os.path.join(output_path, "policy")
                policy_saver = common.Checkpointer(
                    ckpt_dir=policy_dir, policy=agent.collect_policy
                )

                if config.neptune.monitoring_times:
                    npt_run["monitoring/times"].extend(
                        remove_empty_lists(wrapperMgr.get_times())
                    )

                agent_stats.update({"agent_errors": {}})

                envs_times, agent_stats = train_agent(
                    train_env,
                    agent,
                    baseRewardMgr,
                    scenario,
                    generation,
                    config.training.active_environments_per_iteration,
                    npt_run,
                    config.neptune.monitoring_times,
                    envs_times,
                    agent_stats,
                    training_recorder,
                    output_path,
                )

                train_env.close()

            policy_saver.save(agent.train_step_counter.numpy())
            for policy_file in os.listdir(policy_dir):
                npt_run[f"training/agent/policy/{policy_file}"].upload(
                    os.path.join(policy_dir, policy_file)
                )

            wrapperMgr.close_PE()

            return EXIT_SUCCESS

        except UserInterruptException as exception:
            Logger.log_message(
                Logger.Category.INFO,
                Logger.Module.MAIN,
                f"{exception.get_msg()}",
            )

            wrapperMgr.close_PE()

            return EXIT_FAILURE


def train_agent(
    train_env: tf_py_environment.TFPyEnvironment,
    agent: tf_agent.TFAgent,
    reward_manager: RewardManager,
    scenario: str,
    generation: int,
    active_environments_per_iteration: int,
    npt_run: neptune.Run,
    log_times: bool,
    envs_times: dict,
    agent_stats: dict,
    training_recorder: TrainingRecorder,
    output_path: str,
):
    loss = RunningMetric()
    policy_gradient_loss = RunningMetric()
    value_estimation_loss = RunningMetric()
    l2_regularization_loss = RunningMetric()
    entropy_regularization_loss = RunningMetric()
    kl_penalty_loss = RunningMetric()
    clip_fraction = RunningMetric()

    max_steps_per_env = 600  # 10Hz * 60s = 600 steps
    total_envs = train_env.num_envs

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=total_envs,
        max_length=max_steps_per_env,
    )

    def custom_add_batch(experience):
        active_count = train_env.batch_size

        active_mask = [True] * active_count + [False] * (total_envs - active_count)
        padded_experience = pad_batch(experience, active_mask, total_envs)

        replay_buffer.add_batch(padded_experience)

    driver = DynamicStepDriverOpt(
        train_env,
        agent.collect_policy,
        observers=[custom_add_batch],
    )

    os.makedirs(
        (checkpoint_dir := os.path.join(output_path, "checkpoints")), exist_ok=True
    )
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter,
    )
    train_checkpointer.initialize_or_restore()

    if total_envs < active_environments_per_iteration:
        active_flags = [True] * total_envs
    else:
        active_flags = [True] * active_environments_per_iteration + [False] * (
            total_envs - active_environments_per_iteration
        )

    new_active_envs = train_env.set_active_envs(active_flags)
    train_env._batch_size = train_env.pyenv.batch_size
    train_env._time_step = train_env.pyenv.current_time_step()

    envs_times = extract_computation_times(new_active_envs, envs_times)

    envs_errors = {}

    if not train_env.active_envs:
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.MAIN,
            "No active environments found. Exiting training loop.",
        )
        return envs_times, agent_stats

    while not all([sub_env.call("is_done")() for sub_env in train_env.active_envs]):
        driver._num_steps = max_steps_per_env * train_env.batch_size
        enable_checkpoint = len(train_env.active_envs) == total_envs

        try:
            times = {}

            start = time.time()
            driver.run()
            times.update({"envs_collect": [time.time() - start]})

            start = time.time()
            dataset = replay_buffer.as_dataset(
                single_deterministic_pass=True,
                num_steps=max_steps_per_env,
                sample_batch_size=total_envs,
            )
            experience, info = next(iter(dataset))
            active_count = train_env.batch_size
            experience_filtered = tf.nest.map_structure(
                lambda t: t[:active_count], experience
            )
            train_loss = agent.train(experience_filtered)
            times.update({"agent_train": [time.time() - start]})

            if log_times:
                npt_run["monitoring/times"].extend(remove_empty_lists(times))

                envs_times = extract_computation_times(
                    train_env.active_envs, envs_times
                )
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

            agent_stats, raw_envs_errors = extract_agent_stats(
                train_env.active_envs, agent_stats
            )
            for env, errors in raw_envs_errors.items():
                errors.set_index("Epoch", inplace=True)

                if env not in envs_errors:
                    envs_errors[env] = errors
                else:
                    duplicated_epochs = errors.index.intersection(
                        envs_errors[env].index
                    )
                    if not duplicated_epochs.empty:
                        envs_errors[env].drop(duplicated_epochs, inplace=True)

                    envs_errors[env] = pd.concat(
                        [envs_errors[env], errors]
                    ).sort_index()

            # Plotting agent rewards
            fig_rewards = generate_rewards_plot(
                {k: v for k, v in agent_stats.items() if k != "agent_errors"}
            )
            if npt_run._mode == "debug":
                fig_rewards.write_html(
                    os.path.join(reward_manager.output_path, "rewards.html")
                )

            else:
                npt_run["training/train/rewards"].upload(
                    fig_rewards, include_plotlyjs="cdn"
                )
                npt_run[
                    f"training/{scenario}/AI_generation{generation}/rewards"
                ].upload(fig_rewards, include_plotlyjs="cdn")

            # Plotting agent vs baseline errors
            baseline_errors = reward_manager.limit_baseline_log(
                reward_manager.initial_epoch, reward_manager.final_epoch
            )
            agent_errors = agent_stats["agent_errors"]

            ## H-V plot
            for env, errors in envs_errors.items():
                fig_HV = generate_HV_errors_plot(
                    baseline_errors=baseline_errors,
                    agent_errors=errors.to_dict("index"),
                    combined=False,
                )

                if npt_run._mode == "debug":
                    fig_HV.write_html(
                        os.path.join(
                            reward_manager.output_path, f"{env}", "HV_errors.html"
                        )
                    )
                else:
                    npt_run[f"training/train/{env}/HV"].upload(
                        fig_HV, include_plotlyjs="cdn"
                    )
                    npt_run[
                        f"training/{scenario}/AI_generation{generation}/{env}/HV"
                    ].upload(fig_HV, include_plotlyjs="cdn")

            fig_HV = generate_HV_errors_plot(
                baseline_errors=baseline_errors,
                agent_errors=agent_errors,
            )

            if npt_run._mode == "debug":
                fig_HV.write_html(
                    os.path.join(reward_manager.output_path, "HV_errors.html")
                )

            else:
                npt_run[f"training/train/HV"].upload(fig_HV, include_plotlyjs="cdn")
                npt_run[f"training/{scenario}/AI_generation{generation}/HV"].upload(
                    fig_HV, include_plotlyjs="cdn"
                )

            ## NEU plot
            for env, errors in envs_errors.items():
                fig_NEU = generate_NEU_errors_plot(
                    baseline_errors=baseline_errors,
                    agent_errors=errors.to_dict("index"),
                    combined=False,
                )

                if npt_run._mode == "debug":
                    fig_NEU.write_html(
                        os.path.join(
                            reward_manager.output_path, f"{env}", "NEU_errors.html"
                        )
                    )

                else:
                    npt_run[
                        f"training/{scenario}/AI_generation{generation}/{env}/NEU"
                    ].upload(fig_NEU, include_plotlyjs="cdn")

            fig_NEU = generate_NEU_errors_plot(
                baseline_errors=baseline_errors,
                agent_errors=agent_errors,
            )

            if npt_run._mode == "debug":
                fig_NEU.write_html(
                    os.path.join(reward_manager.output_path, "NEU_errors.html")
                )

            else:
                npt_run[f"training/train/NEU"].upload(fig_NEU, include_plotlyjs="cdn")
                npt_run[f"training/{scenario}/AI_generation{generation}/NEU"].upload(
                    fig_NEU, include_plotlyjs="cdn"
                )

            loss.update(train_loss.loss.numpy())
            policy_gradient_loss.update(train_loss.extra.policy_gradient_loss.numpy())
            value_estimation_loss.update(train_loss.extra.value_estimation_loss.numpy())
            l2_regularization_loss.update(
                train_loss.extra.l2_regularization_loss.numpy()
            )
            entropy_regularization_loss.update(
                train_loss.extra.entropy_regularization_loss.numpy()
            )
            kl_penalty_loss.update(train_loss.extra.kl_penalty_loss.numpy())
            clip_fraction.update(train_loss.extra.clip_fraction.numpy())

            train_metrics = {
                "loss": loss.get_running_value(),
                "policy_gradient_loss": policy_gradient_loss.get_running_value(),
                "value_estimation_loss": value_estimation_loss.get_running_value(),
                "l2_regularization_loss": l2_regularization_loss.get_running_value(),
                "entropy_regularization_loss": entropy_regularization_loss.get_running_value(),
                "kl_penalty_loss": kl_penalty_loss.get_running_value(),
                "clip_fraction": clip_fraction.get_running_value(),
            }
            training_recorder.record_metrics(train_metrics)
            npt_run["training/train"].extend({k: [v] for k, v in train_metrics.items()})

            replay_buffer.clear()

            if agent.train_step_counter.numpy() % 10 == 0:
                envs_positions = {}
                for i, sub_env in enumerate(train_env.active_envs):
                    sub_env.call("update_map")()
                    sub_env_position = sub_env.call("get_ai_positions")()
                    envs_positions.update({f"env_{i}": sub_env_position})

                if envs_positions:
                    df_list = list(envs_positions.values())
                    df_all = pd.concat(df_list, axis=0)
                    df_mean_positions = df_all.groupby(df_all.index).mean()[
                        ["LAT_PROP", "LON_PROP"]
                    ]
                    reward_manager.ai_positions = df_mean_positions

                map_file = reward_manager.update_map(reset=True)
                npt_run[f"training/train/map"].upload(map_file)

            # Guardar un checkpoint.
            if enable_checkpoint:
                train_checkpointer.save(agent.train_step_counter.numpy())
                if agent.train_step_counter.numpy() % 50 == 0:
                    if npt_run.exists(f"training/agent/checkpoint"):
                        del npt_run[f"training/agent/checkpoint"]
                    for checkpoint_file in os.listdir(checkpoint_dir):
                        npt_run[f"training/agent/checkpoint/{checkpoint_file}"].upload(
                            os.path.join(checkpoint_dir, checkpoint_file)
                        )

            current_active = len(train_env.active_envs)
            if current_active < total_envs:
                additional = min(
                    active_environments_per_iteration, total_envs - current_active
                )
                new_active_count = current_active + additional
                active_flags = [True] * new_active_count + [False] * (
                    total_envs - new_active_count
                )
                new_active_envs = train_env.set_active_envs(active_flags)
                train_env._batch_size = train_env.pyenv.batch_size
                train_env._time_step = train_env.pyenv.current_time_step()

                envs_times = extract_computation_times(new_active_envs, envs_times)

        except WrapperDataAttributeError:
            pass

    for i, sub_env in enumerate(train_env.active_envs):
        map_file = sub_env.call("update_map")()
        npt_run[f"training/{scenario}/AI_generation{generation}/env_{i}/map"].upload(
            map_file
        )
        npt_run[f"training/{scenario}/AI_generation{generation}/env_{i}/reward"].upload(
            sub_env.call("get_reward_filepath")()
        )

    train_checkpointer.save(agent.train_step_counter.numpy())
    for checkpoint_file in os.listdir(checkpoint_dir):
        npt_run[f"training/agent/checkpoint/{checkpoint_file}"].upload(
            os.path.join(checkpoint_dir, checkpoint_file)
        )

    return envs_times, agent_stats


def main(argv=sys.argv):
    # Disable absl logging and use python logging
    absl_logging.use_python_logging()
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, absl_logging.ABSLHandler):
            root_logger.removeHandler(handler)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if not args.output_path or not args.debug_level:
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.MAIN,
            f"Not enough parameters. The command line shall be: {argv[0]} --config_file --output_directory --debug_level(INFO/WARNING/ERROR/DEBUG/TRACE) --parsing_rate(optional)",
        )
        parser.print_help()
        return EXIT_FAILURE

    Logger.set_category(args.debug_level)

    if args.parsing_rate:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Parsing rate set at {args.parsing_rate}",
        )
    else:
        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f" Parsing rate input not received. Epochs are processed without rate filter",
        )

    return_value = run_training_loop(
        args.config_file_path,
        args.output_path,
        args.parsing_rate,
        run,
    )

    Logger.reset()
    run.stop()

    return return_value


if __name__ == "__main__":
    ################################################
    ###############  INPUTS SECTION  ###############
    ################################################
    parser = argparse.ArgumentParser(description="Allowed options")
    parser.version = RELEASE_INFO
    parser.add_argument("-v", "--version", help="print version", action="version")
    parser.add_argument(
        "-c", "--config_file", help="config file", dest="config_file_path"
    )
    parser.add_argument(
        "-o", "--output_directory", help="output directory", dest="output_path"
    )
    parser.add_argument(
        "-g", "--debug_level", help="debug level (INFO/WARNING/ERROR/DEBUG)"
    )
    parser.add_argument(
        "-p", "--parsing_rate", type=int, default=0, help="parsing_rate(optional)"
    )

    args = parser.parse_args(sys.argv[1:])
    args.output_path = os.path.normpath(os.path.join(args.output_path))
    sys.argv.insert(1, "--")  # This is needed to avoid absl flags parsing errors

    run = neptune.init_run(
        project="AI-PE/RL-GSharp",
        monitoring_namespace="monitoring/03_agent_training",
        source_files=["config/RLNav/params.yaml", "src/rlnav/**/*.py"],
    )
    Logger(args.output_path, run)

    from tf_agents.system import multiprocessing as mp

    result = mp.handle_main(main)

    sys.exit(result)
