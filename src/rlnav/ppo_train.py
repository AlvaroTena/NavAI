import argparse
import gc
import logging
import os
import signal
import sys
import time
import warnings
from multiprocessing import Process
from typing import Any

import neptune
import numpy as np
import pandas as pd
import rlnav.utils.neptune_handler as npt_handler
import tensorflow as tf
from absl import logging as absl_logging
from dotenv import load_dotenv
from navutils.config import load_config
from navutils.logger import Logger
from navutils.user_interrupt import UserInterruptException, signal_handler
from neptune_tensorboard import enable_tensorboard_logging
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO
from rlnav.agent.ppo_agent import create_ppo_agent
from rlnav.data.experience import pad_batch
from rlnav.drivers.dynamic_step_driver_opt import DynamicStepDriverOpt
from rlnav.env.cascade_parallel_py_environment import CascadeParallelPyEnvironment
from rlnav.env.pe_env import (
    PE_Env,
    create_action_spec,
    create_observation_spec,
    create_reward_spec,
    get_transformers_min_max,
)
from rlnav.env.wrapper import WrapperDataAttributeError
from rlnav.managers.reward_mgr import RewardManager
from rlnav.managers.wrapper_mgr import WrapperManager
from rlnav.recorder.training_recorder import TrainingRecorder
from rlnav.types.running_metric import RunningMetric
from rlnav.utils.envs_monitoring import extract_agent_stats, extract_computation_times
from tf_agents.agents import tf_agent
from tf_agents.environments import tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

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
        min_values,
        max_values,
        output_path,
        name,
        shared_queue,
    ):
        self.config = config
        self.configMgr = configMgr
        self.wrapper_data = wrapper_data
        self.rewardMgr = rewardMgr
        self.scenario = scenario
        self.min_values = min_values
        self.max_values = max_values
        self.output_path = output_path
        self.name = name
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
            num_generations=self.config.scenarios.n_generations,
            min_values=self.min_values,
            max_values=self.max_values,
            transformers_path=self.config.transformed_data.path,
            window_size=(
                self.config.training.rnn.window_size
                if self.config.training.rnn.enable
                else 1
            ),
            output_path=self.output_path,
            name=self.name,
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
        wrapperMgr (WrapperManager): Manager object for wrappers.
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
                min_values=min_values,
                max_values=max_values,
                output_path=output_path,
                name=f"env_{i}",
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
            Logger.Module.ENV,
            f"Error initializing parallel environments: {e}",
        )
        raise


def load_scenarios_list(
    scenarios_path, first_scen, last_scen, priority_scen, scenarios_subset
):
    scenarios = []

    if os.path.exists(scenarios_path):
        scenarios_path = scenarios_path
        all_scenarios = sorted(
            [
                dir
                for dir in os.listdir(scenarios_path)
                if dir.startswith("scenario_") and not dir.endswith(".dvc")
            ]
        )

        for scen in reversed(priority_scen):
            if scen in all_scenarios:
                all_scenarios.remove(scen)
                all_scenarios.insert(0, scen)
            else:
                Logger.log_message(
                    Logger.Category.WARNING,
                    Logger.Module.CONFIG,
                    f"Priority scenario '{priority_scen}' not found. Maintaining normal order.",
                )

        if last_scen != -1:
            scenarios = all_scenarios[first_scen:last_scen]

        else:
            scenarios = all_scenarios[first_scen:]

        if scenarios_subset:
            updated_scenarios = []
            for scenario in scenarios:
                if any([scenario in subset for subset in scenarios_subset]):
                    updated_scenarios.extend(
                        [
                            subset
                            for subset in scenarios_subset
                            if subset.startswith(scenario)
                        ]
                    )
                else:
                    updated_scenarios.append(scenario)
            scenarios = updated_scenarios

    else:
        log_msg = (
            f"Error getting scenarios from non existing directory {scenarios_path}"
        )
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.CONFIG,
            log_msg,
        )
        raise FileNotFoundError(log_msg)

    return scenarios


def run_scenario(scen, scen_path, min_max_values, config, output_path, parsing_rate):
    run_id = os.getenv("NEPTUNE_CUSTOM_RUN_ID")
    run_name = f"{run_id}-{scen[0]}"
    npt_run = neptune.init_run(
        project="AI-PE/RL-GSharp",
        custom_run_id=run_name,
        name=scen[1],
        monitoring_namespace="monitoring/03_agent_training",
        source_files=["config/RLNav/params.yaml", "src/rlnav/**/*.py"],
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )
    npt_run["sys/group_tags"].add(run_id)
    exit_code = run_training_loop(
        scen[1], scen_path, min_max_values, config, output_path, parsing_rate, npt_run
    )
    npt_run.stop()
    return exit_code


def training_orchestrator(config_path, output_path, parsing_rate):
    load_dotenv()
    config = load_config(config_path)

    scenarios_path = config.scenarios.path
    first_scen = config.scenarios.skip_first
    last_scen = config.scenarios.n_scenarios
    priority_scen = config.scenarios.priority
    scenarios_subset = config.scenarios.subscenarios

    min_max_values = get_transformers_min_max(
        os.path.join(config.transformed_data.path, "transforms")
    )

    scenarios = load_scenarios_list(
        scenarios_path, first_scen, last_scen, priority_scen, scenarios_subset
    )

    current_process = None
    try:
        for i, scen in enumerate(scenarios):
            current_process = Process(
                target=run_scenario,
                args=(
                    (i, scen),
                    scenarios_path,
                    min_max_values,
                    config,
                    output_path,
                    parsing_rate,
                ),
            )
            current_process.start()
            current_process.join()

            if current_process.exitcode != 0:
                Logger.log_message(
                    Logger.Category.ERROR,
                    Logger.Module.MAIN,
                    f"Scenario {scen} failed with exit code {current_process.exitcode}. Stopping further execution.",
                )
                return EXIT_FAILURE

    except UserInterruptException as exception:
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{exception.get_msg()}",
        )

        if current_process is not None and current_process.is_alive():
            current_process.terminate()
            current_process.join()

        return EXIT_FAILURE

    return EXIT_SUCCESS


def run_training_loop(
    scenarios,
    scenarios_path,
    min_max_values,
    config,
    output_path,
    parsing_rate,
    npt_run: neptune.Run,
):
    enable_tensorboard_logging(npt_run)

    min_values, max_values = min_max_values

    baseRewardMgr = RewardManager()
    wrapperMgr = WrapperManager(
        scenarios,
        scenarios_path,
        output_path,
        baseRewardMgr,
        npt_run,
    )

    agent = create_ppo_agent(
        (
            tensor_spec.from_spec(create_observation_spec(min_values, max_values)),
            tensor_spec.from_spec(create_action_spec()),
            tensor_spec.from_spec(create_reward_spec()),
        ),
        npt_run,
        rnn=config.training.rnn.enable,
    )

    policy_dir = os.path.join(output_path, "policy")
    policy_saver = common.Checkpointer(ckpt_dir=policy_dir, policy=agent.collect_policy)

    with TrainingRecorder(
        output_path=os.path.join(output_path, "Training_Tracing")
    ) as training_recorder:
        envs_times, agent_stats = {}, {}

        try:
            while wrapperMgr.next_scenario(parsing_rate):
                tf.keras.backend.clear_session()
                gc.collect()

                scenario = wrapperMgr.scenario

                train_env = create_parallel_environment(
                    config,
                    wrapperMgr,
                    baseRewardMgr,
                    min_values,
                    max_values,
                    baseRewardMgr.output_path,
                    config.training.num_parallel_environments,
                )

                if config.neptune.monitoring_times:
                    npt_run["monitoring/times"].extend(
                        remove_empty_lists(wrapperMgr.get_times())
                    )

                agent_stats.update(
                    {
                        "agent_errors": {},
                        "instant_rewards": {"mean": [], "std": []},
                        "running_rewards": {"mean": [], "std": []},
                        "cumulative_rewards": {"mean": [], "std": []},
                    }
                )

                envs_times, agent_stats = train_agent(
                    train_env,
                    agent,
                    baseRewardMgr,
                    scenario,
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

                del train_env
                gc.collect()

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

    if log_times:
        envs_times = extract_computation_times(new_active_envs, envs_times)

    envs_errors = {}

    if not train_env.active_envs:
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.ENV,
            "No active environments found. Exiting training loop.",
        )
        return envs_times, agent_stats

    envs_generation = [None] * total_envs

    while not all([sub_env.call("is_done")() for sub_env in train_env.active_envs]):
        driver._num_steps = max_steps_per_env * train_env.batch_size
        enable_checkpoint = len(train_env.active_envs) == total_envs

        # Collect generation information from active sub-environments
        sub_envs_generations = [
            sub_env.call("get_generation")() for sub_env in train_env.active_envs
        ]

        # Pad with None values if needed to match envs_generation length
        if len(sub_envs_generations) < len(envs_generation):
            sub_envs_generations.extend(
                [None] * (len(envs_generation) - len(sub_envs_generations))
            )

        # Update reward manager's output path if the first generation has changed
        if sub_envs_generations[0] != envs_generation[0]:
            new_path = os.path.join(
                reward_manager.output_path,
                f"AI_gen{sub_envs_generations[0]}",
            )
            reward_manager.set_output_path(new_path)

        # Update envs_generation if the generations have changed
        if sub_envs_generations != envs_generation:
            envs_generation = sub_envs_generations

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
            experience, _ = next(iter(dataset))
            active_count = train_env.batch_size
            experience_filtered = tf.nest.map_structure(
                lambda t: t[:active_count], experience
            )
            train_loss = agent.train(experience_filtered)
            times.update({"agent_train": [time.time() - start]})

            del experience
            del experience_filtered

            if log_times:
                npt_run["monitoring/times"].extend(remove_empty_lists(times))

                envs_times = extract_computation_times(
                    train_env.active_envs, envs_times
                )
                npt_handler.log_computation_times(envs_times, output_path, npt_run)

            agent_stats, raw_envs_errors = extract_agent_stats(
                train_env.active_envs, agent_stats
            )
            next_gen_errors = {}

            for env, errors in raw_envs_errors.items():
                errors.set_index("Epoch", inplace=True)

                if env not in envs_errors:
                    envs_errors[env] = errors.copy(deep=False)
                else:
                    duplicated_epochs = errors.index.intersection(
                        envs_errors[env].index
                    )
                    if not duplicated_epochs.empty:
                        next_gen_errors[env] = errors.loc[duplicated_epochs].copy(
                            deep=False
                        )
                        errors = errors.drop(duplicated_epochs)

                    if not errors.empty:
                        envs_errors[env] = pd.concat(
                            [envs_errors[env], errors]
                        ).sort_index()

                del errors

            baseline_errors = reward_manager.limit_baseline_log(
                reward_manager.initial_epoch, reward_manager.final_epoch
            )
            npt_handler.log_stats(
                agent_stats,
                envs_errors,
                next_gen_errors,
                baseline_errors,
                reward_manager.output_path,
                scenario,
                envs_generation,
                npt_run,
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
                del envs_positions

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

                if log_times:
                    envs_times = extract_computation_times(new_active_envs, envs_times)

        except WrapperDataAttributeError:
            pass

        except StopIteration:
            break

        finally:
            gc.collect()

    for i, sub_env in enumerate(train_env.active_envs):
        map_file = sub_env.call("update_map")()
        npt_run[f"training/{scenario}/AI_gen{envs_generation[i]}/env_{i}/map"].upload(
            map_file
        )
        npt_run[
            f"training/{scenario}/AI_gen{envs_generation[i]}/env_{i}/reward"
        ].upload(sub_env.call("get_reward_filepath")())

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

    return_value = training_orchestrator(
        args.config_file_path,
        args.output_path,
        args.parsing_rate,
    )

    Logger.reset()

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

    Logger(args.output_path)

    from tf_agents.system import multiprocessing as mp

    result = mp.handle_main(main)

    sys.exit(result)
    Logger(args.output_path)

    from tf_agents.system import multiprocessing as mp

    result = mp.handle_main(main)

    sys.exit(result)
