import argparse
import logging
import os
import signal
import sys
import time
import warnings

import neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from absl import logging as absl_logging
from neptune_tensorboard import enable_tensorboard_logging
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import parallel_py_environment, tf_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from navutils.config import load_config
from navutils.logger import Logger
from navutils.user_interrupt import UserInterruptException, signal_handler
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO
from rlnav.agent.ppo_agent import create_ppo_agent
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
    config,
    wrapperMgr,
    rewardMgr,
    min_values,
    max_values,
    output_path,
    num_parallel_environments,
):
    """
    Crea un entorno paralelo utilizando ParallelPyEnvironment.
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

        # Crear el entorno paralelo
        parallel_env = parallel_py_environment.ParallelPyEnvironment(env_creators)

        # Convertir a entorno compatible con TensorFlow
        return tf_py_environment.TFPyEnvironment(parallel_env)
    except Exception as e:
        print(f"Error al inicializar entornos paralelos: {e}")
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

                scenario_logpath = os.path.join(output_path, scenario)

                train_env = create_parallel_environment(
                    config,
                    wrapperMgr,
                    baseRewardMgr,
                    min_values,
                    max_values,
                    scenario_logpath,
                    config.training.num_parallel_environments,
                )

                for i, sub_env in enumerate(train_env._envs):
                    sub_env.rewardMgr.set_output_path(
                        os.path.join(baseRewardMgr.output_path, f"env_{i}"),
                        scenario,
                        generation,
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
    train_env,
    agent,
    reward_manager,
    scenario,
    generation,
    npt_run,
    log_times,
    envs_times,
    agent_stats,
    training_recorder,
    output_path,
):
    loss = RunningMetric()
    policy_gradient_loss = RunningMetric()
    value_estimation_loss = RunningMetric()
    l2_regularization_loss = RunningMetric()
    entropy_regularization_loss = RunningMetric()
    kl_penalty_loss = RunningMetric()
    clip_fraction = RunningMetric()

    # Configura el replay buffer y el driver.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=600,
    )

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=600,
    )

    # Configura el checkpointer.
    checkpoint_dir = os.path.join(output_path, "checkpoints")
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        agent=agent,
        policy=agent.policy,
        replay_buffer=replay_buffer,
        global_step=agent.train_step_counter,
    )

    # Restaurar desde el último checkpoint si existe.
    train_checkpointer.initialize_or_restore()

    while not all([sub_env.call("is_done")() for sub_env in train_env._envs]):
        try:
            times = {}

            start = time.time()
            driver.run()
            times.update({"envs_collect": [time.time() - start]})

            start = time.time()
            experience = replay_buffer.gather_all()
            train_loss = agent.train(experience)
            times.update({"agent_train": [time.time() - start]})

            if log_times:
                npt_run["monitoring/times"].extend(remove_empty_lists(times))

                envs_times = extract_computation_times(train_env._envs, envs_times)
                for time_key in envs_times:
                    fig_time = generate_times_plot(envs_times, time_key)
                    npt_run[f"monitoring/times/{time_key}"].upload(
                        fig_time, include_plotlyjs="cdn"
                    )

            agent_stats = extract_agent_stats(train_env._envs, agent_stats)

            # Plotting agent rewards
            fig_rewards = generate_rewards_plot(
                {k: v for k, v in agent_stats.items() if k != "agent_errors"}
            )
            npt_run["training/train/rewards"].upload(
                fig_rewards, include_plotlyjs="cdn"
            )
            npt_run[f"training/{scenario}/AI_generation{generation}/rewards"].upload(
                fig_rewards, include_plotlyjs="cdn"
            )

            # Plotting agent vs baseline errors
            baseline_errors = reward_manager.limit_baseline_log(
                reward_manager.initial_epoch, reward_manager.final_epoch
            )
            agent_errors = agent_stats["agent_errors"]

            ## H-V plot
            fig_HV = generate_HV_errors_plot(
                baseline_errors=baseline_errors,
                agent_errors=agent_errors,
            )

            npt_run[f"training/train/HV"].upload(fig_HV, include_plotlyjs="cdn")
            npt_run[f"training/{scenario}/AI_generation{generation}/HV"].upload(
                fig_HV, include_plotlyjs="cdn"
            )

            ## NEU plot
            fig_NEU = generate_NEU_errors_plot(
                baseline_errors=baseline_errors,
                agent_errors=agent_errors,
            )

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

            train_metrics = {}
            train_metrics["loss"] = loss.get_running_value()
            train_metrics["policy_gradient_loss"] = (
                policy_gradient_loss.get_running_value()
            )
            train_metrics["value_estimation_loss"] = (
                value_estimation_loss.get_running_value()
            )
            train_metrics["l2_regularization_loss"] = (
                l2_regularization_loss.get_running_value()
            )
            train_metrics["entropy_regularization_loss"] = (
                entropy_regularization_loss.get_running_value()
            )
            train_metrics["kl_penalty_loss"] = kl_penalty_loss.get_running_value()
            train_metrics["clip_fraction"] = clip_fraction.get_running_value()
            training_recorder.record_metrics(train_metrics)
            npt_run["training/train"].extend({k: [v] for k, v in train_metrics.items()})

            replay_buffer.clear()

            if agent.train_step_counter.numpy() % 10 == 0:
                envs_positions = {}
                for i, sub_env in enumerate(train_env._envs):
                    sub_env.call("update_map")()
                    sub_env_position = sub_env.call("get_ai_positions")()
                    envs_positions.update({f"env_{i}": sub_env_position})

                if envs_positions:
                    df_list = list(envs_positions.values())
                    df_mean_positions = pd.DataFrame(
                        {
                            "LAT_PROP": np.mean(
                                [df["LAT_PROP"] for df in df_list], axis=0
                            ),
                            "LON_PROP": np.mean(
                                [df["LON_PROP"] for df in df_list], axis=0
                            ),
                        }
                    )
                    df_mean_positions.index = df_list[0].index
                    reward_manager.ai_positions = pd.concat(
                        [reward_manager.ai_positions, df_mean_positions]
                    )

                map_file = reward_manager.update_map()
                npt_run[f"training/train/map"].upload(map_file)

                # Guardar un checkpoint.
            if agent.train_step_counter.numpy() % 1000 == 0:
                train_checkpointer.save(agent.train_step_counter.numpy())
                if npt_run.exists(f"training/agent/checkpoint"):
                    del npt_run[f"training/agent/checkpoint"]
                for checkpoint_file in os.listdir(checkpoint_dir):
                    npt_run[f"training/agent/checkpoint/{checkpoint_file}"].upload(
                        os.path.join(checkpoint_dir, checkpoint_file)
                    )

        except WrapperDataAttributeError:
            pass

    for i, sub_env in enumerate(train_env._envs):
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
    args.output_path = os.path.join(args.output_path)
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
