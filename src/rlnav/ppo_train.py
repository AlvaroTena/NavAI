import argparse
import os
import signal
import sys
import time
import warnings

import neptune
import numpy as np
from navutils.config import load_config
from navutils.logger import Logger
from navutils.user_interrupt import UserInterruptException, signal_handler
from neptune_tensorboard import enable_tensorboard_logging
from pewrapper.misc.version_wrapper_bin import RELEASE_INFO
from rlnav.agent.ppo_agent import create_ppo_agent
from rlnav.env.pe_env import PE_Env
from rlnav.env.wrapper import WrapperDataAttributeError
from rlnav.managers.reward_mgr import RewardManager
from rlnav.managers.wrapper_mgr import WrapperManager
from rlnav.recorder.training_recorder import TrainingRecorder
from rlnav.types.running_metric import RunningMetric
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

warnings.simplefilter(action="ignore", category=FutureWarning)

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

eps = np.finfo(np.float32).eps.item()


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


def run_training_loop(config_path, output_path, parsing_rate, npt_run: neptune.Run):
    enable_tensorboard_logging(npt_run)
    config = load_config(config_path)

    rewardMgr = RewardManager(npt_run)
    wrapperMgr = WrapperManager(
        config.scenarios.path,
        (config.scenarios.skip_first, config.scenarios.n_scenarios),
        config.scenarios.priority,
        config.scenarios.subscenarios,
        output_path,
        npt_run,
        rewardMgr,
        num_generations=config.scenarios.n_generations,
    )
    pe_env = PE_Env(
        wrapperMgr=wrapperMgr,
        rewardMgr=rewardMgr,
        transformers_path=config.transformed_data.path,
        window_size=config.rnn.window_size if config.rnn.enable else 1,
    )
    train_env = tf_py_environment.TFPyEnvironment(pe_env)

    # Crea el agente.
    agent = create_ppo_agent(train_env, npt_run, rnn=config.rnn.enable)

    # Configura el replay buffer y el driver.
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=100,
    )

    driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=100,
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

    policy_dir = os.path.join(output_path, "policy")
    policy_saver = common.Checkpointer(ckpt_dir=policy_dir, policy=agent.collect_policy)

    # Restaurar desde el último checkpoint si existe.
    train_checkpointer.initialize_or_restore()

    early_stopping_threshold = 0.1
    early_stopping_patience = 600
    loss_window = []

    training_tracing_dir = os.path.join(output_path, "Training_Tracing")
    training_recorder = TrainingRecorder(output_path=training_tracing_dir)
    training_recorder.initialize()

    loss = RunningMetric()
    policy_gradient_loss = RunningMetric()
    value_estimation_loss = RunningMetric()
    l2_regularization_loss = RunningMetric()
    entropy_regularization_loss = RunningMetric()
    kl_penalty_loss = RunningMetric()
    clip_fraction = RunningMetric()

    try:
        while wrapperMgr.next_scenario(parsing_rate):
            scenario_priority = (
                wrapperMgr.scenario == config.scenarios.priority
                and wrapperMgr.scenario_generation[wrapperMgr.scenario] == 1
            )

            loss.reset()
            policy_gradient_loss.reset()
            value_estimation_loss.reset()
            l2_regularization_loss.reset()
            entropy_regularization_loss.reset()
            kl_penalty_loss.reset()
            clip_fraction.reset()

            while hasattr(wrapperMgr.ai_wrapper, "wrapper_data"):
                try:
                    driver.run()

                    start = time.time()
                    # Muestreo de datos del buffer y entrenamiento del agente.
                    experience = replay_buffer.gather_all()
                    train_loss = agent.train(experience)

                    times = {}
                    times["agent_train"] = []
                    times["agent_train"].append(time.time() - start)
                    times.update(wrapperMgr.get_times())
                    times.update(pe_env.get_times())
                    times.update(rewardMgr.get_times())
                    if config.neptune.monitoring_times:
                        npt_run["monitoring/times"].extend(remove_empty_lists(times))

                    loss.update(train_loss.loss.numpy())
                    policy_gradient_loss.update(
                        train_loss.extra.policy_gradient_loss.numpy()
                    )
                    value_estimation_loss.update(
                        train_loss.extra.value_estimation_loss.numpy()
                    )
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
                    train_metrics["kl_penalty_loss"] = (
                        kl_penalty_loss.get_running_value()
                    )
                    train_metrics["clip_fraction"] = clip_fraction.get_running_value()
                    training_recorder.record_metrics(train_metrics)
                    npt_run["training/train"].extend(
                        {k: [v] for k, v in train_metrics.items()}
                    )

                    replay_buffer.clear()

                    if agent.train_step_counter.numpy() % 10 == 0:
                        rewardMgr.update_map()

                    # Guardar un checkpoint.
                    if agent.train_step_counter.numpy() % 1000 == 0:
                        train_checkpointer.save(agent.train_step_counter.numpy())
                        if npt_run.exists(f"training/agent/checkpoint"):
                            del npt_run[f"training/agent/checkpoint"]
                        for checkpoint_file in os.listdir(checkpoint_dir):
                            npt_run[
                                f"training/agent/checkpoint/{checkpoint_file}"
                            ].upload(os.path.join(checkpoint_dir, checkpoint_file))

                    if not scenario_priority:
                        loss_window.append(loss.get_running_value())
                        if len(loss_window) > early_stopping_patience:
                            loss_window.pop(0)
                        if (
                            len(loss_window) == early_stopping_patience
                            and max(loss_window) - min(loss_window)
                            < early_stopping_threshold
                        ):
                            Logger.log_message(
                                Logger.Category.INFO,
                                Logger.Module.MAIN,
                                f"Early stopping for scenario '{wrapperMgr.scenario}' generation '{wrapperMgr.scenario_generation[wrapperMgr.scenario]}' due to minimal loss improvement.",
                            )
                            break

                except WrapperDataAttributeError:
                    pass

            map_file = rewardMgr.update_map()
            npt_run[
                f"training/{wrapperMgr.scenario}/AI_generation{wrapperMgr.scenario_generation[wrapperMgr.scenario]}/map"
            ].upload(map_file)
            npt_run[
                f"training/{wrapperMgr.scenario}/AI_generation{wrapperMgr.scenario_generation[wrapperMgr.scenario]}/reward"
            ].upload(wrapperMgr.reward_rec.file_path)

            train_checkpointer.save(agent.train_step_counter.numpy())
            for checkpoint_file in os.listdir(checkpoint_dir):
                npt_run[f"training/agent/checkpoint/{checkpoint_file}"].upload(
                    os.path.join(checkpoint_dir, checkpoint_file)
                )

        policy_saver.save(agent.train_step_counter.numpy())
        for policy_file in os.listdir(policy_dir):
            npt_run[f"training/agent/policy/{policy_file}"].upload(
                os.path.join(policy_dir, policy_file)
            )

        wrapperMgr.close_PE()
        training_recorder.close()

        return EXIT_SUCCESS

    except UserInterruptException as exception:
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"{exception.get_msg()}",
        )

        wrapperMgr.close_PE()

        return EXIT_FAILURE


def main(argv):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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

    args = parser.parse_args(argv[1:])
    args.output_path = os.path.join(args.output_path)

    run = neptune.init_run(
        project="AI-PE/RL-GSharp",
        monitoring_namespace="monitoring/03_agent_training",
        source_files=["config/RLNav/params.yaml", "src/rlnav/**/*.py"],
    )

    Logger(args.output_path, run, use_dual_logging=True)

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
    result = main(sys.argv)

    sys.exit(result)
