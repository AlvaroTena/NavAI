import argparse
import os
import sys
import signal
import logging
from tf_agents.environments import tf_py_environment

from rlnav.env.pe_env import PE_Env
from dotenv import load_dotenv
import neptune
import tensorflow as tf
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

from rlnav.ppo_train import create_parallel_environment, load_scenarios_list
from navutils.config import load_config
from rlnav.env.pe_env import (
    get_transformers_min_max,
    create_observation_spec,
    create_action_spec,
    create_reward_spec,
)
from navutils.user_interrupt import UserInterruptException, signal_handler
from rlnav.agent.ppo_agent import create_ppo_agent
from rlnav.managers.wrapper_mgr import WrapperManager
from rlnav.managers.reward_mgr import RewardManager
from navutils.logger import Logger

EXIT_SUCCESS = 0
EXIT_FAILURE = 1

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluación de políticas PPO en escenarios base"
    )
    parser.add_argument(
        "-c",
        "--config_file",
        dest="config_file",
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-p",
        "--policy_path",
        dest="policy_path",
        required=True,
        help="Path to the policy directory",
    )
    parser.add_argument(
        "-o",
        "--output_directory",
        dest="output_path",
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "-g",
        "--debug_level",
        dest="debug_level",
        default="INFO",
        help="Logging level (DEBUG/INFO/WARNING/ERROR)",
    )
    parser.add_argument(
        "-p",
        "--parsing_rate",
        dest="parsing_rate",
        default=0,
        type=int,
        help="Parsing rate for the wrapper (optional)",
    )
    return parser.parse_args()


def main():
    parser = create_parser()
    args = parser.parse_args()

    Logger(args.output_path)

    if not args.policy_path or not args.output_path or not args.debug_level:
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.MAIN,
            f"Not enough parameters. The command line shall be: {sys.argv[0]} --config_file --output_directory --debug_level(INFO/WARNING/ERROR/DEBUG/TRACE) --parsing_rate(optional)",
        )
        parser.print_help()
        return EXIT_FAILURE

    Logger.set_category(args.debug_level)

    load_dotenv(override=True)
    config = load_config(args.config_file)

    min_values, max_values = get_transformers_min_max(
        os.path.join(config.transformed_data.path, "transforms")
    )

    run_id = os.getenv("NEPTUNE_CUSTOM_RUN_ID", None)
    npt_run = neptune.init_run(
        project="AI-PE/RL-GSharp",
        custom_run_id=run_id,
        name="ppo_evaluation",
        monitoring_namespace="monitoring/04_agent_evaluation",
        source_files=[args.config_file, "src/rlnav/**/*.py"],
        api_token=os.getenv("NEPTUNE_API_TOKEN"),
    )

    scenarios = load_scenarios_list(
        config.scenarios.path,
        config.scenarios.skip_first,
        config.scenarios.n_scenarios,
        config.scenarios.priority,
        [],
        curriculum_learning=False,
    )

    reward_mgr = RewardManager()
    wrapper_mgr = WrapperManager(
        scenarios,
        config.scenarios.path,
        args.output_path,
        reward_mgr,
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

    policy_dir = os.path.normpath(args.policy_path)
    if not os.path.exists(policy_dir):
        Logger.log_message(
            Logger.Category.ERROR,
            Logger.Module.MAIN,
            f"Policy path {policy_dir} does not exist.",
        )
        return EXIT_FAILURE

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=policy_dir,
        policy=agent.policy,
    )
    policy_checkpointer.initialize_or_restore()

    while wrapper_mgr.next_scenario(parsing_rate=args.parsing_rate):
        current = wrapper_mgr.scenario
        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"Scenario {current} eval started",
        )

        eval_env = PE_Env(
            configMgr=wrapper_mgr.configMgr,
            wrapper_data=wrapper_mgr.wrapper_data,
            rewardMgr=wrapper_mgr.rewardMgr,
            scenario=wrapper_mgr.scenario,
            num_generations=config.scenarios.n_generations,
            min_values=min_values,
            max_values=max_values,
            transformers_path=config.transformed_data.path,
            window_size=(
                config.training.rnn.window_size if config.training.rnn.enable else 1
            ),
            output_path=wrapper_mgr.output_path,
            name="eval_env",
        )
        eval_env = tf_py_environment.TFPyEnvironment(eval_env)

        time_step = eval_env.reset()
        policy_state = agent.policy.get_initial_state(batch_size=eval_env.batch_size)

        while not eval_env.is_done():
            policy_step = agent.policy.action(time_step, policy_state)
            time_step = eval_env.step(policy_step.action)
            policy_state = policy_step.state

        map_file = eval_env.update_map()
        reward_file = eval_env.get_reward_filepath()
        if npt_run._mode != "debug":
            npt_run[f"eval/{current}/map"].upload(map_file)
            npt_run[f"eval/{current}/reward"].upload(reward_file)

        Logger.log_message(
            Logger.Category.INFO,
            Logger.Module.MAIN,
            f"Scenario {current} eval finished",
        )

    eval_env.close()
    npt_run.stop()

    Logger.log_message(
        Logger.Category.INFO,
        Logger.Module.MAIN,
        "Evaluation finished. Exiting.",
    )
    return EXIT_SUCCESS


if __name__ == "__main__":
    main()
