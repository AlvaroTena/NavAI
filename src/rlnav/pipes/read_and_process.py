import argparse
import glob
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

import neptune
from navutils.config import load_config
from navutils.logger import Logger
from neptune.utils import stringify_unsupported
from pewrapper.managers import ConfigurationManager
from pewrapper.misc.parser_utils import parse_session_file
from rlnav.data.reader import Reader


def p_read_and_process(data_filepath, config_signals):
    basename = os.path.basename(data_filepath)
    file_dst = os.path.join(
        base_config.base_path,
        base_config.processed_data.path,
        f"{os.path.splitext(basename)[0]}.parquet",
    )

    df = Reader.read_and_process(
        base_config.scenarios.path, data_filepath, config_signals
    )

    df.to_parquet(
        file_dst,
        compression="gzip",
        index=True,
        coerce_timestamps="us",
        allow_truncated_timestamps=True,
    )
    return basename, file_dst


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", required=True)
    args_parser.add_argument("-o", "--output", required=True)
    args_parser.add_argument(
        "-g",
        "--loglevel",
        dest="loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = args_parser.parse_args()

    run = neptune.init_run(
        project="AI-PE/RL-GSharp",
        monitoring_namespace="monitoring/01_data_processing",
    )

    Logger(args.output, run).set_category(args.loglevel)

    base_config = load_config(args.config)

    configMgr = ConfigurationManager(args.output, "")

    Logger.log_message(
        Logger.Category.DEBUG, Logger.Module.MAIN, f"Collect all scenarios information"
    )

    raw_data_path = os.path.join(base_config.base_path, base_config.processed_data.path)
    if os.path.exists(raw_data_path):
        Logger.log_message(
            Logger.Category.DEBUG, Logger.Module.MAIN, f"Cleaning dir {raw_data_path}"
        )
        shutil.rmtree(raw_data_path)
    os.makedirs(
        raw_data_path,
        exist_ok=True,
    )

    run["raw_data/num_scenarios"] = base_config.scenarios.n_scenarios

    scenarios = []
    for num, scen in enumerate(
        sorted(glob.glob(os.path.join(base_config.scenarios.path, "scenario_*/"))),
        start=1,
    ):
        if (
            base_config.scenarios.n_scenarios > 0
            and num > base_config.scenarios.n_scenarios
        ):
            break

        run["raw_data/metadata"].track_files(scen)

        session_file = glob.glob(scen + "CONFIG/session.ini")[0]
        result = parse_session_file(session_file, verbose=False)
        config_file = result[1] if result[0] else None

        config_parsed = configMgr.parse_config_file(config_file, verbose=False)[0]

        scenario_info = {
            "scen_dir": scen.replace(base_config.scenarios.path, "./"),
            "session": session_file.replace(base_config.scenarios.path, "./"),
            "config": config_file.replace(base_config.scenarios.path, "./"),
            "dataset": glob.glob(scen + "**/AI_Multipath_*", recursive=True)[0].replace(
                base_config.scenarios.path, "./"
            ),
            "config_signals": [
                [
                    configMgr.config_info_.Signal_1_GPS,
                    configMgr.config_info_.Signal_2_GPS,
                ],
                [
                    configMgr.config_info_.Signal_1_GAL,
                    configMgr.config_info_.Signal_2_GAL,
                ],
                [
                    configMgr.config_info_.Signal_1_BDS,
                    configMgr.config_info_.Signal_2_BDS,
                ],
            ],
        }

        run[f"raw_data/scenarios/{scen.split('/')[-2]}"] = stringify_unsupported(
            scenario_info
        )

        scenarios.append(scenario_info)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(
                p_read_and_process, scenario["dataset"], scenario["config_signals"]
            )
            for scenario in scenarios
        ]

        for future in as_completed(futures):
            basename, file_dst = future.result()
            run["processed_data/metadata"].track_files(file_dst)
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.MAIN,
                f"Saving dataset {basename} to {file_dst}",
            )

    run.stop()
