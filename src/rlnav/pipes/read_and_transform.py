import argparse
import os

import neptune

from navutils.config import load_config
from navutils.logger import Logger
from rlnav.data.transform import read_and_transform

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
        monitoring_namespace="monitoring/02_data_transforming",
    )

    Logger(args.output, run).set_category(args.loglevel)

    base_config = load_config(args.config)

    os.makedirs(
        (
            output_dir := os.path.join(
                base_config.base_path, base_config.transformed_data.path
            )
        ),
        exist_ok=True,
    )

    ct = read_and_transform(
        input_path=os.path.join(base_config.base_path, base_config.processed_data.path),
        output_dir=output_dir,
        npt_run=run,
    )

    run.stop()
