import argparse
import datetime as dt
from os import makedirs, path, remove
from time import time

import neptune
import pandas as pd

from mlnav.data.reader import Reader
from mlnav.features.prepro import preprocess_df
from mlnav.utils.logger import Logger
from navutils.config import load_config


def preprocess(config_path: str, npt_run: neptune.Run) -> None:
    """Preprocess raw data into one DataFrame.

    Args:
        config_path (str): path to config
    """

    st = time()

    config = load_config(config_path)

    processed_filedir = path.join(config.base_path, config.process.path)
    processed_filename = path.join(processed_filedir, config.process.name)

    if path.isfile(processed_filename) or path.islink(processed_filename):
        remove(processed_filename)

    makedirs(processed_filedir, exist_ok=True)

    data_dir = path.join(config.base_path, config.reader.path)
    npt_run["dataset/GSHARP2.2.2"].track_files(data_dir)

    dfs = Reader().perform_reading(
        data_dir=data_dir,
        n_files=config.reader.n_files,
    )
    first_epoch: dt.datetime = dfs[0]["epoch"].iloc[0]
    last_epoch: dt.datetime = dfs[-1]["epoch"].iloc[-1]

    # Create empty DataFrame for saving df
    pd.DataFrame(
        columns=["epoch", "cons", "sat_id"]
        + dfs[0].columns.tolist()[2:]
        + ["cmc", "code_rate_cons"]
    ).to_hdf(processed_filename, key="data", mode="w", format="table", index=False)

    delta_t = dt.timedelta()
    for df in dfs:
        delta, data = preprocess_df(df=df, drop_nan_rows=True, drop_outliers=False)
        if data.empty:
            continue

        data.to_hdf(
            processed_filename,
            key="data",
            mode="r+",
            append=True,
            format="table",
            index=False,
        )
        delta_t += delta

        df_first_epoch: dt.datetime = df["epoch"].iloc[0]
        df_last_epoch: dt.datetime = df["epoch"].iloc[-1]
        Logger.getLogger().debug(
            f"Processed data from {df_first_epoch} to {df_last_epoch} saved in: {processed_filename}"
        )

    npt_run["dataset/metadata"].track_files(processed_filename)
    npt_run["dataset/n_files"] = config.reader.n_files

    Logger.getLogger().info(
        f'Processed data from {first_epoch.strftime("%Y-%m-%d %H:%M:%S")} to {last_epoch.strftime("%Y-%m-%d %H:%M:%S")}, a total of {delta_t} saved in: {processed_filename}'
    )

    et = time()
    Logger.getLogger().info(
        f"Processing stage has taken {str(dt.timedelta(seconds=et-st))}"
    )


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("-c", "--config", dest="config", required=True)
    args_parser.add_argument(
        "-g",
        "--loglevel",
        dest="loglevel",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
    )
    args = args_parser.parse_args()

    run = neptune.init_run(
        project="AI-PE/ML-GSharp",
        monitoring_namespace="monitoring/data_processing",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    preprocess(config_path=args.config, npt_run=run["data_processing"])
    run.stop()
