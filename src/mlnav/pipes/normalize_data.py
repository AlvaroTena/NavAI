import argparse
import datetime as dt
import shutil
from os import makedirs, path, remove
from time import time
from typing import Text

import joblib
import neptune
import numpy as np
import pandas as pd
from neptune.utils import stringify_unsupported
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from mlnav.features.normalization import normalize_column
from mlnav.utils.logger import Logger
from navutils.config import load_config


def normalize(config_path: Text, npt_run: neptune.Run) -> None:
    """Normalize processed data.

    Args:
        config_path (Text): path to config
    """

    st = time()

    config = load_config(config_path)

    processed_filename = path.join(
        config.base_path, config.process.path, config.process.name
    )
    normalized_filedir = path.join(config.base_path, config.normalize.path)
    normalized_filename = path.join(normalized_filedir, config.normalize.name)
    columntransformers_filedir = path.join(
        config.base_path, config.normalize.transformers_path
    )

    if path.isfile(normalized_filename) or path.islink(normalized_filename):
        remove(normalized_filename)
    if path.isdir(columntransformers_filedir):
        shutil.rmtree(columntransformers_filedir)

    makedirs(normalized_filedir, exist_ok=True)
    makedirs(columntransformers_filedir, exist_ok=True)

    norm_df = pd.DataFrame()

    cons_categories = ["E", "G", "B"]
    sat_id_categories = [str(i).zfill(2) for i in range(1, 64)]
    freq_categories = [
        "FREQ_L1_GPS",
        "FREQ_L2_GPS",
        "FREQ_L5_GPS",
        "FREQ_E1_GAL",
        "FREQ_E5A_GAL",
        "FREQ_E5B_GAL",
        "FREQ_E5AB_GAL",
        "FREQ_E6_GAL",
        "FREQ_B1I_BDS",
        "FREQ_B1C_BDS",
        "FREQ_B1A_BDS",
        "FREQ_B2A_BDS",
        "FREQ_B3I_BDS",
    ]
    # List of tuples (column name, transformer, columns expected) with:
    #   + column name: Name to identify the transformer
    #   + transformer: estimator to apply to each column
    #   + columns: list of names of the columns to apply each transformer
    transformers = [
        ("epoch", "drop", ["epoch"]),
        (
            "cons",
            "drop",
            # OneHotEncoder(
            #     categories=[cons_categories],
            #     dtype=np.uint8,
            #     handle_unknown="ignore",
            #     sparse_output=False,
            # ),
            cons_categories,
        ),
        (
            "sat_id",
            "drop",
            # OneHotEncoder(
            #     categories=[sat_id_categories],
            #     dtype=np.uint8,
            #     handle_unknown="ignore",
            #     sparse_output=False,
            # ),
            sat_id_categories,
        ),
        (
            "freq",
            "drop",
            # OneHotEncoder(
            #     categories=[freq_categories],
            #     dtype=np.uint8,
            #     handle_unknown="ignore",
            #     sparse_output=False,
            # ),
            freq_categories,
        ),
        ("code", "drop", ["code"]),  # RobustScaler()
        ("phase", "drop", ["phase"]),  # RobustScaler()
        ("doppler", "drop", ["doppler"]),  # RobustScaler()
        ("snr", RobustScaler(), ["snr"]),
        ("elevation", StandardScaler(), ["elevation"]),
        ("residual", MinMaxScaler((-1, 1)), ["residual"]),
        ("iono", MinMaxScaler((-1, 1)), ["iono"]),
        ("delta_cmc", MinMaxScaler((-1, 1)), ["delta_cmc"]),
        ("code_rate_cons", MinMaxScaler((-1, 1)), ["code_rate_cons"]),
    ]
    npt_run["transformers/list"] = stringify_unsupported(transformers)
    transformers = [
        transformer for transformer in transformers if transformer[1] != "drop"
    ]

    for transformer in transformers:
        col = pd.read_hdf(processed_filename, key="data", columns=[transformer[0]])
        col, ct = normalize_column(column=col, transformer=transformer)

        with open(
            path.join(
                columntransformers_filedir, f"{transformer[0]}_ColumnTransformer.pkl"
            ),
            "wb",
        ) as ct_filename:
            joblib.dump(ct, ct_filename)
            npt_run[f"transformers/{transformer[0]}"].upload(ct_filename.name)
            Logger.getLogger().debug(f"ColumnTransformer saved to: {ct_filename.name}")

        if not col.empty:
            norm_df = pd.concat([norm_df, col], axis=1)
            norm_df.to_hdf(
                normalized_filename, key="data", mode="w", format="table", index=False
            )
            Logger.getLogger().debug(
                f'Normalized data from "{transformer[0]}" column saved in: {normalized_filename}'
            )

    try:
        col = pd.read_hdf(processed_filename, key="data", columns=["multipath"])
        col.reset_index(drop=True, inplace=True)
        norm_df = pd.concat([norm_df, col], axis=1)
        norm_df.to_hdf(
            normalized_filename, key="data", mode="w", format="table", index=False
        )
        npt_run["dataset/metadata"].track_files(normalized_filename)
    except:
        Logger.getLogger().debug("Data does not have multipath column")

    Logger.getLogger().info(f"Normalized data saved in: {normalized_filename}")

    et = time()
    Logger.getLogger().info(
        f"Normalizing stage has taken {str(dt.timedelta(seconds=et-st))}"
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
        monitoring_namespace="monitoring/data_normalize",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    normalize(config_path=args.config, npt_run=run["data_normalize"])
    run.stop()
