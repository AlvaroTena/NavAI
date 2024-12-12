import argparse
import datetime as dt
from os import makedirs, path
from time import time
from typing import Text

import joblib
import neptune
import pandas as pd

from mlnav.report.prediction import Predictor
from mlnav.utils.logger import Logger
from navutils.config import load_config


def predict_data(config_path: Text, npt_run: neptune.Run) -> None:
    """Predict and save data.

    Args:
        config_path (Text): path to config
    """
    st = time()

    config = load_config(config_path)

    processed_filename = path.join(
        config.base_path, config.process.path, config.process.name
    )
    normalized_filename = path.join(
        config.base_path, config.normalize.path, config.normalize.name
    )
    model_filename = path.join(config.base_path, config.model.path, config.model.name)

    reports_filedir = path.join(config.base_path, config.reports.path)
    predictions_filename = path.join(reports_filedir, config.reports.predictions.name)

    makedirs(reports_filedir, exist_ok=True)

    predictor = Predictor(model=joblib.load(model_filename))
    predictor.get_predictions(
        prediction_data=pd.read_hdf(normalized_filename, index_col=0),
        # normalized=False,
        # transformers_dir=path.join(
        #     config.base_path, config.normalize.transformers_path
        # ),
    )
    predictions = predictor.map_predictions(
        mapping_data=pd.read_hdf(processed_filename, index_col=0)
    )

    predictions.to_hdf(
        predictions_filename, key="data", mode="w", format="table", index=False
    )
    npt_run["metadata"].track_files(predictions_filename)

    Logger.getLogger().info(f"Prediction data saved to: {predictions_filename}")

    et = time()
    Logger.getLogger().info(
        f"Predictor stage has taken {str(dt.timedelta(seconds=et-st))}"
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
        monitoring_namespace="monitoring/data_prediction",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    predict_data(config_path=args.config, npt_run=run["data_prediction"])
    run.stop()
