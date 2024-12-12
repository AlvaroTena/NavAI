import argparse
import datetime as dt
import json
import shutil
from os import environ, makedirs, path, remove
from time import time
from typing import Text

import joblib
import neptune
import pandas as pd

from mlnav.evaluate.metrics import Unsupervised_Metrics
from mlnav.utils.logger import Logger
from navutils.config import load_config


def evaluate_model(config_path: Text, npt_run: neptune.Run) -> None:
    """Evaluate model with unsupervised metrics.

    Args:
        config_path (Text): path to config
    """
    environ["OPENBLAS_NUM_THREADS"] = "64"
    environ["NUM_THREADS"] = "64"
    environ["OMP_NUM_THREADS"] = "64"

    st = time()

    config = load_config(config_path)

    normalized_filename = path.join(
        config.base_path, config.normalize.path, config.normalize.name
    )
    model_filename = path.join(config.base_path, config.model.path, config.model.name)
    reports_filedir = path.join(config.base_path, config.reports.path)
    metrics_filename = path.join(
        reports_filedir, config.reports.evaluation.unsupervised_metrics.name
    )
    # plots_filedir = path.join(config.base_path, config.reports.plots.path)

    if path.isfile(metrics_filename) or path.islink(metrics_filename):
        remove(metrics_filename)
    # if path.isdir(plots_filedir):
    #     shutil.rmtree(plots_filedir)

    makedirs(reports_filedir, exist_ok=True)
    # makedirs(plots_filedir, exist_ok=True)

    data = pd.read_hdf(normalized_filename, index_col=0)
    model = joblib.load(model_filename)

    metrics = Unsupervised_Metrics(
        model=model,
        data=data.drop(columns=["multipath"]) if "multipath" in data else data,
    ).get_metrics(
        compute_ch=config.reports.evaluation.unsupervised_metrics.calinski_harabasz,
        compute_dbs=config.reports.evaluation.unsupervised_metrics.davies_bouldin_score,
        compute_sil=config.reports.evaluation.unsupervised_metrics.silhouette,
        sil_split=config.reports.evaluation.unsupervised_metrics.silhouette_split,
        sil_show=config.reports.evaluation.unsupervised_metrics.silhouette_show,
        sil_save=config.reports.evaluation.unsupervised_metrics.silhouette_show,
        save_path=config.reports.path,
    )
    npt_run["metrics"] = metrics
    with open(metrics_filename, "w") as metrics_file:
        json.dump(metrics, metrics_file, indent=4)

    Logger.getLogger().info(f"Metrics saved to: {metrics_filename}")

    et = time()
    Logger.getLogger().info(
        f"Evaluating stage has taken {str(dt.timedelta(seconds=et-st))}"
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
        monitoring_namespace="monitoring/model_evaluation",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    evaluate_model(config_path=args.config, npt_run=run["model_evaluation"])
    run.stop()
