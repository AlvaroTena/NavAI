import argparse
import datetime as dt
import shutil
import warnings
from os import makedirs, path
from time import time

import neptune
import pandas as pd

from mlnav.report.plot_distributions import plot_pairplots
from mlnav.report.plot_features import plot_features
from mlnav.utils.logger import Logger
from navutils.config import load_config

warnings.filterwarnings("ignore", category=DeprecationWarning)


def model_reports(config_path: str, npt_run: neptune.Run) -> None:
    """Predict and save data.

    Args:
        config_path (str): path to config
    """
    st = time()

    config = load_config(config_path)

    processed_filename = path.join(
        config.base_path, config.process.path, config.process.name
    )
    # normalized_filename = path.join(
    #     config.base_path, config.normalize.path, config.normalize.name
    # )
    # model_filename = path.join(config.base_path, config.model.path, config.model.name)

    reports_filedir = path.join(config.base_path, config.reports.path)
    predictions_filename = path.join(reports_filedir, config.reports.predictions.name)
    plots_filesdir = path.join(reports_filedir, config.reports.model_plots.path)

    makedirs(reports_filedir, exist_ok=True)
    makedirs(plots_filesdir, exist_ok=True)

    data = pd.read_hdf(processed_filename, key="data", index_col=0)

    pred = pd.read_hdf(predictions_filename, key="data", mode="r")
    data.loc[data["epoch"] == pred["epoch"], "predictions"] = pred["predictions"]
    del pred

    cluster_distribution, cluster_pairplots, pairplots2d, pairplots3d = plot_pairplots(
        data,
        labels=data["predictions"],
        sample_size=10**6,
        drop_noise=True,
        plot_cluster_pairplots=config.reports.model_plots.cluster_pairplots,
        plot_pairplots2d=config.reports.model_plots.pairplots2d,
        plot_pairplots3d=config.reports.model_plots.pairplots3d,
    )

    npt_run["model/cluster_distribution"].upload(cluster_distribution)
    cluster_distribution.savefig(
        path.join(plots_filesdir, "cluster_distribution.png"),
        format="png",
        dpi=300,
        bbox_inches="tight",
    )
    del cluster_distribution

    if cluster_pairplots:
        for cluster, pairplot in enumerate(cluster_pairplots):
            npt_run[f"model/pairplot_cluster-{cluster}"].upload(pairplot)
            pairplot.savefig(
                path.join(plots_filesdir, f"pairplot_cluster{cluster}.png"),
                format="png",
                dpi=300,
                bbox_inches="tight",
            )
            del pairplot
    if pairplots2d:
        npt_run["model/pairplots_2d"].upload(pairplots2d)
        pairplots2d.savefig(
            path.join(plots_filesdir, "pairplots2d.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        del pairplots2d
    if pairplots3d:
        npt_run["model/pairplots_3d"].upload(pairplots3d)
        pairplots3d.savefig(
            path.join(plots_filesdir, "pairplots3d.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        del pairplots3d

    et = time()
    Logger.getLogger().info(
        f"Report stage has taken {str(dt.timedelta(seconds=et-st))}"
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
        monitoring_namespace="monitoring/model_reports",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    model_reports(config_path=args.config, npt_run=run["reports"])
    run.stop()
