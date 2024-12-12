import argparse
import datetime as dt
import shutil
from os import makedirs, path
from time import time

import neptune
import pandas as pd

from mlnav.report.plot_distributions import plot_pairplots
from mlnav.report.plot_features import plot_features
from mlnav.utils.logger import Logger
from navutils.config import load_config


def data_reports(config_path: str, npt_run: neptune.Run) -> None:
    st = time()

    config = load_config(config_path)

    processed_filename = path.join(
        config.base_path, config.process.path, config.process.name
    )

    reports_filedir = path.join(config.base_path, config.reports.path)
    plots_filesdir = path.join(reports_filedir, config.reports.data_plots.path)

    makedirs(reports_filedir, exist_ok=True)
    makedirs(plots_filesdir, exist_ok=True)

    data = pd.read_hdf(processed_filename, key="data", index_col=0)

    columns = [
        "code",
        "phase",
        "doppler",
        "snr",
        "elevation",
        "residual",
        "iono",
        "delta_cmc",
        "code_rate_cons",
    ]
    constellation_plot, box_plot, density_plot = plot_features(
        df=data,
        columns=columns,
        constellation_plot=config.reports.data_plots.constellation_plot,
        box_plot=config.reports.data_plots.box_plot,
        density_plot=config.reports.data_plots.density_plot,
    )

    if constellation_plot:
        npt_run["dataset/processed/constelation_plot"].upload(constellation_plot)
        constellation_plot.savefig(
            path.join(plots_filesdir, "constellation_hist.png"),
            format="png",
            # dpi=300,
            bbox_inches="tight",
        )
        del constellation_plot
    if box_plot:
        npt_run["dataset/processed/box_plot"].upload(box_plot)
        box_plot.savefig(
            path.join(plots_filesdir, "box_plot.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        del box_plot
    if density_plot:
        npt_run["dataset/processed/density_plot"].upload(density_plot)
        density_plot.savefig(
            path.join(plots_filesdir, "density_plot.png"),
            format="png",
            dpi=300,
            bbox_inches="tight",
        )
        del density_plot

    et = time()
    Logger.getLogger().info(
        f"Data report stage has taken {str(dt.timedelta(seconds=et-st))}"
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
        monitoring_namespace="monitoring/data_reports",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    data_reports(config_path=args.config, npt_run=run["reports"])
    run.stop()
