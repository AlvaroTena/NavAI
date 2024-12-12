import argparse
import datetime as dt
import sys
import warnings
from os import environ, makedirs, path, remove, rename
from time import time
from typing import Text

import joblib
import neptune
import pandas as pd
import tqdm
from hyperopt import hp
from neptune.utils import stringify_unsupported
from sklearn.cluster import KMeans

from mlnav.train.train import train_model
from mlnav.utils.logger import Logger
from navutils.config import load_config

warnings.filterwarnings("ignore", category=FutureWarning)


def save_model(model, model_filename):
    try:
        with open(model_filename, "wb") as model_file:
            joblib.dump(model, model_file)
    except IOError as e:
        if e.errno == 13:
            Logger.getLogger().warning(f"{e}. Creating backup...")
            rename(model_filename, f"{model_filename}.bkp")
            with open(model_filename, "wb") as model_file:
                joblib.dump(model, model_file)
        else:
            Logger.getLogger().error(e)


def training_model(config_path: Text, npt_run: neptune.Run) -> None:
    """Train model with normalized data.

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
    model_filedir = path.join(config.base_path, config.model.path)
    model_filename = path.join(model_filedir, config.model.name)
    trials_filedir = path.join(model_filedir, config.model.trials.path)
    trials_filename = path.join(trials_filedir, config.model.trials.name)

    if path.isfile(model_filedir) or path.islink(model_filedir):
        remove(model_filedir)

    makedirs(model_filedir, exist_ok=True)
    makedirs(trials_filedir, exist_ok=True)

    if config.model.train.chunksize and config.model.train.hyperparameter_tunning:
        Logger.getLogger().error(
            "Training by chunks is NOT compatible with hyperparameter tunning"
        )
        return
    elif config.model.train.chunksize:
        kmeans = KMeans(
            n_clusters=config.model.train.param_grid.n_clusters[0],
            init=config.model.train.param_grid.init[0],
            n_init=config.model.train.param_grid.n_init[0],
            max_iter=config.model.train.param_grid.max_iter[0],
            tol=config.model.train.param_grid.tol[0],
            verbose=config.model.train.param_grid.verbose[0],
            random_state=config.model.train.param_grid.random_state[0],
            copy_x=config.model.train.param_grid.copy_x[0],
            algorithm=config.model.train.param_grid.algorithm[0],
        )

        with tqdm.tqdm(
            range(
                pd.HDFStore(path=normalized_filename, mode="r").get_storer("data").nrows
            )
        ) as pbar:
            for idx, chunk in enumerate(
                pd.read_hdf(
                    normalized_filename,
                    key="data",
                    chunksize=config.model.train.chunksize,
                )
            ):
                kmeans = train_model(
                    model=kmeans,
                    data=(
                        chunk.drop(columns=["multipath"])
                        if "multipath" in chunk
                        else chunk
                    ),
                    hyperparams=None,
                    hyperopt=False,
                )
                save_model(model=kmeans, model_filename=model_filename)
                Logger.getLogger().info(
                    f"Trained KMeans model with chunk {idx} and {kmeans.inertia_} inertia saved in: {model_filename}"
                )
                pbar.update(config.model.train.chunksize)
    elif config.model.train.hyperparameter_tunning:
        # Space
        params = {
            "n_clusters": hp.choice(
                "n_clusters", config.model.train.param_grid.n_clusters
            ),
            "init": hp.choice("init", config.model.train.param_grid.init),
            "n_init": hp.choice("n_init", config.model.train.param_grid.n_init),
            "max_iter": hp.choice("max_iter", config.model.train.param_grid.max_iter),
            "tol": hp.uniform(
                "tol",
                min(config.model.train.param_grid.tol),
                max(config.model.train.param_grid.tol),
            ),
            "verbose": hp.choice("verbose", config.model.train.param_grid.verbose),
            "random_state": hp.choice(
                "random_state", config.model.train.param_grid.random_state
            ),
            "copy_x": hp.choice("copy_x", config.model.train.param_grid.copy_x),
            "algorithm": hp.choice(
                "algorithm", config.model.train.param_grid.algorithm
            ),
        }

        data = pd.read_hdf(normalized_filename, key="data")
        kmeans = train_model(
            model=KMeans(),
            data=data.drop(columns=["multipath"]) if "multipath" in data else data,
            hyperparams=params,
            hyperopt=config.model.train.hyperparameter_tunning,
            hyperopt_iters=config.model.train.hyperparameter_tunning_iters,
            trials_save_file=trials_filename,
        )
        save_model(model=kmeans, model_filename=model_filename)
    else:
        data = pd.read_hdf(normalized_filename, key="data")
        kmeans = train_model(
            model=KMeans(
                n_clusters=config.model.train.param_grid.n_clusters[0],
                init=config.model.train.param_grid.init[0],
                n_init=config.model.train.param_grid.n_init[0],
                max_iter=config.model.train.param_grid.max_iter[0],
                tol=config.model.train.param_grid.tol[0],
                verbose=config.model.train.param_grid.verbose[0],
                random_state=config.model.train.param_grid.random_state[0],
                copy_x=config.model.train.param_grid.copy_x[0],
                algorithm=config.model.train.param_grid.algorithm[0],
            ),
            data=data.drop(columns=["multipath"]) if "multipath" in data else data,
            hyperparams=None,
            hyperopt=False,
        )
        save_model(model=kmeans, model_filename=model_filename)

    npt_run["model/kmeans/parameters"] = kmeans.get_params()
    npt_run["model/kmeans/inertia"] = kmeans.inertia_
    npt_run["model/kmeans/bin"].upload(model_filename)

    Logger.getLogger().info(
        f"Final KMeans model with {kmeans.inertia_} inertia saved in: {model_filename}"
    )

    et = time()
    Logger.getLogger().info(
        f"Training stage has taken {str(dt.timedelta(seconds=et-st))}"
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
        monitoring_namespace="monitoring/training",
    )
    logger = Logger(
        level=args.loglevel.upper(),
        npt_run=run,
        stdout=True,
        filename=None,
        handle_prints=False,
    )

    training_model(config_path=args.config, npt_run=run["training"])
    run.stop()
