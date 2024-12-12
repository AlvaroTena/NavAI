import datetime as dt
from time import time

import pandas as pd

from mlnav.train.tuning import Model_Tuning
from mlnav.utils.logger import Logger


def train_model(
    model,
    data: pd.DataFrame,
    hyperparams,
    hyperopt: bool = True,
    hyperopt_iters: int = 64,
    trials_save_file="",
):
    """Trains a kmeans model using config hiperparameters and data provided"""
    # get the start time
    st = time()

    if hyperopt:
        Logger.getLogger().debug("Start hyperparameters optimization...")
        model_params = Model_Tuning(
            model=model, param_grid=hyperparams, verbose=2
        ).bayesianOptimizer(
            data=data, iters=hyperopt_iters, trials_save_file=trials_save_file
        )
        Logger.getLogger().debug(f"Best hyperparameters found: {model_params}")
        model = type(model)(**model_params)

    # train
    Logger.getLogger().debug(f"Start training with {len(data.index)} rows...")
    model.fit(data)
    # get the end time
    et = time()
    Logger.getLogger().info(
        f"Training finished in {str(dt.timedelta(seconds=(et-st)))}"
    )

    return model
