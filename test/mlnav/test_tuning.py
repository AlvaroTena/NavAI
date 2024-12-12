import numpy as np
import pandas as pd
import pytest
from hyperopt import hp
from sklearn.cluster import KMeans

from mlnav.train.train import train_model


@pytest.fixture(scope="module")
def create_model():
    """Returns a KMeans model"""
    return KMeans(3)


@pytest.fixture(scope="module")
def get_data():
    """Returns a numpy array with data to train"""
    data = pd.read_csv("Files_Test/tuning_tests/test_scenario.csv")
    return data


def test_simple_tuning_kmeans(create_model, get_data):
    """No model file created"""
    params = {
        "n_clusters": hp.choice("n_clusters", [2, 3]),
        "init": hp.choice("init", ["k-means++"]),
        "n_init": hp.choice("n_init", ["auto"]),
        "max_iter": hp.choice("max_iter", [400]),
        "tol": hp.loguniform(
            "tol",
            min([0.001, 0.01]),
            max([0.001, 0.01]),
        ),
        "verbose": hp.choice("verbose", [0]),
        "random_state": hp.choice("random_state", [4, 5]),
        "copy_x": hp.choice("copy_x", [True]),
        "algorithm": hp.choice("algorithm", ["lloyd"]),
    }

    mt = train_model(
        create_model,
        get_data,
        params,
        True,
        10,
    )

    assert type(mt) == type(create_model)
