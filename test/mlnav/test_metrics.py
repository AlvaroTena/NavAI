import os

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mlnav.evaluate.metrics import Unsupervised_Metrics


@pytest.fixture(scope="module")
def get_data():
    """Returns a numpy array with data to train"""
    data = pd.read_csv("Files_Test/metrics_tests/test_scenario.csv")
    train_data = data.to_numpy()
    train_data = np.delete(train_data, 0, 1)

    return train_data


@pytest.fixture(scope="module")
def get_model(get_data):
    """Returns a list with kmeans models"""
    km3 = KMeans(
        n_clusters=3,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=0.0001,
        verbose=0,
        random_state=5,
        copy_x=True,
        algorithm="lloyd",
    )
    km3.fit(get_data)

    return km3


def test_one_model(get_model, get_data):
    """Silhouette, CH, DBI, DBCV"""

    metrics = Unsupervised_Metrics(get_model, get_data)
    metrics_dict = metrics.get_metrics(True, True, True, -1, False, False)

    assert len(metrics_dict.keys()) == 3 and (
        str(metrics_dict["Silhouette"]).isnumeric()
        == str(metrics_dict["Calinski_harabasz"]).isnumeric()
        == str(metrics_dict["Davies_bouldin_score"]).isnumeric()
    )


def test_one_model_split(get_model, get_data):
    """Silhouette, CH, DBI, DBCV"""

    metrics = Unsupervised_Metrics(get_model, get_data)
    metrics_dict = metrics.get_metrics(True, True, True, 10, False, False)

    assert len(metrics_dict.keys()) == 3 and (
        str(metrics_dict["Silhouette"]).isnumeric()
        == str(metrics_dict["Calinski_harabasz"]).isnumeric()
        == str(metrics_dict["Davies_bouldin_score"]).isnumeric()
    )


def test_outputs(get_model, get_data):
    """Save elbow and silhouette plots of multiple models"""

    # Get the path of the files
    base_path = os.path.abspath("Files_Test/metrics_tests/outputs_test/")
    silhouette_path = os.path.join(base_path, "silhouette.png")

    metrics = Unsupervised_Metrics(get_model, get_data)
    metrics.get_metrics(True, True, True, -1, True, True, base_path)

    # Check if files were created
    assert os.path.isfile(silhouette_path) == True

    # Delete files
    os.remove(silhouette_path)
