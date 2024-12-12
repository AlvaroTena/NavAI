import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mlnav.train.train import train_model

data = pd.read_csv("Files_Test/train_tests/test_scenario.csv")
params = {
    "n_clusters": 4,
    "init": "k-means++",
    "n_init": "auto",
    "max_iter": 100,
    "tol": 0.0001,
    "verbose": 0,
    "random_state": 5,
    "copy_x": True,
    "algorithm": "lloyd",
}


@pytest.fixture(scope="module")
def create_model():
    """Returns a KMeans model"""
    kmeans = KMeans(**params)
    return kmeans


def test_train_kmeans_centroids(create_model):
    """Centroids correspond to number of clusters"""

    km = train_model(create_model, data, params, hyperopt=False)
    centroids = km.cluster_centers_
    assert len(centroids) == params["n_clusters"]
