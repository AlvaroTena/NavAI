import os

import pandas as pd
import pytest
from sklearn.cluster import KMeans

from mlnav.report.plot_distributions import plot_pairplots


@pytest.fixture(scope="module")
def get_data():
    """Returns a numpy array with data to train"""
    data = pd.read_hdf(
        "Files_Test/plot_distributions_tests/test_scenario.h5", key="data"
    )

    return data


@pytest.fixture(scope="module")
def create_model():
    """Returns a KMeans model"""
    return KMeans(2, n_init="auto")


def test_plot_distributions_2d(get_data, create_model):
    """Tests if plot_features saves the kde features for a non corrupt file"""

    model: KMeans = create_model.fit(get_data)
    labels = model.predict(get_data)

    cluster_pairplots, _, _ = plot_pairplots(
        get_data,
        labels,
        -1,
        drop_noise=False,
        plot_pairplots2d=False,
        plot_pairplots3d=False,
    )

    assert model.get_params()["n_clusters"] == len(cluster_pairplots)


def test_plot_distributions_2d_mixed(get_data, create_model):
    """Tests if plot_features saves the kde features for a non corrupt file"""

    model: KMeans = create_model.fit(get_data)
    labels = model.predict(get_data)

    _, pairplots2d, _ = plot_pairplots(
        get_data,
        labels,
        -1,
        drop_noise=False,
        plot_cluster_pairplots=False,
        plot_pairplots3d=False,
    )

    assert pairplots2d


def test_plot_distributions_3d(get_data, create_model):
    """Tests if plot_features saves the kde features for a non corrupt file"""

    model: KMeans = create_model.fit(get_data)
    labels = model.predict(get_data)

    _, _, pairplots3d = plot_pairplots(
        get_data,
        labels,
        -1,
        drop_noise=False,
        plot_cluster_pairplots=False,
        plot_pairplots2d=False,
    )

    assert pairplots3d
