import sys

import numpy as np
import psutil
from scipy.spatial import distance
from sklearn.model_selection import StratifiedShuffleSplit


def get_closest_centroid(obs, centroids):
    """
    Function for retrieving the closest centroid to the given observation
    in terms of the Euclidean distance.

    Parameters
    ----------
    obs : array
        An array containing the observation to be matched to the nearest centroid
    centroids : array
        An array containing the centroids

    Returns
    -------
    min_centroid : array
        The centroid closes to the obs
    """
    min_distance = sys.float_info.max
    min_centroid = 0

    for c in centroids:
        dist = distance.euclidean(obs, c)
        if dist < min_distance:
            min_distance = dist
            min_centroid = c

    return min_centroid


def get_prediction_strength(k, train_centroids, x_test, test_labels):
    """
    Function for calculating the prediction strength of clustering

    Parameters
    ----------
    k : int
        The number of clusters
    train_centroids : array
        Centroids from the clustering on the training set
    x_test : array
        Test set observations
    test_labels : array
        Labels predicted for the test set

    Returns
    -------
    prediction_strength : float
        Calculated prediction strength
    """
    n_test = len(x_test)
    available_ram = int(psutil.virtual_memory().available * 0.7)
    max_mem_size_bytes = n_test * n_test * np.dtype(np.uint8).itemsize

    if max_mem_size_bytes > available_ram:
        # Reduce the size of the subsample to fit the percentage of available RAM
        subsample_size = max(int(available_ram**0.5), 5)
        # Make sure that the subsample size is not larger than the number of samples in x_test
        subsample_size = min(subsample_size, n_test)
    else:
        # Use n_test as the size of the subsample if the maximum size of the co-membership matrix fits in the available RAM
        subsample_size = n_test

    _, subsample_indices = next(
        StratifiedShuffleSplit(
            n_splits=1, test_size=subsample_size, random_state=42
        ).split(X=x_test, y=test_labels)
    )
    subsample = x_test.iloc[subsample_indices]

    # populate the co-membership matrix
    D = np.zeros(shape=(subsample_size, subsample_size), dtype=np.uint8)
    for c1, x1 in enumerate(subsample.values):
        for c2, x2 in enumerate(subsample.values):
            if tuple(x1) != tuple(x2):
                if tuple(get_closest_centroid(x1, train_centroids)) == tuple(
                    get_closest_centroid(x2, train_centroids)
                ):
                    D[c1, c2] = 1

    # calculate the prediction strengths for each cluster
    ss = []
    for j in range(k):
        s = 0
        examples_j = x_test[test_labels == j, :].tolist()
        n_examples_j = len(examples_j)
        for x1, l1, c1 in zip(x_test, test_labels, list(range(n_test))):
            for x2, l2, c2 in zip(x_test, test_labels, list(range(n_test))):
                if tuple(x1) != tuple(x2) and l1 == l2 and l1 == j:
                    s += D[c1, c2]
        ss.append(s / (n_examples_j * (n_examples_j - 1)))

    prediction_strength = min(ss)

    return prediction_strength
