from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def silhouette_scorer(estimator, x, y=None):
    """Compute Silhouette score for CV

    Args:
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        x : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning. Not used, present here for API consistency by convention.

    Returns:
        score : Float
            The resulting Silhouette score.
    """
    labels = estimator.predict(x)
    score = silhouette_score(x, labels)
    return -score


def davies_scorer(estimator, x, y=None):
    """Compute Davies-Bouldin score for CV

    Args:
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        x : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning. Not used, present here for API consistency by convention.

    Returns:
        score : Float
            The resulting Davies-Bouldin score.
    """
    labels = estimator.predict(x)
    db_score = davies_bouldin_score(x, labels)
    return db_score


def calinski_scorer(estimator, x, y=None):
    """Compute Calinski-Harabasz score for CV

    Args:
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        x : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning. Not used, present here for API consistency by convention.

    Returns:
        score : Float
            The resulting Calinski-Harabasz score.
    """
    labels = estimator.predict(x)
    ch_score = calinski_harabasz_score(x, labels)
    return -ch_score


def calinski_davies_score(estimator, x, y=None):
    """Compute Calinski-Harabasz & Davies-Bouldin score for CV

    Args:
        estimator : estimator object implementing 'fit'
            The object to use to fit the data.
        x : array-like of shape (n_samples, n_features)
            The data to fit. Can be for example a list, or an array.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
            The target variable to try to predict in the case of supervised learning. Not used, present here for API consistency by convention.

    Returns:
        score : Float
            The resulting Calinski-Harabasz & Davies-Bouldin score.
    """
    labels = estimator.predict(x)

    ch_score = calinski_harabasz_score(x, labels)
    db_score = davies_bouldin_score(x, labels)

    # Mix both metrics, weights can be adjusted
    score = 0.5 * ch_score - 0.5 * db_score

    return -score
