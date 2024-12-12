import numpy as np
import tensorflow as tf


def log_prob_per_dim(single_distribution, single_action):
    loc = single_distribution.loc
    scale_diag = single_distribution.scale.diag

    log_prob_per_dim = (
        -0.5 * tf.math.log(2 * np.pi)
        - tf.math.log(scale_diag)
        - 0.5 * ((single_action - loc) / scale_diag) ** 2
    )

    return log_prob_per_dim


def entropy_per_dim(distribution):
    scale_diag = distribution.scale.diag

    entropy_per_dim = 0.5 * tf.math.log(2 * np.pi * tf.square(scale_diag)) + 0.5

    return entropy_per_dim


def kl_divergence_per_dim(from_distribution, to_distribution):
    """Computes the KL divergence per action dimension between two distributions."""
    from_loc = from_distribution.loc
    from_scale = from_distribution.scale.diag
    to_loc = to_distribution.loc
    to_scale = to_distribution.scale.diag

    from_var = tf.square(from_scale)
    to_var = tf.square(to_scale)

    kl_per_dim = 0.5 * (
        tf.square(from_loc - to_loc) / to_var
        + (from_var / to_var)
        - tf.math.log(from_var / to_var)
        - 1
    )

    return kl_per_dim
