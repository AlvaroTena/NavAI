import tensorflow as tf


def sigmoid_squash_to_spec(inputs, spec):
    """Maps inputs with arbitrary range to range defined by spec using `sigmoid`."""
    return spec.minimum + (spec.maximum - spec.minimum) * tf.nn.sigmoid(inputs)


def softsign_squash_to_spec(inputs, spec):
    """Maps inputs with arbitrary range to range defined by spec using `softsign`."""
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0

    return means + magnitudes * tf.nn.softsign(inputs)


def clip_squash_to_spec(inputs, spec):
    """Maps inputs with arbitrary range to range defined by spec using `clip`."""
    means = (spec.maximum + spec.minimum) / 2.0
    magnitudes = (spec.maximum - spec.minimum) / 2.0

    return means + magnitudes * tf.clip_by_value(inputs, -1, 1)
