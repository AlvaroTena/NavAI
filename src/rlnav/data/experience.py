import numpy as np
import tensorflow as tf


def pad_batch(experience, active_mask, total_envs):
    """
    Dado un batch de experiencia (estructura anidada de tensores) cuya primera dimensión
    corresponde al número de entornos activos (active_count) y una máscara booleana (active_mask)
    de longitud total_envs, rellena (paddea) cada tensor para que su primera dimensión sea
    total_envs.

    Args:
        experience: Estructura anidada (por ejemplo, dict o list) de tensores con shape [active_count, ...].
        active_mask: Lista booleana de longitud total_envs indicando qué posiciones están activas.
                     Se asume que los valores True están al principio, es decir:
                     [True] * active_count + [False] * (total_envs - active_count).
        total_envs: Número total de entornos (batch_size esperado por el replay buffer).

    Returns:
        La misma estructura anidada en la que cada tensor tiene shape [total_envs, ...],
        donde las posiciones inactivas se rellenan con ceros.
    """
    active_count = sum(active_mask)

    def pad_tensor(t):
        t = tf.convert_to_tensor(t)
        pad_size = total_envs - active_count
        if pad_size == 0:
            return t
        # Calcula la forma para el padding: [pad_size] concatenado con la forma de t a partir del segundo eje.
        dummy_shape = tf.concat([[pad_size], tf.shape(t)[1:]], axis=0)
        dummy = tf.zeros(dummy_shape, dtype=t.dtype)
        # Concatenamos a lo largo del eje 0 para obtener un tensor de shape [total_envs, ...]
        padded = tf.concat([t, dummy], axis=0)
        return padded

    return tf.nest.map_structure(pad_tensor, experience)
