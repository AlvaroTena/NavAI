import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network, utils
from tf_agents.specs import distribution_spec, tensor_spec


class BernoulliProjectionNetwork(network.DistributionNetwork):
    """Generates a Bernoulli distribution to predict 0/1 decisions for each action,
    with support for masking actions (masking is equivalent to forcing to 0)."""

    def __init__(
        self,
        sample_spec,
        logits_init_output_factor=0.1,
        name="BernoulliProjectionNetwork",
    ):
        """Creates an instance of BernoulliProjectionNetwork.

        Args:
          sample_spec: A tensor_spec.BoundedTensorSpec that details the shape and dtype
            of the samples in the distribution. Sample_spec is expected to have shape (286,)
            in this case.
          logits_init_output_factor: Factor for the initialization of the logits layer weights.
          name: Name of the network.

        """
        if not tensor_spec.is_bounded(sample_spec):
            raise ValueError(
                "sample_spec debe ser bounded. Recibido: %s." % type(sample_spec)
            )
        if not tensor_spec.is_discrete(sample_spec):
            raise ValueError(
                "sample_spec debe ser discreto. Recibido: %s." % sample_spec
            )

        output_shape = sample_spec.shape
        output_spec = self._output_distribution_spec(output_shape, sample_spec, name)

        super(BernoulliProjectionNetwork, self).__init__(
            input_tensor_spec=None, state_spec=(), output_spec=output_spec, name=name
        )

        self._sample_spec = sample_spec
        self._output_shape = output_shape

        # Projection layer generates np.prod(sample_spec.shape) logits
        self._projection_layer = tf.keras.layers.Dense(
            np.prod(self._sample_spec.shape),
            kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(
                scale=logits_init_output_factor
            ),
            bias_initializer=tf.keras.initializers.Zeros(),
            name="logits",
        )

    def _output_distribution_spec(self, output_shape, sample_spec, network_name):
        input_param_spec = {
            "logits": tensor_spec.TensorSpec(
                shape=output_shape, dtype=tf.float32, name=network_name + "_logits"
            )
        }
        return distribution_spec.DistributionSpec(
            tfp.distributions.Bernoulli,
            input_param_spec,
            sample_spec=sample_spec,
            dtype=sample_spec.dtype,
        )

    def call(self, inputs, outer_rank, training=False, mask=None):
        """Calculate the logits and construct the Bernoulli distribution.

        If a `mask` is provided, the logits corresponding to invalid positions
        are assigned a very negative value to force a probability of 0.
        """
        batch_squash = utils.BatchSquash(outer_rank)
        inputs = batch_squash.flatten(inputs)
        inputs = tf.cast(inputs, tf.float32)

        logits = self._projection_layer(inputs, training=training)
        logits = tf.reshape(logits, [-1] + list(self._output_shape))
        logits = batch_squash.unflatten(logits)

        if mask is not None:
            # If the mask has lower rank than logits, the dimension is expanded.
            if mask.shape.rank < logits.shape.rank:
                mask = tf.expand_dims(mask, -1)
            # For masking: invalid entries are assigned a very negative value.
            almost_neg_inf = tf.constant(logits.dtype.min, dtype=logits.dtype)
            logits = tf.compat.v2.where(tf.cast(mask, tf.bool), logits, almost_neg_inf)

        # The Bernoulli distribution is constructed using the calculated logits.
        dist = self.output_spec.build_distribution(logits=logits)
        # The distribution is wrapped in an Independent distribution.
        independent_dist = tfp.distributions.Independent(
            dist, reinterpreted_batch_ndims=len(self._output_shape)
        )

        return independent_dist, ()
