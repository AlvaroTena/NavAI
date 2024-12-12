import sys

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import distribution_spec, tensor_spec


class BetaProjectionNetwork(network.DistributionNetwork):
    """
    Generates a Beta distribution of tfp by predicting the alpha and beta parameters.
    """

    def __init__(
        self,
        sample_spec,
        activation_fn=None,
        alpha_init_value=1.0,
        beta_init_value=1.0,
        alpha_transform=tf.nn.softplus,
        beta_transform=tf.nn.softplus,
        scale_distribution=False,
        seed=None,
        seed_stream_class=tfp.util.SeedStream,
        name="BetaProjectionNetwork",
    ):
        """
        Creates an instance of BetaProjectionNetwork.
        Args:
            sample_spec: A `tensor_spec.BoundedTensorSpec` that details the shape and data types of the samples drawn from the output distribution.
                data types of the samples drawn from the output distribution.
            activation_fn: Activation function for use in dense layers.
            alpha_init_value: Initial value for alpha parameters.
            beta_init_value: Initial value for beta parameters.
            alpha_transform: Transformation to apply to the alpha parameters.
            beta_transform: Transformation to be applied to beta parameters.
            scale_distribution: If True, scales the distribution to match the limits of the `sample
                with the `sample_spec` limits.
            seed: Seed used to initialize Keras kernels.
            seed_stream_class: The seed stream class, usually tfp.util.SeedStream.
            name: A string representing the name of the network.

        """
        if len(tf.nest.flatten(sample_spec)) != 1:
            raise ValueError("BetaProjectionNetwork only supports single spec samples.")

        self._scale_distribution = scale_distribution
        output_spec = self._output_distribution_spec(sample_spec, name)
        super(BetaProjectionNetwork, self).__init__(
            input_tensor_spec=None,
            state_spec=(),
            output_spec=output_spec,
            name=name,
        )

        self._sample_spec = sample_spec
        self._is_multivariate = sample_spec.shape.ndims > 0
        self._alpha_transform = alpha_transform
        self._beta_transform = beta_transform
        seed_stream = seed_stream_class(
            seed=seed, salt="tf_agents_beta_projection_network"
        )
        alpha_seed = seed_stream()
        if alpha_seed is not None:
            alpha_seed = alpha_seed % sys.maxsize
        self._alpha_projection_layer = tf.keras.layers.Dense(
            sample_spec.shape.num_elements(),
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.VarianceScaling(seed=alpha_seed),
            bias_initializer=tf.keras.initializers.Constant(value=alpha_init_value),
            name="alpha_projection_layer",
        )

        beta_seed = seed_stream()
        if beta_seed is not None:
            beta_seed = beta_seed % sys.maxsize
        self._beta_projection_layer = tf.keras.layers.Dense(
            sample_spec.shape.num_elements(),
            activation=activation_fn,
            kernel_initializer=tf.keras.initializers.VarianceScaling(seed=beta_seed),
            bias_initializer=tf.keras.initializers.Constant(value=beta_init_value),
            name="beta_projection_layer",
        )

    def _output_distribution_spec(self, sample_spec, network_name):
        is_multivariate = sample_spec.shape.ndims > 0
        param_properties = tfp.distributions.Beta.parameter_properties()
        input_param_spec = {
            name: tensor_spec.TensorSpec(
                shape=properties.shape_fn(sample_spec.shape),
                dtype=sample_spec.dtype,
                name=network_name + "_" + name,
            )
            for name, properties in param_properties.items()
        }

        def distribution_builder(*args, **kwargs):
            distribution = tfp.distributions.Beta(*args, **kwargs)
            if is_multivariate:
                distribution = tfp.distributions.Independent(
                    distribution, reinterpreted_batch_ndims=1
                )
            if self._scale_distribution:
                distribution = self._scale_beta_distribution_to_spec(
                    distribution, self._sample_spec
                )
            return distribution

        return distribution_spec.DistributionSpec(
            distribution_builder, input_param_spec, sample_spec=sample_spec
        )

    def _scale_beta_distribution_to_spec(self, distribution, spec):
        """Scales a beta distribution to match the limits of the spec."""
        if not tensor_spec.is_bounded(spec):
            raise ValueError("The spec must have limits to scale the distribution.")

        # Construct the bijector mapping from [0, 1] to the spec boundaries.
        shift = spec.minimum
        scale = spec.maximum - spec.minimum
        shift = tf.cast(shift, tf.float32)
        scale = tf.cast(scale, tf.float32)
        bijector = tfp.bijectors.Chain(
            [
                tfp.bijectors.Shift(shift=shift),
                tfp.bijectors.Scale(scale=scale),
            ]
        )

        transformed_distribution = tfp.distributions.TransformedDistribution(
            distribution=distribution,
            bijector=bijector,
        )
        return transformed_distribution

    def call(self, inputs, outer_rank, training=False, mask=None):
        if inputs.dtype != self._sample_spec.dtype:
            raise ValueError(
                "BetaProjectionNetwork entries must match sample_spec.dtype."
            )

        if mask is not None:
            raise NotImplementedError(
                "BetaProjectionNetwork does not yet implement action masking; got mask={}".format(
                    mask
                )
            )

        batch_squash = network_utils.BatchSquash(outer_rank)
        inputs = batch_squash.flatten(inputs)

        alpha = self._alpha_projection_layer(inputs, training=training)
        beta = self._beta_projection_layer(inputs, training=training)

        alpha = tf.reshape(alpha, [-1] + self._sample_spec.shape.as_list())
        beta = tf.reshape(beta, [-1] + self._sample_spec.shape.as_list())

        if self._alpha_transform is not None:
            alpha = self._alpha_transform(alpha)
        alpha = tf.cast(alpha, self._sample_spec.dtype)

        if self._beta_transform is not None:
            beta = self._beta_transform(beta)
        beta = tf.cast(beta, self._sample_spec.dtype)

        alpha = batch_squash.unflatten(alpha)
        beta = batch_squash.unflatten(beta)

        return (
            self.output_spec.build_distribution(
                concentration1=alpha, concentration0=beta
            ),
            (),
        )
