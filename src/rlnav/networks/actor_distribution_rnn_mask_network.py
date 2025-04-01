import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from rlnav.networks import bernoulli_projection_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import nest_utils


def _bernoulli_projection_net(action_spec, logits_init_output_factor=0.1):
    return bernoulli_projection_network.BernoulliProjectionNetwork(
        action_spec, logits_init_output_factor=logits_init_output_factor
    )


class ActorDistributionRnnMaskNetwork(
    actor_distribution_rnn_network.ActorDistributionRnnNetwork
):

    def __init__(
        self,
        input_tensor_spec,
        output_tensor_spec,
        preprocessing_layers=None,
        preprocessing_combiner=None,
        conv_layer_params=None,
        input_fc_layer_params=(200, 100),
        input_dropout_layer_params=None,
        lstm_size=None,
        output_fc_layer_params=(200, 100),
        activation_fn=tf.keras.activations.relu,
        dtype=tf.float32,
        discrete_projection_net=_bernoulli_projection_net,
        continuous_projection_net=actor_distribution_rnn_network._normal_projection_net,
        rnn_construction_fn=None,
        rnn_construction_kwargs={},
        name="ActorDistributionRnnNetwork",
    ):
        super(ActorDistributionRnnMaskNetwork, self).__init__(
            input_tensor_spec,
            output_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            input_fc_layer_params=input_fc_layer_params,
            input_dropout_layer_params=input_dropout_layer_params,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            activation_fn=activation_fn,
            dtype=dtype,
            discrete_projection_net=discrete_projection_net,
            continuous_projection_net=continuous_projection_net,
            rnn_construction_fn=rnn_construction_fn,
            rnn_construction_kwargs=rnn_construction_kwargs,
            name=name,
        )

    def call(self, observation, step_type, network_state=(), training=False, mask=None):
        state, network_state = self._lstm_encoder(
            observation,
            step_type=step_type,
            network_state=network_state,
            training=training,
        )
        outer_rank = nest_utils.get_outer_rank(observation, self.input_tensor_spec)

        if mask is None:
            output_actions = tf.nest.map_structure(
                lambda proj_net: proj_net(state, outer_rank, training=training)[0],
                self._projection_networks,
            )
        else:
            output_actions = tf.nest.map_structure(
                lambda proj_net, m: proj_net(
                    state, outer_rank, training=training, mask=m
                )[0],
                self._projection_networks,
                mask,
            )

        return output_actions, network_state
