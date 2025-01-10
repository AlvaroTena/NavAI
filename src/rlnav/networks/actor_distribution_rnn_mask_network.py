import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.utils import nest_utils


class ActorDistributionRnnMaskNetwork(
    actor_distribution_rnn_network.ActorDistributionRnnNetwork
):
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
