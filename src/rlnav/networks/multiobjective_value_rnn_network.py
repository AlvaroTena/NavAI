import tensorflow as tf
from tf_agents.networks import value_rnn_network


class MultiObjectiveValueRnnNetwork(value_rnn_network.ValueRnnNetwork):
    """Value RNN network for multi-objective reinforcement learning.

    This network extends the ValueRnnNetwork to output multiple values:
      - V_sub_region: Scalar value for Clipped Surrogate Objective
      - V_global: Vector with `num_objectives` values for Global Estimation

    Theses values are used to optimize the reward frontier in multi-objective.

    For theory, see
    "Multi-Objective Exploration for Proximal Policy  Optimization"
    by Khoi, Nguyen Do Hoang and Pham Van, Cuong and Tran, Hoang Vu and Truong, Cao Dung.
    See: https://doi.org/10.1109/ATiGB50996.2021.9423319 for full paper.
    """

    def __init__(
        self,
        input_tensor_spec,
        num_objectives,  # Número de objetivos (por ejemplo, 3 para Norte, Este y Arriba)
        preprocessing_layers=None,
        preprocessing_combiner=None,
        conv_layer_params=None,
        input_fc_layer_params=(75, 40),
        lstm_size=(40,),
        output_fc_layer_params=(75, 40),
        activation_fn=tf.keras.activations.relu,
        dtype=tf.float32,
        name="MultiObjectiveValueRnnNetwork",
    ):
        super(MultiObjectiveValueRnnNetwork, self).__init__(
            input_tensor_spec=input_tensor_spec,
            preprocessing_layers=preprocessing_layers,
            preprocessing_combiner=preprocessing_combiner,
            conv_layer_params=conv_layer_params,
            input_fc_layer_params=input_fc_layer_params,
            input_dropout_layer_params=None,
            lstm_size=lstm_size,
            output_fc_layer_params=output_fc_layer_params,
            activation_fn=activation_fn,
            dtype=dtype,
            name=name,
        )
        del self._postprocessing_layers

        self._subregion_layer = tf.keras.layers.Dense(
            1,
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
            name="V_subregion",
        )
        self._global_layer = tf.keras.layers.Dense(
            num_objectives,
            activation=tf.keras.activations.tanh,
            kernel_initializer=tf.random_uniform_initializer(minval=-0.03, maxval=0.03),
            name="V_global",
        )

    def call(self, observation, step_type=None, network_state=(), training=False):
        state, network_state = self._lstm_encoder(
            observation,
            step_type=step_type,
            network_state=network_state,
            training=training,
        )

        subregion_value = self._subregion_layer(state, training=training)
        global_value = self._global_layer(state, training=training)

        subregion_value = 10 * tf.squeeze(subregion_value, -1)
        global_value = 10 * global_value

        return (subregion_value, global_value), network_state
