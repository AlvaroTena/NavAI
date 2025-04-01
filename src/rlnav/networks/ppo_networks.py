import tensorflow as tf
from rlnav.networks import (
    actor_distribution_rnn_mask_network,
    multiobjective_value_rnn_network,
)
from tf_agents.networks import actor_distribution_network, value_network


def create_actor_net(observation_spec, action_spec, fc_layer_params=(64, 64)):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=fc_layer_params,
        kernel_initializer="he_normal",
        activation_fn=tf.keras.activations.relu,
    )
    return actor_net


def create_value_net(observation_spec, fc_layer_params=(64, 64)):
    value_net = value_network.ValueNetwork(
        observation_spec,
        fc_layer_params=fc_layer_params,
        kernel_initializer="he_normal",
        activation_fn=tf.keras.activations.relu,
    )
    return value_net


def create_actor_rnn_net(
    observation_spec,
    action_spec,
    input_fc_layer_params=(256, 128),
    lstm_size=(64, 32),
    output_fc_layer_params=(64, 128),
):
    actor_net = actor_distribution_rnn_mask_network.ActorDistributionRnnMaskNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=tf.keras.activations.relu,
    )
    return actor_net


def create_value_rnn_net(
    observation_spec,
    input_fc_layer_params=(256, 128),
    lstm_size=(64, 32),
    output_fc_layer_params=(64,),
):
    value_net = multiobjective_value_rnn_network.MultiObjectiveValueRnnNetwork(
        observation_spec,
        num_objectives=3,  #! Change to dynamic number of objectives
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=tf.keras.activations.relu,
    )
    return value_net
