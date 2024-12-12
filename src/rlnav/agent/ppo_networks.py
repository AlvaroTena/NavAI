import numpy as np
import tensorflow as tf
from tf_agents.networks import (
    actor_distribution_network,
    actor_distribution_rnn_network,
    normal_projection_network,
    value_network,
    value_rnn_network,
)

from rlnav.agent.ppo_beta_projection import BetaProjectionNetwork
from rlnav.agent.ppo_normal_projection import (
    clip_squash_to_spec,
    sigmoid_squash_to_spec,
    softsign_squash_to_spec,
)


def tanh_normal_projection_net(
    action_spec, init_action_stddev=0.5, init_means_output_factor=1.0
):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        state_dependent_std=True,
    )


def sigmoid_normal_projection_net(
    action_spec, init_action_stddev=0.5, init_means_output_factor=1.0
):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        mean_transform=sigmoid_squash_to_spec,
        state_dependent_std=True,
    )


def softsign_normal_projection_net(
    action_spec, init_action_stddev=0.5, init_means_output_factor=1.0
):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        mean_transform=softsign_squash_to_spec,
        state_dependent_std=True,
    )


def clip_normal_projection_net(
    action_spec, init_action_stddev=0.5, init_means_output_factor=1.0
):
    std_bias_initializer_value = np.log(np.exp(init_action_stddev) - 1)

    return normal_projection_network.NormalProjectionNetwork(
        action_spec,
        init_means_output_factor=init_means_output_factor,
        std_bias_initializer_value=std_bias_initializer_value,
        mean_transform=clip_squash_to_spec,
        state_dependent_std=True,
    )


def beta_projection_net(action_spec, alpha_init_value=1, beta_init_value=5):
    return BetaProjectionNetwork(
        action_spec,
        alpha_init_value=alpha_init_value,
        beta_init_value=beta_init_value,
        scale_distribution=True,
    )


def create_actor_net(observation_spec, action_spec, fc_layer_params=(64, 64)):
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        observation_spec,
        action_spec,
        fc_layer_params=fc_layer_params,
        kernel_initializer="he_normal",
        activation_fn=tf.keras.activations.relu,
        continuous_projection_net=beta_projection_net,
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
    actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
        observation_spec,
        action_spec,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=tf.keras.activations.relu,
        continuous_projection_net=beta_projection_net,
    )
    return actor_net


def create_value_rnn_net(
    observation_spec,
    input_fc_layer_params=(256, 128),
    lstm_size=(64, 32),
    output_fc_layer_params=(64,),
):
    value_net = value_rnn_network.ValueRnnNetwork(
        observation_spec,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=tf.keras.activations.relu,
    )
    return value_net
