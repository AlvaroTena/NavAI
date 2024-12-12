from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ppo import ppo_policy
from tf_agents.distributions import reparameterized_sampling
from tf_agents.networks import network
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common

import rlnav.agent.beta_distribution as beta


class PPOPolicyMasked(ppo_policy.PPOPolicy):
    """An ActorPolicy that also returns policy_info needed for PPO training.

    This policy requires two networks: the usual `actor_network` and the
    additional `value_network`. The value network can be executed with the
    `apply_value_network()` method.

    When the networks have state (RNNs, LSTMs) you must be careful to pass the
    state for the actor network to `action()` and the state of the value network
    to `apply_value_network()`. Use `get_initial_value_state()` to access
    the state of the value network.
    """

    def __init__(
        self,
        observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
        *args,
        **kwargs,
    ):
        super(PPOPolicyMasked, self).__init__(*args, **kwargs)
        self._observation_and_action_constraint_splitter = (
            observation_and_action_constraint_splitter
        )

    def apply_ray_mask(self, actions, action_mask, a_min, a_max, a_r_min, a_r_max):
        """
        Applies the Ray Mask to adjust irrelevant actions within a given range and keeps relevant actions unchanged.

        This function maps the irrelevant actions (indicated by 0s in the `action_mask`) to the specified range
        [a_r_min, a_r_max] using the Ray Mask transformation, while keeping the relevant actions (indicated by 1s in the
        `action_mask`) unchanged. The mapping is done by scaling the original actions within the bounds of the relevant
        action space and adjusting them according to the specified range.

        Parameters:
        -----------
        actions : np.ndarray
            Array of original actions with shape (batch_size, num_actions).

        action_mask : np.ndarray
            Binary mask indicating relevant (1) and irrelevant (0) actions, with shape (batch_size, num_actions).

        a_min : float
            Lower bound of the original action space.

        a_max : float
            Upper bound of the original action space.

        a_r_min : float
            Lower bound of the range to which irrelevant actions should be mapped.

        a_r_max : float
            Upper bound of the range to which irrelevant actions should be mapped.

        Returns:
        --------
        np.ndarray
            Array of actions where irrelevant actions have been mapped to the specified range and relevant actions remain unchanged.
        """

        a_min = tf.fill(tf.shape(actions)[1:], a_min)
        a_max = tf.fill(tf.shape(actions)[1:], a_max)
        a_r_min = tf.fill(tf.shape(actions)[1:], a_r_min)
        a_r_max = tf.fill(tf.shape(actions)[1:], a_r_max)

        c = 0.5 * (a_r_min + a_r_max)

        delta_a = actions - c

        # lambda_A
        lambda_A_numerador = tf.where(delta_a > 0, a_max - actions, a_min - actions)
        lambda_A_denominador = tf.where(delta_a != 0, delta_a, 1e-6)
        lambda_A = tf.where(
            delta_a != 0, lambda_A_numerador / lambda_A_denominador, tf.float32.max
        )

        lambda_A_scalar = tf.reduce_min(lambda_A, axis=1, keepdims=True)

        # lambda_Ar
        lambda_Ar_numerador = tf.where(
            delta_a > 0, a_r_max - actions, a_r_min - actions
        )
        lambda_Ar_denominador = tf.where(delta_a != 0, delta_a, 1e-6)
        lambda_Ar = tf.where(
            delta_a != 0, lambda_Ar_numerador / lambda_Ar_denominador, tf.float32.max
        )

        lambda_Ar_scalar = tf.reduce_min(lambda_Ar, axis=1, keepdims=True)

        scaling_factor = tf.where(
            lambda_Ar_scalar != 0, lambda_A_scalar / lambda_Ar_scalar, 1.0
        )

        # mapping function
        a_r = c + scaling_factor * (actions - c)
        adjusted_actions = tf.clip_by_value(a_r, a_r_min, a_r_max)

        masked_actions = tf.where(action_mask == 0, adjusted_actions, actions)

        return masked_actions

    def _action(
        self,
        time_step: ts.TimeStep,
        policy_state: types.NestedTensor,
        seed: Optional[types.Seed] = None,  # type: ignore
    ) -> policy_step.PolicyStep:
        """Implementation of `action`.

        Args:
            time_step: A `TimeStep` tuple corresponding to `time_step_spec()`.
            policy_state: A Tensor, or a nested dict, list or tuple of Tensors
                representing the previous policy_state.
            seed: Seed to use if action performs sampling (optional).

        Returns:
            A `PolicyStep` named tuple containing:
                `action`: An action Tensor matching the `action_spec`.
                `state`: A policy state tensor to be fed into the next call to action.
                `info`: Optional side information such as action log probabilities.
        """
        seed_stream = tfp.util.SeedStream(seed=seed, salt="tf_agents_tf_policy")
        distribution_step = self._distribution(
            time_step, policy_state
        )  # pytype: disable=wrong-arg-types
        actions = tf.nest.map_structure(
            lambda d: reparameterized_sampling.sample(d, seed=seed_stream()),
            distribution_step.action,
        )
        info = distribution_step.info

        actions_mask = None
        if self.observation_and_action_constraint_splitter is not None:
            _, actions_mask = self.observation_and_action_constraint_splitter(
                time_step.observation
            )

            a_min = self.action_spec.minimum
            a_max = self.action_spec.maximum

            epsilon = 1e-6
            a_r_min = 10.0
            a_r_max = 15.0

            masked_actions = self.apply_ray_mask(
                actions,
                actions_mask,
                a_min,
                a_max,
                a_r_min + epsilon,
                a_r_max - epsilon,
            )
        else:
            masked_actions = actions

        if self.emit_log_probability:
            try:

                def _compute_log_prob(single_distribution, single_action, single_mask):
                    epsilon = 1e-6
                    # Inverse transformation on actions
                    single_action = single_distribution.bijector.inverse(single_action)
                    single_action = tf.clip_by_value(
                        single_action, epsilon, 1 - epsilon
                    )

                    # Compute log_prob per dimension
                    single_log_prob = (
                        single_distribution.distribution.distribution.log_prob(
                            single_action
                        )
                    )

                    # Apply mask if provided
                    if single_mask is not None:
                        single_log_prob *= single_mask

                    # single_log_prob = tf.where(
                    #     tf.math.is_finite(single_log_prob),
                    #     single_log_prob,
                    #     tf.zeros_like(single_log_prob),
                    # )

                    # Sum log-probs over action dimensions
                    return tf.reduce_sum(single_log_prob, axis=-1)

                log_probability = tf.nest.map_structure(
                    _compute_log_prob(distribution_step.action, actions, actions_mask)
                )
                info = policy_step.set_log_probability(info, log_probability)
            except:
                raise TypeError(
                    "%s does not support emitting log-probabilities."
                    % type(self).__name__
                )

        return distribution_step._replace(action=masked_actions, info=info)

    def _distribution(self, time_step, policy_state, training=False):
        if not policy_state:
            policy_state = {"actor_network_state": (), "value_network_state": ()}
        else:
            policy_state = policy_state.copy()

        if "actor_network_state" not in policy_state:
            policy_state["actor_network_state"] = ()
        if "value_network_state" not in policy_state:
            policy_state["value_network_state"] = ()

        new_policy_state = {"actor_network_state": (), "value_network_state": ()}

        (distributions, new_policy_state["actor_network_state"]) = (
            self._apply_actor_network(
                time_step, policy_state["actor_network_state"], training=training
            )
        )

        if self._collect:
            policy_info = {
                "dist_params": beta.get_distribution_params(
                    distributions,
                    legacy_distribution_network=isinstance(
                        self._actor_network, network.DistributionNetwork
                    ),
                )
            }

            if not self._compute_value_and_advantage_in_train:
                # If value_prediction is not computed in agent.train it needs to be
                # computed and saved here.
                (
                    policy_info["value_prediction"],
                    new_policy_state["value_network_state"],
                ) = self.apply_value_network(
                    time_step.observation,
                    time_step.step_type,
                    value_state=policy_state["value_network_state"],
                    training=False,
                )
        else:
            policy_info = ()

        # Disable lint for TF arrays.
        if not common.safe_has_state(
            new_policy_state["actor_network_state"]
        ) and not common.safe_has_state(new_policy_state["value_network_state"]):
            new_policy_state = ()
        elif not common.safe_has_state(new_policy_state["value_network_state"]):
            del new_policy_state["value_network_state"]
        elif not common.safe_has_state(new_policy_state["actor_network_state"]):
            del new_policy_state["actor_network_state"]

        return policy_step.PolicyStep(distributions, new_policy_state, policy_info)
