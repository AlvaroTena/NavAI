import logging
from typing import Optional, Tuple

import rlnav.utils.vectorized_value_ops as vectorized_value_ops
import tensorflow as tf
import tensorflow_probability as tfp
from rlnav.agent.masked_ppo_agent import PPOAgentMasked
from rlnav.utils import multiobjective_ppo_utils as moe_ppo_utils
from tf_agents.agents.ppo import ppo_agent, ppo_utils
from tf_agents.networks import network
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, eager_utils, nest_utils, object_identity


class MultiObjectiveMaskedPPOAgent(PPOAgentMasked):
    def update_weight_vector(
        self, returns: tf.Tensor, value_preds: tf.Tensor
    ) -> tf.Tensor:
        """
        Updates the weight vector based on the absolute mean difference between returns and value predictions.

        The weight vector is calculated by normalizing the absolute differences between returns and value
        predictions across each objective dimension, resulting in weights that sum to 1.

        Args:
            returns: TD-lambda returns tensor with shape [T, B, d].
            value_preds: Value network predictions tensor with shape [T, B, d].

        Returns:
            A tensor of shape [d] with values in [0, 1] that sum to 1, representing the updated weight vector.
        """
        # Calculate absolute difference in each dimension and average over time and batch
        diff = tf.reduce_mean(tf.abs(returns - value_preds), axis=[0, 1])  # shape [d]

        # Normalize to ensure weights sum to 1, with epsilon to prevent division by zero
        total_diff = tf.reduce_sum(diff) + 1e-6
        new_weight_vector = diff / total_diff

        # Note: For smoothed updates, you could use:
        # alpha = 0.1  # Smoothing factor (optional hyperparameter)
        # new_weight_vector = alpha * new_weight_vector + (1 - alpha) * self._weight_vector

        return new_weight_vector

    def compute_return_and_advantage(
        self, next_time_steps: ts.TimeStep, value_preds: types.Tensor  # type: ignore
    ) -> Tuple[types.Tensor, types.Tensor]:  # type: ignore
        """Compute the Monte Carlo return and advantage.

        Args:
        next_time_steps: batched tensor of TimeStep tuples after action is taken.
        value_preds: Batched value prediction tensor. Should have one more entry
            in time index than time_steps, with the final value corresponding to the
            value prediction of the final state.

        Returns:
        tuple of (return, advantage), both are batched tensors.
        """
        discounts = next_time_steps.discount * tf.constant(
            self._discount_factor, dtype=tf.float32
        )

        rewards = next_time_steps.reward
        if self._debug_summaries:
            # Summarize rewards before they get normalized below.
            num_objectives = rewards.shape[-1]
            if not tf.config.list_logical_devices("TPU"):
                for i in range(num_objectives):
                    tf.compat.v2.summary.histogram(
                        name=f"rewards_{i}",
                        data=rewards[..., i],
                        step=self.train_step_counter,
                    )
            rewards_mean = tf.reduce_mean(rewards, axis=[0, 1])
            for i in range(num_objectives):
                tf.compat.v2.summary.scalar(
                    name=f"rewards_mean_{i}",
                    data=rewards_mean[i],
                    step=self.train_step_counter,
                )

        # Normalize rewards if self._reward_normalizer is defined.
        if self._reward_normalizer:
            rewards = self._reward_normalizer.normalize(
                rewards, center_mean=False, clip_value=self._reward_norm_clipping
            )
            if self._debug_summaries:
                if not tf.config.list_logical_devices("TPU"):
                    for i in range(num_objectives):
                        tf.compat.v2.summary.histogram(
                            name=f"rewards_normalized_{i}",
                            data=rewards[..., i],
                            step=self.train_step_counter,
                        )
                rewards_norm_mean = tf.reduce_mean(rewards, axis=[0, 1])
                for i in range(num_objectives):
                    tf.compat.v2.summary.scalar(
                        name=f"rewards_normalized_mean_{i}",
                        data=rewards_norm_mean[i],
                        step=self.train_step_counter,
                    )

        # Make discount 0.0 at end of each episode to restart cumulative sum
        #   end of each episode.
        episode_mask = common.get_episode_mask(next_time_steps)
        discounts *= episode_mask

        # Compute Monte Carlo returns. Data from incomplete trajectories, not
        #   containing the end of an episode will also be used, with a bootstrapped
        #   estimation from the last value.
        # Note that when a trajectory driver is used, then the final step is
        #   terminal, the bootstrapped estimation will not be used, as it will be
        #   multiplied by zero (the discount on the last step).
        final_value_bootstrapped = value_preds[:, -1, :]
        returns = vectorized_value_ops.discounted_return(
            rewards,
            discounts,
            time_major=False,
            final_value=final_value_bootstrapped,
        )
        if self._debug_summaries and not tf.config.list_logical_devices("TPU"):
            for i in range(num_objectives):
                tf.compat.v2.summary.histogram(
                    name=f"returns_{i}",
                    data=returns[..., i],
                    step=self.train_step_counter,
                )

        # Compute advantages.
        advantages = self.compute_advantages(rewards, returns, discounts, value_preds)

        if self._debug_summaries and not tf.config.list_logical_devices("TPU"):
            for i in range(num_objectives):
                tf.compat.v2.summary.histogram(
                    name=f"advantages_{i}",
                    data=advantages[..., i],
                    step=self.train_step_counter,
                )

        # Return TD-Lambda returns if both use_td_lambda_return and use_gae.
        if self._use_td_lambda_return:
            if not self._use_gae:
                logging.warning(
                    "use_td_lambda_return was True, but use_gae was "
                    "False. Using Monte Carlo return."
                )
            else:
                returns = tf.add(
                    advantages, value_preds[:, :-1, :], name="td_lambda_returns"
                )

        self._weight_vector = self.update_weight_vector(returns, value_preds[:, :-1, :])

        return returns, advantages

    def compute_advantages(
        self,
        rewards: types.NestedTensor,
        returns: types.Tensor,  # type: ignore
        discounts: types.Tensor,  # type: ignore
        value_preds: types.Tensor,  # type: ignore
    ) -> types.Tensor:  # type: ignore
        """Compute advantages, optionally using GAE.

        Based on baselines ppo1 implementation. Removes final timestep, as it needs
        to use this timestep for next-step value prediction for TD error
        computation.

        Args:
        rewards: Tensor of per-timestep rewards.
        returns: Tensor of per-timestep returns.
        discounts: Tensor of per-timestep discounts. Zero for terminal timesteps.
        value_preds: Cached value estimates from the data-collection policy.

        Returns:
        advantages: Tensor of length (len(rewards) - 1), because the final
            timestep is just used for next-step value prediction.
        """
        # Arg value_preds was appended with final next_step value. Make tensors
        #   next_value_preds by stripping first and last elements respectively.
        value_preds = value_preds[:, :-1, :]

        # Expand discounts dimension for broadcasting if rewards is 3D
        if discounts.shape.ndims == 2:
            discounts = tf.expand_dims(discounts, axis=-1)  # [T, B] -> [T, B, 1]

        if self._use_gae:
            advantages = vectorized_value_ops.generalized_advantage_estimation(
                values=value_preds,
                final_value=value_preds[:, -1, :],
                rewards=rewards,
                discounts=discounts,
                td_lambda=self._lambda,
                time_major=False,
            )
        else:
            with tf.name_scope("empirical_advantage"):
                advantages = returns - value_preds

        return advantages

    def _preprocess(self, experience):
        """Performs advantage calculation for the collected experience.

        Args:
        experience: A (batch of) experience in the form of a `Trajectory`. The
            structure of `experience` must match that of `self.collect_data_spec`.
            All tensors in `experience` must be shaped `[batch, time + 1, ...]` or
            [time + 1, ...]. The "+1" is needed as the last action from the set of
            trajectories cannot be used for training, as its advantage and returns
            are unknown.

        Returns:
        The processed experience which has normalized_advantages and returns
        filled in its policy info. The advantages and returns for the last
        transition are filled with 0s as they cannot be calculated.
        """
        # Try to be agnostic about the input type of experience before we call
        # to_transition() below.
        outer_rank = nest_utils.get_outer_rank(
            ppo_agent._get_discount(experience), self.collect_data_spec.discount
        )

        # Add 1 as the batch dimension for inputs that just have the time dimension,
        # as all utility functions below require the batch dimension.
        if outer_rank == 1:
            batched_experience = nest_utils.batch_nested_tensors(experience)
        else:
            batched_experience = experience

        # Get individual tensors from experience.
        num_steps = ppo_agent._get_discount(batched_experience).shape[1]
        if num_steps and num_steps <= 1:
            raise ValueError(
                "Experience used for advantage calculation must have >1 num_steps."
            )

        transition = self._collected_as_transition(batched_experience)
        time_steps, _, next_time_steps = transition

        # Compute the value predictions for states using the current value function.
        # To be used for return & advantage computation.
        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        if self._compute_value_and_advantage_in_train:
            value_state = self._collect_policy.get_initial_value_state(batch_size)
            value_preds_tuple, _ = self._collect_policy.apply_value_network(
                batched_experience.observation,
                batched_experience.step_type,
                value_state=value_state,
                training=False,
            )
            subregion_value = tf.stop_gradient(value_preds_tuple[0])
            global_value = tf.stop_gradient(value_preds_tuple[1])
        else:
            subregion_value = batched_experience.policy_info["value_prediction"]

        new_policy_info = {
            "dist_params": batched_experience.policy_info["dist_params"],
            "value_prediction": subregion_value,
        }

        # Add the calculated advantage and return into the input experience.
        returns, advantages = self.compute_return_and_advantage(
            next_time_steps, global_value
        )

        # Pad returns and normalized_advantages in the time dimension so that the
        # time dimensions are aligned with the input experience's time dimension.
        # When the output trajectory gets sliced by trajectory.to_transition during
        # training, the padded last timesteps will be automatically dropped.
        returns_shape = tf.shape(returns)
        num_objectives = returns_shape[-1]
        last_transition_padding = tf.zeros(
            (batch_size, 1, num_objectives), dtype=tf.float32
        )

        # Concatenate paddings with returns and advantages
        new_policy_info["return"] = tf.concat(
            [returns, last_transition_padding], axis=1
        )
        new_policy_info["advantage"] = tf.concat(
            [advantages, last_transition_padding], axis=1
        )

        # Remove the batch dimension iff the input experience does not have it.
        if outer_rank == 1:
            new_policy_info = nest_utils.unbatch_nested_tensors(new_policy_info)
        # The input experience with its policy info filled with the calculated
        # advantages and returns for each action.
        return experience.replace(policy_info=new_policy_info)

    def _train(self, experience, weights):
        if self._optimizer is None:
            raise ValueError("Optimizer is undefined.")

        experience = self._as_trajectory(experience)

        if self._compute_value_and_advantage_in_train:
            processed_experience = self._preprocess(experience)
        else:
            processed_experience = experience

        # Mask trajectories that cannot be used for training.
        valid_mask = moe_ppo_utils.make_trajectory_mask(processed_experience)
        if weights is None:
            masked_weights = valid_mask
        else:
            masked_weights = weights * valid_mask

        # Reconstruct per-timestep policy distribution from stored distribution
        #   parameters.
        old_action_distribution_parameters = processed_experience.policy_info[
            "dist_params"
        ]

        old_actions_distribution = ppo_utils.distribution_from_spec(
            self._action_distribution_spec,
            old_action_distribution_parameters,
            legacy_distribution_network=isinstance(
                self._actor_net, network.DistributionNetwork
            ),
        )

        independent_old_actions_distribution = tfp.distributions.Independent(
            old_actions_distribution,
            reinterpreted_batch_ndims=len(self.action_spec.shape),
        )

        # Compute log probability of actions taken during data collection, using the
        #   collect policy distribution.
        old_act_log_probs = common.log_probability(
            independent_old_actions_distribution,
            processed_experience.action,
            self._action_spec,
        )

        if self._debug_summaries and not tf.config.list_logical_devices("TPU"):
            actions_list = tf.nest.flatten(processed_experience.action)
            show_action_index = len(actions_list) != 1
            for i, single_action in enumerate(actions_list):
                action_name = "actions_{}".format(i) if show_action_index else "actions"
                tf.compat.v2.summary.histogram(
                    name=action_name, data=single_action, step=self.train_step_counter
                )

        time_steps = ts.TimeStep(
            step_type=processed_experience.step_type,
            reward=processed_experience.reward,
            discount=processed_experience.discount,
            observation=processed_experience.observation,
        )
        actions = processed_experience.action
        returns = processed_experience.policy_info["return"]
        advantages = processed_experience.policy_info["advantage"]

        # Combine advantages from all dimensions
        advantages = tf.reduce_sum(self._weight_vector * advantages, axis=-1)

        normalized_advantages = ppo_agent._normalize_advantages(
            advantages, variance_epsilon=1e-8
        )

        if self._debug_summaries and not tf.config.list_logical_devices("TPU"):
            tf.compat.v2.summary.histogram(
                name="advantages_normalized",
                data=normalized_advantages,
                step=self.train_step_counter,
            )
        old_value_predictions = processed_experience.policy_info["value_prediction"]

        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        # Loss tensors across batches will be aggregated for summaries.
        policy_gradient_losses = []
        value_estimation_losses = []
        l2_regularization_losses = []
        entropy_regularization_losses = []
        kl_penalty_losses = []

        loss_info = None  # TODO(b/123627451): Remove.
        variables_to_train = list(
            object_identity.ObjectIdentitySet(
                self._actor_net.trainable_weights + self._value_net.trainable_weights
            )
        )
        # Sort to ensure tensors on different processes end up in same order.
        variables_to_train = sorted(variables_to_train, key=lambda x: x.name)

        for i_epoch in range(self._num_epochs):
            with tf.name_scope("epoch_%d" % i_epoch):
                # Only save debug summaries for first and last epochs.
                debug_summaries = self._debug_summaries and (
                    i_epoch == 0 or i_epoch == self._num_epochs - 1
                )

                with tf.GradientTape() as tape:
                    loss_info = self.get_loss(
                        time_steps,
                        actions,
                        old_act_log_probs,
                        returns,
                        normalized_advantages,
                        old_action_distribution_parameters,
                        masked_weights,
                        self.train_step_counter,
                        debug_summaries,
                        old_value_predictions=old_value_predictions,
                        training=True,
                    )

                grads = tape.gradient(loss_info.loss, variables_to_train)
                if self._gradient_clipping > 0:
                    grads, _ = tf.clip_by_global_norm(grads, self._gradient_clipping)

                self._grad_norm = tf.linalg.global_norm(grads)

                # Tuple is used for py3, where zip is a generator producing values once.
                grads_and_vars = tuple(zip(grads, variables_to_train))

                # If summarize_gradients, create functions for summarizing both
                # gradients and variables.
                if self._summarize_grads_and_vars and debug_summaries:
                    eager_utils.add_gradients_summaries(
                        grads_and_vars, self.train_step_counter
                    )
                    eager_utils.add_variables_summaries(
                        grads_and_vars, self.train_step_counter
                    )

                self._optimizer.apply_gradients(grads_and_vars)
                self.train_step_counter.assign_add(1)

                policy_gradient_losses.append(loss_info.extra.policy_gradient_loss)
                value_estimation_losses.append(loss_info.extra.value_estimation_loss)
                l2_regularization_losses.append(loss_info.extra.l2_regularization_loss)
                entropy_regularization_losses.append(
                    loss_info.extra.entropy_regularization_loss
                )
                kl_penalty_losses.append(loss_info.extra.kl_penalty_loss)

        if self._initial_adaptive_kl_beta > 0:
            # After update epochs, update adaptive kl beta, then update observation
            #   normalizer and reward normalizer.
            policy_state = self._collect_policy.get_initial_state(batch_size)
            # Compute the mean kl from previous action distribution.
            kl_divergence = self._kl_divergence(
                time_steps,
                old_action_distribution_parameters,
                self._collect_policy.distribution(time_steps, policy_state).action,
            )
            kl_divergence *= masked_weights
            self.update_adaptive_kl_beta(kl_divergence)

        if self.update_normalizers_in_train:
            self.update_observation_normalizer(time_steps.observation)
            self.update_reward_normalizer(processed_experience.reward)

        loss_info = tf.nest.map_structure(tf.identity, loss_info)

        # Make summaries for total loss averaged across all epochs.
        # The *_losses lists will have been populated by
        #   calls to self.get_loss. Assumes all the losses have same length.
        with tf.name_scope("Losses/"):
            num_epochs = len(policy_gradient_losses)
            total_policy_gradient_loss = tf.add_n(policy_gradient_losses) / num_epochs
            total_value_estimation_loss = tf.add_n(value_estimation_losses) / num_epochs
            total_l2_regularization_loss = (
                tf.add_n(l2_regularization_losses) / num_epochs
            )
            total_entropy_regularization_loss = (
                tf.add_n(entropy_regularization_losses) / num_epochs
            )
            total_kl_penalty_loss = tf.add_n(kl_penalty_losses) / num_epochs
            tf.compat.v2.summary.scalar(
                name="policy_gradient_loss",
                data=total_policy_gradient_loss,
                step=self.train_step_counter,
            )
            tf.compat.v2.summary.scalar(
                name="value_estimation_loss",
                data=total_value_estimation_loss,
                step=self.train_step_counter,
            )
            tf.compat.v2.summary.scalar(
                name="l2_regularization_loss",
                data=total_l2_regularization_loss,
                step=self.train_step_counter,
            )
            tf.compat.v2.summary.scalar(
                name="entropy_regularization_loss",
                data=total_entropy_regularization_loss,
                step=self.train_step_counter,
            )
            tf.compat.v2.summary.scalar(
                name="kl_penalty_loss",
                data=total_kl_penalty_loss,
                step=self.train_step_counter,
            )

            total_abs_loss = (
                tf.abs(total_policy_gradient_loss)
                + tf.abs(total_value_estimation_loss)
                + tf.abs(total_entropy_regularization_loss)
                + tf.abs(total_l2_regularization_loss)
                + tf.abs(total_kl_penalty_loss)
            )

            tf.compat.v2.summary.scalar(
                name="total_abs_loss",
                data=total_abs_loss,
                step=self.train_step_counter,
            )

        with tf.name_scope("LearningRate/"):
            learning_rate = ppo_utils.get_learning_rate(self._optimizer)
            tf.compat.v2.summary.scalar(
                name="learning_rate", data=learning_rate, step=self.train_step_counter
            )

        if self._summarize_grads_and_vars and not tf.config.list_logical_devices("TPU"):
            with tf.name_scope("Variables/"):
                all_vars = (
                    self._actor_net.trainable_weights
                    + self._value_net.trainable_weights
                )
                for var in all_vars:
                    tf.compat.v2.summary.histogram(
                        name=var.name.replace(":", "_"),
                        data=var,
                        step=self.train_step_counter,
                    )

        return loss_info

    def value_estimation_loss(
        self,
        time_steps: ts.TimeStep,
        returns: types.Tensor,  # type: ignore
        weights: types.Tensor,  # type: ignore
        old_value_predictions: Optional[types.Tensor] = None,  # type: ignore
        debug_summaries: bool = False,
        training: bool = False,
    ) -> types.Tensor:  # type: ignore
        """Computes the value estimation loss for actor-critic training,
        adapted for the MOE-PPO approach.

        All tensors should have a single batch dimension.

        Args:
        time_steps: A batch of timesteps.
        returns: Per-timestep returns for value function to predict. (Should come
            from TD-lambda computation.) shape [T, B, d] (d = num objectives).
        weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.  Includes a mask for invalid timesteps.
        old_value_predictions: (Optional) The saved value predictions from
            policy_info, required when self._value_clipping > 0.
        debug_summaries: True if debug summaries should be created.
        training: Whether this loss is going to be used for training.

        Returns:
        value_estimation_loss: A scalar value_estimation_loss loss.

        Raises:
        ValueError: If old_value_predictions was not passed in, but value clipping
            was performed.
        """

        observation = time_steps.observation
        if debug_summaries and not tf.config.list_logical_devices("TPU"):
            observation_list = tf.nest.flatten(observation)
            show_observation_index = len(observation_list) != 1
            for i, single_observation in enumerate(observation_list):
                observation_name = (
                    "observations_{}".format(i)
                    if show_observation_index
                    else "observations"
                )
                tf.compat.v2.summary.histogram(
                    name=observation_name,
                    data=single_observation,
                    step=self.train_step_counter,
                )

        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        value_state = self._collect_policy.get_initial_value_state(batch_size)

        value_preds_tuple, _ = self._collect_policy.apply_value_network(
            time_steps.observation,
            time_steps.step_type,
            value_state=value_state,
            training=training,
        )
        subregion_value = value_preds_tuple[0]  # Scalar value, shape [T, B]
        global_value = value_preds_tuple[1]  # Vector value, shape [T, B, d]

        projected_global_value = tf.reduce_sum(
            global_value * self._weight_vector, axis=-1
        )

        projected_returns = tf.reduce_sum(returns * self._weight_vector, axis=-1)

        value_estimation_error_sub = tf.math.squared_difference(
            projected_returns, subregion_value
        )
        value_estimation_error_global = tf.math.squared_difference(
            subregion_value, projected_global_value
        )

        if self._value_clipping > 0:
            if old_value_predictions is None:
                raise ValueError(
                    "old_value_predictions is None but needed for value clipping."
                )

            clipped_value_preds = old_value_predictions + tf.clip_by_value(
                subregion_value - old_value_predictions,
                -self._value_clipping,
                self._value_clipping,
            )
            clipped_value_estimation_error = tf.math.squared_difference(
                projected_returns, clipped_value_preds
            )
            value_estimation_error_sub = tf.maximum(
                value_estimation_error_sub, clipped_value_estimation_error
            )

        total_value_estimation_error = (
            value_estimation_error_sub + value_estimation_error_global
        )

        if self._aggregate_losses_across_replicas:
            value_estimation_loss = (
                common.aggregate_losses(
                    per_example_loss=total_value_estimation_error, sample_weight=weights
                ).total_loss
                * self._value_pred_loss_coef
            )
        else:
            value_estimation_loss = (
                tf.math.reduce_mean(total_value_estimation_error * weights)
                * self._value_pred_loss_coef
            )

        if debug_summaries:
            tf.compat.v2.summary.scalar(
                name="value_pred_avg",
                data=tf.reduce_mean(input_tensor=subregion_value),
                step=self.train_step_counter,
            )
            tf.compat.v2.summary.scalar(
                name="value_actual_avg",
                data=tf.reduce_mean(input_tensor=projected_returns),
                step=self.train_step_counter,
            )
            tf.compat.v2.summary.scalar(
                name="value_estimation_loss",
                data=value_estimation_loss,
                step=self.train_step_counter,
            )
            if not tf.config.list_logical_devices("TPU"):
                tf.compat.v2.summary.histogram(
                    name="value_preds",
                    data=subregion_value,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="value_estimation_error",
                    data=total_value_estimation_error,
                    step=self.train_step_counter,
                )

        if self._check_numerics:
            value_estimation_loss = tf.debugging.check_numerics(
                value_estimation_loss, "value_estimation_loss"
            )

        return value_estimation_loss

    def _kl_divergence(
        self,
        time_steps,
        action_distribution_parameters,
        current_policy_distribution,
    ):
        outer_dims = list(
            range(nest_utils.get_outer_rank(time_steps, self.time_step_spec))
        )

        old_actions_distribution = ppo_utils.distribution_from_spec(
            self._action_distribution_spec,
            action_distribution_parameters,
            legacy_distribution_network=isinstance(
                self._actor_net, network.DistributionNetwork
            ),
        )

        independent_old_actions_distribution = tfp.distributions.Independent(
            old_actions_distribution,
            reinterpreted_batch_ndims=len(self.action_spec.shape),
        )

        kl_divergence = ppo_utils.nested_kl_divergence(
            independent_old_actions_distribution,
            current_policy_distribution,
            outer_dims=outer_dims,
        )
        return kl_divergence
