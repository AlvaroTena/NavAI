import io
from typing import Optional, Sequence, Text, Tuple

import tensorflow as tf
from absl import logging
from neptune import Run
from tf_agents.agents import data_converter, tf_agent
from tf_agents.agents.ppo import ppo_agent, ppo_utils
from tf_agents.environments import TFPyEnvironment
from tf_agents.networks import layer_utils, network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import (
    common,
    eager_utils,
    nest_utils,
    object_identity,
    tensor_normalizer,
)

import rlnav.agent.multivariatenormaldiag_distribution as multiDiagNormal
from rlnav.agent.ppo_networks import *
from rlnav.agent.ppo_policy import PPOPolicyMasked


def splitter(
    observation: types.NestedSpecTensorOrArray,
) -> Tuple[types.NestedSpecTensorOrArray, types.NestedSpecTensorOrArray]:
    """
    Splits the observation tensor into the observation itself and the mask.
    The mask is the [..., 0] component of the observation.

    Parameters:
    -----------
    observation : tf.Tensor or Nested structure
        The input observation tensor.

    Returns:
    --------
    Tuple[tf.Tensor or Nested structure, tf.Tensor]
        A tuple containing the observation and the mask.
        The mask is derived from the first component of the last dimension.
    """
    mask = observation[..., 0]

    return observation, mask


class PPOAgentMasked(ppo_agent.PPOAgent):
    """A PPO agent which mask actions based on satellite availability."""

    def __init__(
        self,
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        optimizer: Optional[types.Optimizer] = None,
        actor_net: Optional[network.Network] = None,
        value_net: Optional[network.Network] = None,
        greedy_eval: bool = True,
        importance_ratio_clipping: types.Float = 0.0,  # type: ignore
        lambda_value: types.Float = 0.95,  # type: ignore
        discount_factor: types.Float = 0.99,  # type: ignore
        entropy_regularization: types.Float = 0.0,  # type: ignore
        policy_l2_reg: types.Float = 0.0,  # type: ignore
        value_function_l2_reg: types.Float = 0.0,  # type: ignore
        shared_vars_l2_reg: types.Float = 0.0,  # type: ignore
        value_pred_loss_coef: types.Float = 0.5,  # type: ignore
        num_epochs: int = 25,
        use_gae: bool = False,
        use_td_lambda_return: bool = False,
        normalize_rewards: bool = True,
        reward_norm_clipping: types.Float = 10.0,  # type: ignore
        normalize_observations: bool = True,
        log_prob_clipping: types.Float = 0.0,  # type: ignore
        kl_cutoff_factor: types.Float = 2.0,  # type: ignore
        kl_cutoff_coef: types.Float = 1000.0,  # type: ignore
        initial_adaptive_kl_beta: types.Float = 1.0,  # type: ignore
        adaptive_kl_target: types.Float = 0.01,  # type: ignore
        adaptive_kl_tolerance: types.Float = 0.3,  # type: ignore
        gradient_clipping: Optional[types.Float] = None,  # type: ignore
        value_clipping: Optional[types.Float] = None,  # type: ignore
        check_numerics: bool = False,
        # TODO(b/150244758): Change the default to False once we move
        # clients onto Reverb.
        compute_value_and_advantage_in_train: bool = True,
        update_normalizers_in_train: bool = True,
        aggregate_losses_across_replicas: bool = True,
        debug_summaries: bool = False,
        summarize_grads_and_vars: bool = False,
        train_step_counter: Optional[tf.Variable] = None,
        name: Optional[Text] = None,
    ):
        """Creates a PPO agent.

        Args:
            time_step_spec: A `TimeStep` spec of the expected time_steps.
            action_spec: A nest of `BoundedTensorSpec` representing the actions.
            optimizer: Optimizer to use for the agent, default to using
                `tf.compat.v1.train.AdamOptimizer`.
            actor_net: A `network.DistributionNetwork` which maps observations to
                action distributions. Commonly, it is set to
                `actor_distribution_network.ActorDistributionNetwork`.
            value_net: A `Network` which returns the value prediction for input
                states, with `call(observation, step_type, network_state)`. Commonly, it
                is set to `value_network.ValueNetwork`.
            greedy_eval: Whether to use argmax/greedy action selection or sample from
                original action distribution for the evaluation policy. For environments
                such as ProcGen, stochastic is much better than greedy.
            importance_ratio_clipping: Epsilon in clipped, surrogate PPO objective.
                For more detail, see explanation at the top of the doc.
            lambda_value: Lambda parameter for TD-lambda computation.
            discount_factor: Discount factor for return computation. Default to `0.99`
                which is the value used for all environments from (Schulman, 2017).
            entropy_regularization: Coefficient for entropy regularization loss term.
                Default to `0.0` because no entropy bonus was used in (Schulman, 2017).
            policy_l2_reg: Coefficient for L2 regularization of unshared actor_net
                weights. Default to `0.0` because no L2 regularization was applied on
                the policy network weights in (Schulman, 2017).
            value_function_l2_reg: Coefficient for l2 regularization of unshared value
                function weights. Default to `0.0` because no L2 regularization was
                applied on the policy network weights in (Schulman, 2017).
            shared_vars_l2_reg: Coefficient for l2 regularization of weights shared
                between actor_net and value_net. Default to `0.0` because no L2
                regularization was applied on the policy network or value network
                weights in (Schulman, 2017).
            value_pred_loss_coef: Multiplier for value prediction loss to balance with
                policy gradient loss. Default to `0.5`, which was used for all
                environments in the OpenAI baseline implementation. This parameters is
                irrelevant unless you are sharing part of actor_net and value_net. In
                that case, you would want to tune this coeeficient, whose value depends
                on the network architecture of your choice.
            num_epochs: Number of epochs for computing policy updates. (Schulman,2017)
                sets this to 10 for Mujoco, 15 for Roboschool and 3 for Atari.
            use_gae: If True (default False), uses generalized advantage estimation
                for computing per-timestep advantage. Else, just subtracts value
                predictions from empirical return.
            use_td_lambda_return: If True (default False), uses td_lambda_return for
                training value function; here: `td_lambda_return = gae_advantage +
                value_predictions`. `use_gae` must be set to `True` as well to enable TD
                -lambda returns. If `use_td_lambda_return` is set to True while
                `use_gae` is False, the empirical return will be used and a warning will
                be logged.
            normalize_rewards: If true, keeps moving variance of rewards and
                normalizes incoming rewards. While not mentioned directly in (Schulman,
                2017), reward normalization was implemented in OpenAI baselines and
                (Ilyas et al., 2018) pointed out that it largely improves performance.
                You may refer to Figure 1 of https://arxiv.org/pdf/1811.02553.pdf for a
                comparison with and without reward scaling.
            reward_norm_clipping: Value above and below to clip normalized reward.
                Additional optimization proposed in (Ilyas et al., 2018) set to `5` or
                `10`.
            normalize_observations: If `True`, keeps moving mean and variance of
                observations and normalizes incoming observations. Additional
                optimization proposed in (Ilyas et al., 2018). If true, and the
                observation spec is not tf.float32 (such as Atari), please manually
                convert the observation spec received from the environment to tf.float32
                before creating the networks. Otherwise, the normalized input to the
                network (float32) will have a different dtype as what the network
                expects, resulting in a mismatch error.    Example usage: ```python
                observation_tensor_spec, action_spec, time_step_tensor_spec = (
                spec_utils.get_tensor_specs(env)) normalized_observation_tensor_spec =
                tf.nest.map_structure( lambda s: tf.TensorSpec( dtype=tf.float32,
                shape=s.shape, name=s.name ), observation_tensor_spec )    actor_net =
                actor_distribution_network.ActorDistributionNetwork(
                normalized_observation_tensor_spec, ...) value_net =
                value_network.ValueNetwork( normalized_observation_tensor_spec, ...) #
                Note that the agent still uses the original time_step_tensor_spec # from
                the environment. agent = ppo_clip_agent.PPOClipAgent(
                time_step_tensor_spec, action_spec, actor_net, value_net, ...) ```
            log_prob_clipping: +/- value for clipping log probs to prevent inf / NaN
                values.    Default: no clipping.
            kl_cutoff_factor: Only meaningful when `kl_cutoff_coef > 0.0`. A
                multiplier used for calculating the KL cutoff ( = `kl_cutoff_factor *
                adaptive_kl_target`). If policy KL averaged across the batch changes
                more than the cutoff, a squared cutoff loss would be added to the loss
                function.
            kl_cutoff_coef: kl_cutoff_coef and kl_cutoff_factor are additional params
                if one wants to use a KL cutoff loss term in addition to the adaptive KL
                loss term. Default to 0.0 to disable the KL cutoff loss term as this was
                not used in the paper.    kl_cutoff_coef is the coefficient to multiply by
                the KL cutoff loss term, before adding to the total loss function.
            initial_adaptive_kl_beta: Initial value for beta coefficient of adaptive
                KL penalty. This initial value is not important in practice because the
                algorithm quickly adjusts to it. A common default is 1.0.
            adaptive_kl_target: Desired KL target for policy updates. If actual KL is
                far from this target, adaptive_kl_beta will be updated. You should tune
                this for your environment. 0.01 was found to perform well for Mujoco.
            adaptive_kl_tolerance: A tolerance for adaptive_kl_beta. Mean KL above `(1
                + tol) * adaptive_kl_target`, or below `(1 - tol) * adaptive_kl_target`,
                will cause `adaptive_kl_beta` to be updated. `0.5` was chosen
                heuristically in the paper, but the algorithm is not very sensitive to
                it.
            gradient_clipping: Norm length to clip gradients.    Default: no clipping.
            value_clipping: Difference between new and old value predictions are
                clipped to this threshold. Value clipping could be helpful when training
                very deep networks. Default: no clipping.
            check_numerics: If true, adds `tf.debugging.check_numerics` to help find
                NaN / Inf values. For debugging only.
            compute_value_and_advantage_in_train: A bool to indicate where value
                prediction and advantage calculation happen.    If True, both happen in
                agent.train(). If False, value prediction is computed during data
                collection. This argument must be set to `False` if mini batch learning
                is enabled.
            update_normalizers_in_train: A bool to indicate whether normalizers are
                updated as parts of the `train` method. Set to `False` if mini batch
                learning is enabled, or if `train` is called on multiple iterations of
                the same trajectories. In that case, you would need to use `PPOLearner`
                (which updates all the normalizers outside of the agent). This ensures
                that normalizers are updated in the same way as (Schulman, 2017).
            aggregate_losses_across_replicas: only applicable to setups using multiple
                relicas. Default to aggregating across multiple cores using tf_agents.
                common.aggregate_losses. If set to `False`, use `reduce_mean` directly,
                which is faster but may impact learning results.
            debug_summaries: A bool to gather debug summaries.
            summarize_grads_and_vars: If true, gradient summaries will be written.
            train_step_counter: An optional counter to increment every time the train
                op is run.    Defaults to the global_step.
            name: The name of this agent. All variables in this module will fall under
                that name. Defaults to the class name.

        Raises:
            TypeError: if `actor_net` or `value_net` is not of type
                `tf_agents.networks.Network`.
        """
        if not isinstance(actor_net, network.Network):
            raise TypeError("actor_net must be an instance of a network.Network.")
        if not isinstance(value_net, network.Network):
            raise TypeError("value_net must be an instance of a network.Network.")

        # PPOPolicy validates these, so we skip validation here.
        actor_net.create_variables(time_step_spec.observation)
        value_net.create_variables(time_step_spec.observation)

        tf.Module.__init__(self, name=name)

        self._clip_fraction = 0.0
        self._grad_norm = 0.0
        self._optimizer = optimizer
        self._actor_net = actor_net
        self._value_net = value_net
        self._importance_ratio_clipping = importance_ratio_clipping
        self._lambda = lambda_value
        self._discount_factor = discount_factor
        self._entropy_regularization = entropy_regularization
        self._policy_l2_reg = policy_l2_reg
        self._value_function_l2_reg = value_function_l2_reg
        self._shared_vars_l2_reg = shared_vars_l2_reg
        self._value_pred_loss_coef = value_pred_loss_coef
        self._num_epochs = num_epochs
        self._use_gae = use_gae
        self._use_td_lambda_return = use_td_lambda_return
        self._reward_norm_clipping = reward_norm_clipping
        self._log_prob_clipping = log_prob_clipping
        self._kl_cutoff_factor = kl_cutoff_factor
        self._kl_cutoff_coef = kl_cutoff_coef
        self._adaptive_kl_target = adaptive_kl_target
        self._adaptive_kl_tolerance = adaptive_kl_tolerance
        self._gradient_clipping = gradient_clipping or 0.0
        self._value_clipping = value_clipping or 0.0
        self._check_numerics = check_numerics
        self._compute_value_and_advantage_in_train = (
            compute_value_and_advantage_in_train
        )
        self._aggregate_losses_across_replicas = aggregate_losses_across_replicas
        self.update_normalizers_in_train = update_normalizers_in_train
        if not isinstance(self._optimizer, tf.keras.optimizers.Optimizer):
            logging.warning(
                "Only tf.keras.optimizers.Optimiers are well supported, got a "
                "non-TF2 optimizer: %s",
                self._optimizer,
            )

        self._initial_adaptive_kl_beta = initial_adaptive_kl_beta
        if initial_adaptive_kl_beta > 0.0:
            # TODO(kbanoop): Rename create_variable.
            self._adaptive_kl_beta = common.create_variable(
                "adaptive_kl_beta", initial_adaptive_kl_beta, dtype=tf.float32
            )
        else:
            self._adaptive_kl_beta = None

        self._reward_normalizer = None
        if normalize_rewards:
            self._reward_normalizer = tensor_normalizer.StreamingTensorNormalizer(
                tensor_spec.TensorSpec([], tf.float32), scope="normalize_reward"
            )

        self._observation_normalizer = None
        if normalize_observations:
            self._observation_normalizer = tensor_normalizer.StreamingTensorNormalizer(
                time_step_spec.observation, scope="normalize_observations"
            )

        self._observation_and_action_constraint_splitter = splitter

        policy = PPOPolicyMasked(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
            observation_normalizer=self._observation_normalizer,
            clip=False,
            collect=False,
            observation_and_action_constraint_splitter=self._observation_and_action_constraint_splitter,
        )
        if greedy_eval:
            policy = greedy_policy.GreedyPolicy(policy)

        collect_policy = PPOPolicyMasked(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            actor_network=actor_net,
            value_network=value_net,
            observation_normalizer=self._observation_normalizer,
            clip=False,
            collect=True,
            compute_value_and_advantage_in_train=(
                self._compute_value_and_advantage_in_train
            ),
            observation_and_action_constraint_splitter=self._observation_and_action_constraint_splitter,
        )

        if isinstance(self._actor_net, network.DistributionNetwork):
            # Legacy behavior
            self._action_distribution_spec = self._actor_net.output_spec
        else:
            self._action_distribution_spec = self._actor_net.create_variables(
                time_step_spec.observation
            )

        # Set training_data_spec to collect_data_spec with augmented policy info,
        # iff return and normalized advantage are saved in preprocess_sequence.
        if self._compute_value_and_advantage_in_train:
            training_data_spec = None
        else:
            training_policy_info = collect_policy.trajectory_spec.policy_info.copy()
            training_policy_info.update(
                {
                    "value_prediction": collect_policy.trajectory_spec.policy_info[
                        "value_prediction"
                    ],
                    "return": tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
                    "advantage": tensor_spec.TensorSpec(shape=[], dtype=tf.float32),
                }
            )
            training_data_spec = collect_policy.trajectory_spec.replace(
                policy_info=training_policy_info
            )

        super(ppo_agent.PPOAgent, self).__init__(
            time_step_spec,
            action_spec,
            policy,
            collect_policy,
            train_sequence_length=None,
            training_data_spec=training_data_spec,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
        )

        # This must be built after super() which sets up self.data_context.
        self._collected_as_transition = data_converter.AsTransition(
            self.collect_data_context, squeeze_time_dim=False
        )

        self._as_trajectory = data_converter.AsTrajectory(
            self.data_context, sequence_length=None
        )

    def log_probability(self, distributions, actions, action_spec, mask=None):
        """Computes log probability of actions given distribution, applying a mask.

        Args:
            distributions: A possibly batched tuple of distributions.
            actions: A possibly batched action tuple.
            action_spec: A nested tuple representing the action spec.
            mask: Optional mask to apply to the log probability per action dimension.

        Returns:
            A Tensor representing the log probability of each action in the batch.
        """
        outer_rank = nest_utils.get_outer_rank(actions, action_spec)

        def _compute_log_prob(single_distribution, single_action, single_mask):
            epsilon = 1e-6
            # Inverse transformation on actions
            single_action = single_distribution.bijector.inverse(single_action)
            single_action = tf.clip_by_value(single_action, epsilon, 1 - epsilon)

            # Compute log_prob per dimension
            single_log_prob = single_distribution.distribution.distribution.log_prob(
                single_action
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
            rank = single_log_prob.shape.rank
            reduce_dims = list(range(outer_rank, rank))
            return tf.reduce_sum(single_log_prob, axis=reduce_dims)

        nest_utils.assert_same_structure(distributions, actions)
        flat_distributions = tf.nest.flatten(distributions)
        flat_actions = tf.nest.flatten(actions)
        flat_masks = (
            tf.nest.flatten(mask) if mask is not None else [None] * len(flat_actions)
        )

        log_probs = [
            _compute_log_prob(dist, action, m)
            for (dist, action, m) in zip(flat_distributions, flat_actions, flat_masks)
        ]

        total_log_probs = tf.add_n(log_probs)
        return total_log_probs

    def entropy(self, distributions, action_spec, outer_rank=None, mask=None):
        """Computes total entropy of distribution, applying a mask.

        Args:
            distributions: A possibly batched tuple of distributions.
            action_spec: A nested tuple representing the action spec.
            outer_rank: Optional outer rank of the distributions. If not provided use
            distribution.mode() to compute it.
            mask: Optional mask to apply to the entropy per action dimension.

        Returns:
            A Tensor representing the entropy of each distribution in the batch.
            Assumes actions are independent, so that marginal entropies of each action
            may be summed.
        """
        if outer_rank is None:
            nested_modes = tf.nest.map_structure(lambda d: d.mode(), distributions)
            outer_rank = nest_utils.get_outer_rank(nested_modes, action_spec)

        def _compute_entropy(single_distribution, single_mask):
            try:
                # Compute entropy per dimension
                entropy_per_dim = (
                    single_distribution.distribution.distribution.entropy()
                )

                # Apply mask if provided
                if single_mask is not None:
                    entropy_per_dim *= single_mask

                # Sum entropies over action dimensions
                rank = entropy_per_dim.shape.rank
                reduce_dims = list(range(outer_rank, rank))
                return tf.reduce_sum(entropy_per_dim, axis=reduce_dims)
            except NotImplementedError:
                return None

        nest_utils.assert_same_structure(distributions, action_spec)
        flat_distributions = tf.nest.flatten(distributions)
        flat_masks = (
            tf.nest.flatten(mask)
            if mask is not None
            else [None] * len(flat_distributions)
        )

        entropies = []
        for dist, m in zip(flat_distributions, flat_masks):
            entropy_dist = _compute_entropy(dist, m)
            if entropy_dist is not None:
                entropies.append(entropy_dist)

        # Sum entropies over action tuple.
        if not entropies:
            return None

        return tf.add_n(entropies)

    def nested_kl_divergence(
        self,
        nested_from_distribution: types.NestedDistribution,
        nested_to_distribution: types.NestedDistribution,
        outer_dims: Sequence[int] = (),
        mask: Optional[types.Tensor] = None,  # type: ignore
    ) -> types.Tensor:  # type: ignore
        """Given two nested distributions, sum the KL divergences of the leaves, applying a mask."""
        nest_utils.assert_same_structure(
            nested_from_distribution, nested_to_distribution
        )

        # Make list pairs of leaf distributions.
        flat_from_distribution = tf.nest.flatten(nested_from_distribution)
        flat_to_distribution = tf.nest.flatten(nested_to_distribution)
        flat_masks = (
            tf.nest.flatten(mask)
            if mask is not None
            else [None] * len(flat_from_distribution)
        )

        all_kl_divergences = []
        for from_dist, to_dist, m in zip(
            flat_from_distribution, flat_to_distribution, flat_masks
        ):
            kl_divergence = from_dist.distribution.distribution.kl_divergence(
                to_dist.distribution.distribution
            )
            if m is not None:
                kl_divergence *= m  # Apply mask

            # Reduce_sum over non-batch dimensions.
            reduce_dims = list(range(len(kl_divergence.shape)))
            for dim in outer_dims:
                reduce_dims.remove(dim)
            kl_divergence_reduced = tf.reduce_sum(
                input_tensor=kl_divergence, axis=reduce_dims
            )
            all_kl_divergences.append(kl_divergence_reduced)

        # Sum the kl of the leaves.
        total_kl = tf.add_n(all_kl_divergences)

        return total_kl

    def get_loss(
        self,
        time_steps: ts.TimeStep,
        actions: types.NestedTensorSpec,
        act_log_probs: types.Tensor,  # type: ignore
        returns: types.Tensor,  # type: ignore
        normalized_advantages: types.Tensor,  # type: ignore
        action_distribution_parameters: types.NestedTensor,
        weights: types.Tensor,  # type: ignore
        train_step: tf.Variable,
        debug_summaries: bool,
        old_value_predictions: Optional[types.Tensor] = None,  # type: ignore
        training: bool = False,
    ) -> tf_agent.LossInfo:
        """Compute the loss and create optimization op for one training epoch.

        All tensors should have a single batch dimension.

        Args:
            time_steps: A minibatch of TimeStep tuples.
            actions: A minibatch of actions.
            act_log_probs: A minibatch of action probabilities (probability under the
                sampling policy).
            returns: A minibatch of per-timestep returns.
            normalized_advantages: A minibatch of normalized per-timestep advantages.
            action_distribution_parameters: Parameters of data-collecting action
                distribution. Needed for KL computation.
            weights: Optional scalar or element-wise (per-batch-entry) importance
                weights.    Includes a mask for invalid timesteps.
            train_step: A train_step variable to increment for each train step.
                Typically the global_step.
            debug_summaries: True if debug summaries should be created.
            old_value_predictions: (Optional) The saved value predictions, used for
                calculating the value estimation loss when value clipping is performed.
            training: Whether this loss is being used for training.

        Returns:
            A tf_agent.LossInfo named tuple with the total_loss and all intermediate
                losses in the extra field contained in a PPOLossInfo named tuple.
        """
        # Evaluate the current policy on timesteps.

        # batch_size from time_steps
        batch_size = nest_utils.get_outer_shape(time_steps, self._time_step_spec)[0]
        policy_state = self._collect_policy.get_initial_state(batch_size)
        # We must use _distribution because the distribution API doesn't pass down
        # the training= kwarg.
        distribution_step = (
            self._collect_policy._distribution(  # pylint: disable=protected-access
                time_steps, policy_state, training=training
            )
        )
        # TODO(eholly): Rename policy distributions to something clear and uniform.
        current_policy_distribution = distribution_step.action

        # Call all loss functions and add all loss values.
        value_estimation_loss = self.value_estimation_loss(
            time_steps=time_steps,
            returns=returns,
            old_value_predictions=old_value_predictions,
            weights=weights,
            debug_summaries=debug_summaries,
            training=training,
        )
        policy_gradient_loss = self.policy_gradient_loss(
            time_steps,
            actions,
            tf.stop_gradient(act_log_probs),
            tf.stop_gradient(normalized_advantages),
            current_policy_distribution,
            weights,
            debug_summaries=debug_summaries,
        )

        if (
            self._policy_l2_reg > 0.0
            or self._value_function_l2_reg > 0.0
            or self._shared_vars_l2_reg > 0.0
        ):
            l2_regularization_loss = self.l2_regularization_loss(debug_summaries)
        else:
            l2_regularization_loss = tf.zeros_like(policy_gradient_loss)

        with tf.name_scope("entropy_regularization"):
            outer_rank = nest_utils.get_outer_rank(time_steps, self.time_step_spec)

            _, availability_indicator = (
                self._observation_and_action_constraint_splitter(time_steps.observation)
            )
            action_mask = tf.cast(availability_indicator, tf.float32)  # Cast to float

            entropy = self.entropy(
                current_policy_distribution,
                self.action_spec,
                outer_rank,
                mask=action_mask,
            )
            if entropy is None:
                entropy = tf.zeros_like(weights, tf.float32)
            else:
                entropy = tf.cast(entropy, tf.float32)
        if self._entropy_regularization > 0.0:
            entropy_regularization_loss = self.entropy_regularization_loss(
                time_steps, entropy, weights, debug_summaries
            )
        else:
            entropy_regularization_loss = tf.zeros_like(policy_gradient_loss)

        with tf.name_scope("Losses/"):
            tf.compat.v2.summary.scalar(
                name="entropy",
                data=tf.reduce_mean(entropy * weights),
                step=self.train_step_counter,
            )

        # TODO(b/161365079): Move this logic to PPOKLPenaltyAgent.
        if self._initial_adaptive_kl_beta == 0 and self._kl_cutoff_factor == 0:
            kl_penalty_loss = tf.zeros_like(policy_gradient_loss)
        else:
            kl_penalty_loss = self.kl_penalty_loss(
                time_steps,
                action_distribution_parameters,
                current_policy_distribution,
                weights,
                debug_summaries,
            )

        total_loss = (
            policy_gradient_loss
            + value_estimation_loss
            + l2_regularization_loss
            + entropy_regularization_loss
            + kl_penalty_loss
        )

        return tf_agent.LossInfo(
            total_loss,
            ppo_agent.PPOLossInfo(
                policy_gradient_loss=policy_gradient_loss,
                value_estimation_loss=value_estimation_loss,
                l2_regularization_loss=l2_regularization_loss,
                entropy_regularization_loss=entropy_regularization_loss,
                kl_penalty_loss=kl_penalty_loss,
                clip_fraction=self._clip_fraction,
            ),
        )

    def _train(self, experience, weights):
        if self._optimizer is None:
            raise ValueError("Optimizer is undefined.")

        experience = self._as_trajectory(experience)

        if self._compute_value_and_advantage_in_train:
            processed_experience = self._preprocess(experience)
        else:
            processed_experience = experience

        # Mask trajectories that cannot be used for training.
        valid_mask = ppo_utils.make_trajectory_mask(processed_experience)
        if weights is None:
            masked_weights = valid_mask
        else:
            masked_weights = weights * valid_mask

        # Reconstruct per-timestep policy distribution from stored distribution
        #     parameters.
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

        _, availability_indicator = self._observation_and_action_constraint_splitter(
            processed_experience.observation
        )
        action_mask = tf.cast(availability_indicator, tf.float32)  # Cast to float

        # Compute log probability of actions taken during data collection, using the
        #     collect policy distribution.
        old_act_log_probs = self.log_probability(
            old_actions_distribution,
            processed_experience.action,
            self._action_spec,
            action_mask,
        )

        # TODO(b/171573175): remove the condition once histograms are
        # supported on TPUs.
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

        normalized_advantages = ppo_agent._normalize_advantages(
            advantages, variance_epsilon=1e-8
        )

        # TODO(b/171573175): remove the condition once histograms are
        # supported on TPUs.
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

        # TODO(b/1613650790): Move this logic to PPOKLPenaltyAgent.
        if self._initial_adaptive_kl_beta > 0:
            # After update epochs, update adaptive kl beta, then update observation
            #     normalizer and reward normalizer.
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
        #     calls to self.get_loss. Assumes all the losses have same length.
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

        # TODO(b/171573175): remove the condition once histograms are
        # supported on TPUs.
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

    def policy_gradient_loss(
        self,
        time_steps: ts.TimeStep,
        actions: types.NestedTensor,
        sample_action_log_probs: types.Tensor,  # type: ignore
        advantages: types.Tensor,  # type: ignore
        current_policy_distribution: types.NestedDistribution,
        weights: types.Tensor,  # type: ignore
        debug_summaries: bool = False,
    ) -> types.Tensor:  # type: ignore
        """Create tensor for policy gradient loss.

        All tensors should have a single batch dimension.

        Args:
            time_steps: TimeSteps with observations for each timestep.
            actions: Tensor of actions for timesteps, aligned on index.
            sample_action_log_probs: Tensor of sample probability of each action.
            advantages: Tensor of advantage estimate for each timestep, aligned on
                index. Works better when advantage estimates are normalized.
            current_policy_distribution: The policy distribution, evaluated on all
                time_steps.
            weights: Optional scalar or element-wise (per-batch-entry) importance
                weights.    Includes a mask for invalid timesteps.
            debug_summaries: True if debug summaries should be created.

        Returns:
            policy_gradient_loss: A tensor that will contain policy gradient loss for
                the on-policy experience.
        """
        nest_utils.assert_same_structure(time_steps, self.time_step_spec)

        _, availability_indicator = self._observation_and_action_constraint_splitter(
            time_steps.observation
        )
        action_mask = tf.cast(availability_indicator, tf.float32)  # Cast to float

        action_log_prob = self.log_probability(
            current_policy_distribution, actions, self._action_spec, action_mask
        )
        action_log_prob = tf.cast(action_log_prob, tf.float32)
        if self._log_prob_clipping > 0.0:
            action_log_prob = tf.clip_by_value(
                action_log_prob, -self._log_prob_clipping, self._log_prob_clipping
            )
        if self._check_numerics:
            action_log_prob = tf.debugging.check_numerics(
                action_log_prob, "action_log_prob"
            )

        # Prepare both clipped and unclipped importance ratios.
        importance_ratio = tf.exp(action_log_prob - sample_action_log_probs)
        importance_ratio_clipped = tf.clip_by_value(
            importance_ratio,
            1 - self._importance_ratio_clipping,
            1 + self._importance_ratio_clipping,
        )

        if self._check_numerics:
            importance_ratio = tf.debugging.check_numerics(
                importance_ratio, "importance_ratio"
            )
            if self._importance_ratio_clipping > 0.0:
                importance_ratio_clipped = tf.debugging.check_numerics(
                    importance_ratio_clipped, "importance_ratio_clipped"
                )

        # Pessimistically choose the minimum objective value for clipped and
        #     unclipped importance ratios.
        per_timestep_objective = importance_ratio * advantages
        per_timestep_objective_clipped = importance_ratio_clipped * advantages
        per_timestep_objective_min = tf.minimum(
            per_timestep_objective, per_timestep_objective_clipped
        )

        if self._importance_ratio_clipping > 0.0:
            policy_gradient_loss = -per_timestep_objective_min
        else:
            policy_gradient_loss = -per_timestep_objective

        if self._aggregate_losses_across_replicas:
            policy_gradient_loss = common.aggregate_losses(
                per_example_loss=policy_gradient_loss, sample_weight=weights
            ).total_loss
        else:
            policy_gradient_loss = tf.math.reduce_mean(policy_gradient_loss * weights)

        if self._importance_ratio_clipping > 0.0:
            self._clip_fraction = tf.reduce_mean(
                input_tensor=tf.cast(
                    tf.greater(
                        tf.abs(importance_ratio - 1.0),
                        self._importance_ratio_clipping,
                    ),
                    tf.float32,
                )
            )

        if debug_summaries:
            if self._importance_ratio_clipping > 0.0:
                tf.compat.v2.summary.scalar(
                    name="clip_fraction",
                    data=self._clip_fraction,
                    step=self.train_step_counter,
                )
            tf.compat.v2.summary.scalar(
                name="importance_ratio_mean",
                data=tf.reduce_mean(input_tensor=importance_ratio),
                step=self.train_step_counter,
            )
            entropy = self.entropy(
                current_policy_distribution, self.action_spec, mask=action_mask
            )
            tf.compat.v2.summary.scalar(
                name="policy_entropy_mean",
                data=tf.reduce_mean(input_tensor=entropy),
                step=self.train_step_counter,
            )
            # TODO(b/171573175): remove the condition once histograms are supported
            # on TPUs.
            if not tf.config.list_logical_devices("TPU"):
                tf.compat.v2.summary.histogram(
                    name="action_log_prob",
                    data=action_log_prob,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="action_log_prob_sample",
                    data=sample_action_log_probs,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="importance_ratio",
                    data=importance_ratio,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="importance_ratio_clipped",
                    data=importance_ratio_clipped,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="per_timestep_objective",
                    data=per_timestep_objective,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="per_timestep_objective_clipped",
                    data=per_timestep_objective_clipped,
                    step=self.train_step_counter,
                )
                tf.compat.v2.summary.histogram(
                    name="per_timestep_objective_min",
                    data=per_timestep_objective_min,
                    step=self.train_step_counter,
                )

                tf.compat.v2.summary.histogram(
                    name="policy_entropy", data=entropy, step=self.train_step_counter
                )
                for i, (single_action, single_distribution) in enumerate(
                    zip(
                        tf.nest.flatten(self.action_spec),
                        tf.nest.flatten(current_policy_distribution),
                    )
                ):
                    # Categorical distribution (used for discrete actions) doesn't have a
                    # mean.
                    distribution_index = "_{}".format(i) if i > 0 else ""
                    if not tensor_spec.is_discrete(single_action):
                        tf.compat.v2.summary.histogram(
                            name="actions_distribution_mean" + distribution_index,
                            data=single_distribution.mean(),
                            step=self.train_step_counter,
                        )
                        tf.compat.v2.summary.histogram(
                            name="actions_distribution_stddev" + distribution_index,
                            data=single_distribution.stddev(),
                            step=self.train_step_counter,
                        )
                tf.compat.v2.summary.histogram(
                    name="policy_gradient_loss",
                    data=policy_gradient_loss,
                    step=self.train_step_counter,
                )

        if self._check_numerics:
            policy_gradient_loss = tf.debugging.check_numerics(
                policy_gradient_loss, "policy_gradient_loss"
            )

        return policy_gradient_loss

    def kl_penalty_loss(
        self,
        time_steps: ts.TimeStep,
        action_distribution_parameters: types.NestedTensor,
        current_policy_distribution: types.NestedDistribution,
        weights: types.Tensor,  # type: ignore
        debug_summaries: bool = False,
    ) -> types.Tensor:  # type: ignore
        """Compute a loss that penalizes policy steps with high KL."""
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

        # Generar y ajustar la máscara
        _, availability_indicator = self._observation_and_action_constraint_splitter(
            time_steps.observation
        )
        mask = tf.cast(availability_indicator, tf.float32)

        kl_divergence = self.nested_kl_divergence(
            old_actions_distribution,
            current_policy_distribution,
            outer_dims=outer_dims,
            mask=mask,
        )
        kl_divergence *= weights

        # TODO(b/171573175): remove the condition once histograms are supported
        # on TPUs.
        if debug_summaries and not tf.config.list_logical_devices("TPU"):
            tf.compat.v2.summary.histogram(
                name="kl_divergence", data=kl_divergence, step=self.train_step_counter
            )

        kl_cutoff_loss = self.kl_cutoff_loss(kl_divergence, debug_summaries)
        adaptive_kl_loss = self.adaptive_kl_loss(kl_divergence, debug_summaries)
        return tf.add(kl_cutoff_loss, adaptive_kl_loss, name="kl_penalty_loss")


def create_ppo_agent(
    train_env: TFPyEnvironment, npt_run: Run, rnn=True, learning_rate=1e-3
):
    optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0, dtype=tf.int64)

    if rnn:
        actor_net = create_actor_rnn_net(
            train_env.observation_spec(), train_env.action_spec()
        )
        value_net = create_value_rnn_net(train_env.observation_spec())
    else:
        actor_net = create_actor_net(
            train_env.observation_spec(), train_env.action_spec()
        )
        value_net = create_value_net(train_env.observation_spec())

    agent = PPOAgentMasked(
        time_step_spec=train_env.time_step_spec(),
        action_spec=train_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=10,
        train_step_counter=train_step_counter,
        entropy_regularization=0.1,
        normalize_rewards=True,
        use_gae=True,
        debug_summaries=True,
        # summarize_grads_and_vars=True,
    )

    agent.initialize()

    summary = {}
    with io.StringIO() as s:
        layer_utils.print_summary(
            actor_net,
            print_fn=lambda x, **kwargs: s.write(x + "\n"),
            expand_nested=True,
        )
        model_summary = s.getvalue()
    summary["agent/actor_model"] = model_summary
    with io.StringIO() as s:
        layer_utils.print_summary(
            value_net,
            print_fn=lambda x, **kwargs: s.write(x + "\n"),
            expand_nested=True,
        )
        model_summary = s.getvalue()
    summary["agent/value_model"] = model_summary

    npt_run["training"] = summary

    return agent
