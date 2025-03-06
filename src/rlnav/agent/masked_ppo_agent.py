from typing import Optional, Text, Tuple

import tensorflow as tf
from absl import logging
from rlnav.agent.masked_ppo_policy import PPOPolicyMasked
from tf_agents.agents import data_converter
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import network
from tf_agents.policies import greedy_policy
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, tensor_normalizer


def build_mask_for_heads(mask_bin: tf.Tensor) -> tf.Tensor:
    """
    Constructs a mask for each satellite head based on the input binary mask.

    Args:
        mask_bin (tf.Tensor): A tensor of shape (B, 286) or (B, T, 286) with 0/1
                               indicating if satellite i is available.
    Returns:
        tf.Tensor: A tensor of shape (B, 286, 2) or (B, T, 286, 2).
                   The last dimension contains [True, mask_value] for each satellite.

    Raises:
        ValueError: If mask_bin is not of rank 2 or 3.
    """
    rank = tf.rank(mask_bin)

    if rank not in (2, 3):
        raise ValueError("mask_bin must be of rank 2 or 3.")

    # Convert binary mask to boolean and add a dimension
    valid_sat = tf.cast(mask_bin, tf.bool)[
        ..., tf.newaxis
    ]  # (B, 286, 1) or (B, T, 286, 1)

    # Create tensor of all True values with same shape
    all_true = tf.ones_like(valid_sat, dtype=tf.bool)

    # Concatenate to create final mask
    return tf.concat([all_true, valid_sat], axis=-1)


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

    return observation, build_mask_for_heads(mask)


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
                time_step_spec.reward, scope="normalize_reward"
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
