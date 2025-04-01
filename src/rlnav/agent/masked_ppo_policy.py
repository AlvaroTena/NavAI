from typing import Optional

import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.agents.ppo import ppo_policy, ppo_utils
from tf_agents.distributions import utils as distribution_utils
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from tf_agents.utils import common, tensor_normalizer


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
        time_step_spec: ts.TimeStep,
        action_spec: types.NestedTensorSpec,
        actor_network: network.Network,
        value_network: network.Network,
        observation_normalizer: Optional[tensor_normalizer.TensorNormalizer] = None,
        clip: bool = True,
        collect: bool = True,
        compute_value_and_advantage_in_train: bool = False,
        observation_and_action_constraint_splitter: Optional[types.Splitter] = None,
    ):
        if not isinstance(actor_network, network.Network):
            raise TypeError("actor_network is not of type network.Network")
        if not isinstance(value_network, network.Network):
            raise TypeError("value_network is not of type network.Network")

        actor_output_spec = actor_network.create_variables(time_step_spec.observation)

        value_output_spec = value_network.create_variables(time_step_spec.observation)

        # nest_utils.assert_value_spec(value_output_spec, 'value_network') # Disabled for multi-objective case

        distribution_utils.assert_specs_are_compatible(
            actor_output_spec,
            action_spec,
            "actor_network output spec does not match action spec",
        )

        self._compute_value_and_advantage_in_train = (
            compute_value_and_advantage_in_train
        )

        if collect:
            if isinstance(actor_network, network.DistributionNetwork):
                # Legacy DistributionNetwork case.  New code can just provide a regular
                # Network that emits a Distribution object; and we use a different
                # code path using DistributionSpecV2 for that.
                network_output_spec = actor_network.output_spec
                info_spec = {
                    "dist_params": tf.nest.map_structure(
                        lambda spec: spec.input_params_spec, network_output_spec
                    )
                }
            else:
                # We have a Network that emits a nest of distributions.
                def nested_dist_params(spec):
                    if not isinstance(spec, distribution_utils.DistributionSpecV2):
                        raise ValueError(
                            "Unexpected output from `actor_network`.  Expected "
                            "`Distribution` objects, but saw output spec: {}".format(
                                actor_output_spec
                            )
                        )
                    return distribution_utils.parameters_to_dict(
                        spec.parameters, tensors_only=True
                    )

                info_spec = {
                    "dist_params": tf.nest.map_structure(
                        nested_dist_params, actor_output_spec
                    )
                }

            if not self._compute_value_and_advantage_in_train:
                info_spec["value_prediction"] = tensor_spec.TensorSpec(
                    shape=[], dtype=tf.float32, name="value_prediction"
                )
        else:
            info_spec = ()

        policy_state_spec = {}
        if actor_network.state_spec:
            policy_state_spec["actor_network_state"] = actor_network.state_spec
        if (
            collect
            and value_network.state_spec
            and not self._compute_value_and_advantage_in_train
        ):
            policy_state_spec["value_network_state"] = value_network.state_spec
        if not policy_state_spec:
            policy_state_spec = ()

        super(ppo_policy.PPOPolicy, self).__init__(
            time_step_spec=time_step_spec,
            action_spec=action_spec,
            policy_state_spec=policy_state_spec,
            info_spec=info_spec,
            actor_network=actor_network,
            observation_normalizer=observation_normalizer,
            clip=clip,
        )

        self._collect = collect
        self._value_network = value_network
        self._observation_and_action_constraint_splitter = (
            observation_and_action_constraint_splitter
        )

    def _apply_actor_network(self, time_step, policy_state, training=False, mask=None):
        observation = time_step.observation
        if self._observation_normalizer:
            observation = self._observation_normalizer.normalize(observation)

        return self._actor_network(
            observation,
            time_step.step_type,
            network_state=policy_state,
            training=training,
            mask=mask,
        )

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

        observation_and_action_constraint_splitter = (
            self._observation_and_action_constraint_splitter
        )

        mask = None
        if observation_and_action_constraint_splitter is not None:
            _, mask = observation_and_action_constraint_splitter(time_step.observation)

        (distributions, new_policy_state["actor_network_state"]) = (
            self._apply_actor_network(
                time_step,
                policy_state["actor_network_state"],
                training=training,
                mask=mask,
            )
        )

        if self._collect:
            policy_info = {
                "dist_params": ppo_utils.get_distribution_params(
                    (
                        distributions
                        if not isinstance(distributions, tfp.distributions.Independent)
                        else distributions.distribution
                    ),
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
