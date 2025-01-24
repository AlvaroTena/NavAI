from typing import Optional

from tf_agents.agents.ppo import ppo_policy, ppo_utils
from tf_agents.networks import network
from tf_agents.trajectories import policy_step
from tf_agents.typing import types
from tf_agents.utils import common


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
        network_observation = time_step.observation
        mask = None

        if observation_and_action_constraint_splitter is not None:
            network_observation, mask = observation_and_action_constraint_splitter(
                network_observation
            )

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
