from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver, is_bandit_env
from tf_agents.utils import common


class DynamicStepDriverOpt(DynamicStepDriver):
    def __init__(
        self,
        env,
        policy,
        observers=None,
        transition_observers=None,
        num_steps=1,
    ):
        super(DynamicStepDriverOpt, self).__init__(
            env, policy, observers, transition_observers, num_steps
        )

        self._run_fn = common.function_in_tf1(reduce_retracing=True)(self._run)
