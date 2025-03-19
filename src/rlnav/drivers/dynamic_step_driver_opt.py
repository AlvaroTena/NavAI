import tensorflow as tf
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

    def _loop_condition_fn(self):
        """Returns a function with the condition needed for tf.while_loop."""

        def loop_cond(counter, *_):
            """Determines when to stop the loop, based on step counter.

            Args:
            counter: Step counters per batch index. Shape [batch_size] when
                batch_size > 1, else shape [].

            Returns:
            tf.bool tensor, shape (), indicating whether while loop should continue.
            """
            envs_done = all(
                [sub_env.call("is_done")() for sub_env in self.env.pyenv.active_envs]
            )

            if envs_done:
                return tf.constant(False)

            return tf.less(tf.reduce_sum(counter), self._num_steps)

        return loop_cond
