from typing import Sequence

import tensorflow as tf
from tf_agents.environments.parallel_py_environment import (
    EnvConstructor,
    ParallelPyEnvironment,
)


class CascadeParallelPyEnvironment(ParallelPyEnvironment):
    """
    An asynchronous parallel environment that extends the base parallel environment.
    This class manages multiple simulation environments concurrently and allows for selective activation
    via a boolean mask. Only active environments are reset or stepped, and an error is raised if the
    number of actions does not match the number of active environments.
    """

    def __init__(
        self,
        env_constructors: Sequence[EnvConstructor],
        start_serially: bool = True,
        blocking: bool = False,
        flatten: bool = False,
    ):
        super(CascadeParallelPyEnvironment, self).__init__(
            env_constructors, start_serially, blocking, flatten
        )
        self._active_envs_mask = [False] * len(self._envs)

    @property
    def batch_size(self) -> int:
        return sum(1 for active in self._active_envs_mask if active)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def active_envs(self):
        return [
            env for env, active in zip(self._envs, self._active_envs_mask) if active
        ]

    def set_active_envs(self, active_mask: Sequence[bool]):
        """Update active environments based on the provided boolean mask.

        This method refreshes the internal active mask by activating any environments that transition
        from inactive to active. For each newly activated environment, it calls reset (resolving any
        promise if operating in non-blocking mode) and then stacks the resulting time steps.

        Args:
            active_mask (Sequence[bool]): Boolean mask specifying which environments should become active.

        Returns:
            List: A list of environments that were reset.

        Raises:
            ValueError: If the length of active_mask does not match the number of environments.
        """
        if len(active_mask) != len(self._envs):
            raise ValueError("Mask does not match the number of environments")

        new_active_mask = list(active_mask)
        new_active_indices = [
            i
            for i, (old, new) in enumerate(zip(self._active_envs_mask, new_active_mask))
            if (not old and new)
        ]

        new_promises = {
            idx: self._envs[idx].reset(self._blocking) for idx in new_active_indices
        }

        if not self._blocking:
            new_timesteps = {idx: promise() for idx, promise in new_promises.items()}
        else:
            new_timesteps = {idx: ts for idx, ts in new_promises.items()}

        self._active_envs_mask = new_active_mask
        new_ts_list = [new_timesteps[idx] for idx in sorted(new_timesteps.keys())]

        stacked_new_ts = self._stack_time_steps(new_ts_list)

        if self._current_time_step is None:
            self._current_time_step = stacked_new_ts
        else:
            self._merge_time_steps(self._current_time_step, stacked_new_ts)

        return [self._envs[idx] for idx in new_active_indices]

    def _reset(self):
        """
        Resets only the active environments.

        Calls reset on each active environment using the blocking flag and, if necessary, resolves any returned promise objects. The resulting time steps are then stacked and returned.

        Returns:
            A stacked collection of time steps from the active environments.
        """
        active_time_steps = [
            env.reset(self._blocking)
            for env, active in zip(self._envs, self._active_envs_mask)
            if active
        ]
        if not self._blocking:
            active_time_steps = [promise() for promise in active_time_steps]

        return self._stack_time_steps(active_time_steps)

    def _step(self, actions):
        """
        Performs a step in active environments using the provided actions.

        This method unpacks the actions corresponding to active environments, verifies that the number
        of actions matches the count of active environments, and applies each action to the respective
        environment. In non-blocking mode, any returned promise objects are resolved before stacking
        the time steps, which are then returned.

        Raises:
            ValueError: If the number of provided actions does not match the number of active environments.

        Returns:
            A stacked collection of time steps from the active environments.
        """
        active_actions = self._unstack_actions(actions)

        active_indices = [
            i for i, active in enumerate(self._active_envs_mask) if active
        ]
        if len(active_actions) != len(active_indices):
            raise ValueError(
                "The number of actions ({}) does not match the number of active environments ({}).".format(
                    len(active_actions), len(active_indices)
                )
            )

        time_steps = [
            self._envs[idx].step(action, self._blocking)
            for idx, action in zip(active_indices, active_actions)
        ]

        if not self._blocking:
            time_steps = [promise() for promise in time_steps]

        return self._stack_time_steps(time_steps)

    def _unstack_time_steps(self, batched_time_steps):
        """Unstacks batched time steps into a list of individual time steps.

        Flattens the nested structure of the input time steps and then reconstructs each
        time step based on the flatten flag. When flattening is enabled, returns a list
        of tuples; otherwise, repacks the flattened components to match the original
        structure.

        Args:
            batched_time_steps: A nested structure containing batched time step data.

        Returns:
            A list of unstacked time steps.
        """
        flattened = tf.nest.flatten(batched_time_steps)
        if self._flatten:
            unstacked = list(zip(*flattened))
        else:
            unstacked = [
                tf.nest.pack_sequence_as(batched_time_steps, ts)
                for ts in zip(*flattened)
            ]

        return unstacked

    def _merge_time_steps(self, current_ts, new_ts):
        """Merges two sets of time steps by unstacking, concatenating, and restacking them.

        Args:
            current_ts: The current collection of batched time steps.
            new_ts: The new collection of batched time steps to merge.

        Returns:
            A single, merged collection of time steps.
        """
        current_list = self._unstack_time_steps(current_ts)
        new_list = self._unstack_time_steps(new_ts)
        merged_list = current_list + new_list

        merged_stacked = self._stack_time_steps(merged_list)
        self._current_time_step = merged_stacked
