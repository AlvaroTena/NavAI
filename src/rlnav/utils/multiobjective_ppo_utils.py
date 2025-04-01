import tensorflow as tf
from tf_agents.trajectories import trajectory
from tf_agents.typing import types


def make_trajectory_mask(batched_traj: trajectory.Trajectory) -> types.Tensor:  # type: ignore
    """Mask boundary trajectories and those with invalid returns and advantages.

    Args:
      batched_traj: Trajectory, doubly-batched [batch_dim, time_dim,...]. It must
        be preprocessed already.

    Returns:
      A mask, type tf.float32, that is 0.0 for all between-episode Trajectory
        (batched_traj.step_type is LAST) and 0.0 if the return value is
        unavailable. The mask has shape [batch_dim, time_dim].
    """
    # 1.0 for all valid trajectories. 0.0 where between episodes.
    not_between_episodes = ~batched_traj.is_boundary()

    # 1.0 for trajectories with valid return values. 0.0 where return and
    # advantage are both 0 across all objectives. This happens to the last item
    # when the experience gets preprocessed, as insufficient information was
    # available for calculating multi-objective advantages.
    returns_all_zero = tf.reduce_all(
        tf.equal(batched_traj.policy_info["return"], 0), axis=-1
    )
    advantages_all_zero = tf.reduce_all(
        tf.equal(batched_traj.policy_info["advantage"], 0), axis=-1
    )
    valid_return_value = ~(returns_all_zero & advantages_all_zero)

    # Combine both conditions and convert to float32
    return tf.cast(not_between_episodes & valid_return_value, tf.float32)
