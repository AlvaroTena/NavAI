import tensorflow as tf


def discounted_return(
    rewards: tf.Tensor,
    discounts: tf.Tensor,
    final_value: tf.Tensor = None,
    time_major: bool = True,
    provide_all_returns: bool = True,
) -> tf.Tensor:
    """Computes discounted return for vectorized rewards.

    It assumes that `rewards` has shape [T, B, d] for multi-objective case.
    The discounted return is computed as follows:
    ```
    Q_n = sum_{n'=n}^N gamma^(n'-n) * r_{n'} + gamma^(N-n+1)*final_value.
    ```
    for each dimension `d` of the rewards tensor.

    Define abbreviations:
    `B`: batch size representing number of trajectories.
    `T`: number of steps per trajectory.  This is equal to `N - n` in the equation
         above.
    `d`: number of dimensions of the rewards tensor.

    Args:
      rewards: Tensor with shape `[T, B, d]` (or `[T, d]`) representing rewards.
      discounts: Tensor with shape `[T, B]` (or `[T]`) representing discounts. The
        discounts are expanded to shape `[T, B, 1]` for broadcasting with rewards.
      final_value: (Optional.).  Default: An all zeros tensor.  Tensor with shape
        `[B, d]` (or `[1]`) representing value estimate at `T`. This is optional;
        when set, it allows final value to bootstrap the reward computation.
      time_major: A boolean indicating whether input tensors are time major. False
        means input tensors have shape `[B, T]`.
      provide_all_returns: A boolean; if True, this will provide all of the
        returns by time dimension; if False, this will only give the single
        complete discounted return.

    Returns:
      If `provide_all_returns`:
        A tensor with shape `[T, B, d]` (or `[T, d]`) representing the discounted
        returns. The shape is `[B, T, d]` when `not time_major`.
      If `not provide_all_returns`:
        A tensor with shape `[B, d]` (or [d]) representing the discounted returns.
    """
    # Expand discounts dimension for broadcasting if rewards is 3D
    if discounts.shape.ndims == 2:
        discounts = tf.expand_dims(discounts, axis=-1)  # [T, B] -> [T, B, 1]

    # Convert to time-major format if needed
    if not time_major:
        with tf.name_scope("to_time_major_tensors"):
            discounts = tf.transpose(discounts, perm=[1, 0, 2])
            rewards = tf.transpose(rewards, perm=[1, 0, 2])

    # Default final value is zeros
    if final_value is None:
        final_value = tf.zeros_like(rewards[-1])

    # Helper function to compute accumulated discounted reward
    def discounted_return_fn(accumulated_discounted_reward, reward_discount):
        reward, discount = reward_discount
        return accumulated_discounted_reward * discount + reward

    if provide_all_returns:
        # Compute returns for all timesteps
        returns = tf.nest.map_structure(
            tf.stop_gradient,
            tf.scan(
                fn=discounted_return_fn,
                elems=(rewards, discounts),
                reverse=True,
                initializer=final_value,
            ),
        )

        # Convert back to batch-major if needed
        if not time_major:
            with tf.name_scope("to_batch_major_tensors"):
                returns = tf.transpose(returns, perm=[1, 0, 2])
    else:
        # Compute only the complete discounted return
        returns = tf.foldr(
            fn=discounted_return_fn,
            elems=(rewards, discounts),
            initializer=final_value,
            back_prop=False,
        )

    return tf.stop_gradient(returns)


def generalized_advantage_estimation(
    values, final_value, discounts, rewards, td_lambda=1.0, time_major=True
):
    """Computes generalized advantage estimation (GAE) for vectorized rewards.

    For theory, see
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    by John Schulman, Philipp Moritz et al.
    See https://arxiv.org/abs/1506.02438 for full paper.

    Define abbreviations:
      (B) batch size representing number of trajectories
      (T) number of steps per trajectory

    Args:
      values: Tensor with shape `[T, B, d]` representing value estimates.
      final_value: Tensor with shape `[B, d]` representing value estimate at t=T.
      discounts: Tensor with shape `[T, B]` representing discounts received by
        following the behavior policy.
      rewards: Tensor with shape `[T, B, d]` representing rewards received by
        following the behavior policy.
      td_lambda: A float32 scalar between [0, 1]. It's used for variance reduction
        in temporal difference.
      time_major: A boolean indicating whether input tensors are time major. False
        means input tensors have shape `[B, T, d]`.

    Returns:
      A tensor with shape `[T, B, d]` representing advantages. Shape is `[B, T, d]` when
      `not time_major`.
    """
    if discounts.shape.ndims == 2:
        discounts = tf.expand_dims(discounts, -1)

    if not time_major:
        with tf.name_scope("to_time_major_tensors"):
            rewards = tf.transpose(rewards, perm=[1, 0, 2])
            values = tf.transpose(values, perm=[1, 0, 2])
            discounts = tf.transpose(discounts, perm=[1, 0, 2])

    with tf.name_scope("gae"):
        next_values = tf.concat([values[1:], tf.expand_dims(final_value, 0)], axis=0)
        delta = rewards + discounts * next_values - values
        weighted_discounts = discounts * td_lambda

        def weighted_cumulative_td_fn(accumulated_td, reversed_weights_td_tuple):
            weighted_discount, td = reversed_weights_td_tuple
            return td + weighted_discount * accumulated_td

        advantages = tf.nest.map_structure(
            tf.stop_gradient,
            tf.scan(
                fn=weighted_cumulative_td_fn,
                elems=(weighted_discounts, delta),
                initializer=tf.zeros_like(final_value),
                reverse=True,
            ),
        )

    if not time_major:
        with tf.name_scope("to_batch_major_tensors"):
            advantages = tf.transpose(advantages, perm=[1, 0, 2])

    return tf.stop_gradient(advantages)
