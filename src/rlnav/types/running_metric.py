from typing import Any, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


class RunningMetric:
    """
    A class for calculating running metrics (like moving averages) that efficiently
    handles both scalar values and n-dimensional arrays using NumPy.
    """

    def __init__(
        self,
        window_size: int = 100,
        decay: Optional[float] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Initialize the RunningMetric.

        Args:
            window_size (int): The number of recent elements to consider for calculating the average.
                               Use -1 for unlimited window size.
            decay (float, optional): Decay rate for weighting previous values (between 0 and 1).
                                    Higher values give more weight to recent values.
            shape (tuple, optional): Shape of the values if using array inputs.
                                    If None, shape will be determined from the first value.
        """
        self.window_size = window_size
        self.decay = decay
        self.shape = shape
        self.count = 0
        self.values = None
        self.current_idx = 0
        self._is_initialized = False

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    def _initialize(self, value: Union[float, np.ndarray]) -> None:
        """
        Initialize the values array based on the first input value.

        Args:
            value: The first value to determine the shape and dtype.
        """
        value_array = np.asarray(value)

        # If shape was not provided, determine it from the first value
        if self.shape is None:
            self.shape = value_array.shape

        # Create the values array with appropriate shape
        if self.window_size == -1:
            # Start with a reasonable size and expand as needed
            initial_size = 100
            if value_array.shape:  # For n-dimensional arrays
                self.values = np.zeros(
                    (initial_size,) + self.shape, dtype=value_array.dtype
                )
            else:  # For scalars
                self.values = np.zeros(initial_size, dtype=value_array.dtype)
        else:
            if value_array.shape:  # For n-dimensional arrays
                self.values = np.zeros(
                    (self.window_size,) + self.shape, dtype=value_array.dtype
                )
            else:  # For scalars
                self.values = np.zeros(self.window_size, dtype=value_array.dtype)

        self._is_initialized = True

    def _resize_if_needed(self) -> None:
        """
        Resize the values array if using unlimited window size and current array is full.
        """
        if self.window_size == -1 and self.count >= len(self.values):
            # Double the size of the array
            new_size = len(self.values) * 2
            if len(self.values.shape) > 1:  # For n-dimensional arrays
                new_values = np.zeros((new_size,) + self.shape, dtype=self.values.dtype)
                new_values[: self.count] = self.values[: self.count]
            else:  # For scalars
                new_values = np.zeros(new_size, dtype=self.values.dtype)
                new_values[: self.count] = self.values[: self.count]
            self.values = new_values

    def reset(self) -> None:
        """
        Reset the metric, clearing all stored values.
        """
        self.count = 0
        self.current_idx = 0
        if self._is_initialized:
            if self.window_size == -1:
                # Reinitialize with initial size
                if len(self.values.shape) > 1:  # For n-dimensional arrays
                    self.values = np.zeros((100,) + self.shape, dtype=self.values.dtype)
                else:  # For scalars
                    self.values = np.zeros(100, dtype=self.values.dtype)
            else:
                # Just zero out the array
                self.values.fill(0)

    def update(self, new_value: Union[float, np.ndarray]) -> None:
        """
        Add a new value and update the metric.

        Args:
            new_value: The new value to add (scalar or array).
        """
        new_value_array = np.asarray(new_value)

        # Initialize if this is the first update
        if not self._is_initialized:
            self._initialize(new_value_array)

        # Resize if needed (for unlimited window size)
        self._resize_if_needed()

        # Store the new value
        if self.window_size == -1:
            self.values[self.count] = new_value_array
            self.count += 1
        else:
            self.values[self.current_idx] = new_value_array
            self.current_idx = (self.current_idx + 1) % self.window_size
            self.count = min(self.count + 1, self.window_size)

    def get_values(self) -> np.ndarray:
        """
        Get the current stored values.

        Returns:
            np.ndarray: Array of current values.
        """
        if not self._is_initialized:
            return np.array([])

        if self.window_size == -1:
            return self.values[: self.count]

        if self.count < self.window_size:
            return self.values[: self.count]

        # Handle circular buffer when full
        if self.current_idx == 0:
            return self.values

        return np.concatenate(
            (self.values[self.current_idx :], self.values[: self.current_idx])
        )

    def get_running_value(self) -> Union[float, np.ndarray, None]:
        """
        Calculate the running average of stored values.

        Returns:
            The average of values in the window. Returns None if no values have been added.
        """
        if not self._is_initialized or self.count == 0:
            return None

        values = self.get_values()

        if self.decay:
            # Calculate weights with decay (newer values have higher weights)
            weights = np.array(
                [self.decay ** (self.count - 1 - i) for i in range(self.count)]
            )
            # Normalize weights
            weights = weights / np.sum(weights)

            # For scalar values
            if len(values.shape) == 1:
                return np.average(values, weights=weights)

            # For n-dimensional arrays, we need to apply weights along the first axis
            weighted_sum = np.zeros(self.shape, dtype=np.float64)
            for i in range(self.count):
                weighted_sum += weights[i] * values[i]

            return weighted_sum

        # Simple mean
        return np.mean(values, axis=0)

    def get_mean(self) -> Union[float, np.ndarray, None]:
        """
        Calculate the mean of stored values.

        Returns:
            The mean of values in the window. Returns None if no values have been added.
        """
        if not self._is_initialized or self.count == 0:
            return None

        values = self.get_values()

        # For scalar values
        if len(values.shape) == 1:
            return np.mean(values)

        # For n-dimensional arrays
        return np.mean(values, axis=0)

    def get_std(self) -> Union[float, np.ndarray, None]:
        """
        Calculate the standard deviation of stored values.

        Returns:
            The standard deviation of values in the window. Returns None if no values have been added.
        """
        if not self._is_initialized or self.count == 0:
            return None

        values = self.get_values()

        # For scalar values
        if len(values.shape) == 1:
            return np.std(values)

        # For n-dimensional arrays
        return np.std(values, axis=0)


class RunningDiffMetric(RunningMetric):
    """
    Tracks differences between consecutive values and maintains a cumulative sum.
    Extends RunningMetric to monitor changes in metrics over time.
    """

    def __init__(
        self,
        window_size: int = 100,
        decay: Optional[float] = None,
        shape: Optional[Tuple[int, ...]] = None,
    ):
        """
        Initialize the RunningDiffMetric.

        Args:
            window_size: Number of values to keep in history
            decay: Optional exponential weighting factor
            shape: Optional shape for input arrays
        """
        super().__init__(window_size, decay, shape)
        self.previous_value: Optional[NDArray] = None
        self.cumReward = np.zeros(self.shape, dtype=np.float64)

    def reset(self) -> None:
        """Reset the metric to its initial state."""
        super().reset()
        self.previous_value = None
        self.cumReward = np.zeros(self.shape, dtype=np.float64)

    def update(self, new_value: Union[float, NDArray]) -> None:
        """
        Update the metric with a new value.

        Args:
            new_value: The new value to process

        Returns:
            The result of the parent class update method
        """
        new_value_array = np.asarray(new_value)

        # Calculate difference or use the value itself if it's the first update
        differentiated_value = (
            new_value_array - self.previous_value
            if self.previous_value is not None
            else new_value_array
        )

        super().update(differentiated_value)
        self.previous_value = new_value_array
        self.cumReward += differentiated_value

    def get_differentiated_value(self) -> Union[float, NDArray]:
        """
        Returns the differentiated value.

        Returns:
            The last differentiated value.
        """
        if not self._is_initialized or self.count == 0:
            return np.array([])

        if self.window_size == -1:
            return self.values[self.count - 1]

        return self.values[(self.current_idx - 1) % self.window_size]

    def get_cumulative_value(self) -> Union[float, NDArray]:
        """
        Returns the cumulative value.

        Returns:
            The cumulative value.
        """
        return self.cumReward.copy()
