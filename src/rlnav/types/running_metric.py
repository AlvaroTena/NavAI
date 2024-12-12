import numpy as np


class RunningMetric:
    def __init__(self, window_size=100, decay=None):
        """
        Initializes the class to calculate the running metric.

        Args:
            window_size (int): The number of recent elements to consider for calculating the average.
            decay (float): Decay rate of the previous values.
        """
        self.window_size = window_size
        self.decay = decay
        self.values = []

    def reset(self):
        self.values = []

    def update(self, new_value):
        """
        Adds a new value and updates the moving average.

        Args:
            new_value (float): the new value to add.
        """
        self.values.append(new_value)
        if self.window_size != -1 and len(self.values) > self.window_size:
            self.values.pop(0)

    def get_running_value(self):
        """
        Returns the value of the moving average.

        Returns:
            float: the average of the last values in the window.
        """
        if self.decay:
            weights = np.array([self.decay**i for i in range(len(self.values))])
            return np.average(self.values, weights=weights)
        return np.mean(self.values)
