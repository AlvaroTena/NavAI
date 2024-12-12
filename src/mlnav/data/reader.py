import datetime as dt
import os
import time
from typing import List

import numpy as np
import pandas as pd

from mlnav.utils.logger import Logger


class Reader:
    """Parse a txt file into a pandas dataframe and concatenate it with the global dataframe"""

    def __init__(self):
        self.OUT_COLUMN_NAMES = [
            "epoch",
            "sat_id",
            "freq",
            "code",
            "phase",
            "doppler",
            "snr",
            "elevation",
            "residual",
            "iono",
        ]  # Attributes to print
        self.IN_COLUMN_NAMES = [
            "Year",
            "Month",
            "Day",
            "Hours",
            "Minutes",
            "Seconds",
            "sat_id",
            "freq",
            "code",
            "phase",
            "doppler",
            "snr",
            "elevation",
            "residual",
            "iono",
        ]  # Attributes to read
        self.IN_COLUMN_TYPES = [
            np.uint16,
            np.uint8,
            np.uint8,
            np.uint8,
            np.uint8,
            np.float32,
            str,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
        ]  # Attributes to read
        self.output_dt: List[pd.DataFrame] = []  # Array containing all dataframes

    def __read_directory(self, file_path: str):
        """Perform parsing, casting and contcatenating"""

        Logger.getLogger().debug("Reading " + file_path)

        in_column_names = self.IN_COLUMN_NAMES
        out_column_names = self.OUT_COLUMN_NAMES

        with open(file_path) as f:
            labels = f.readline()

        if (
            set(out_column_names) == set(labels[1:].lower().split()[:-1])
            and "Multipath" in labels
        ):
            in_column_names += ["multipath"]
            out_column_names += ["multipath"]
            self.IN_COLUMN_TYPES += [np.float32]

        # Read txt and convert it to pandas dataframe
        current_data = pd.read_csv(
            file_path,
            sep="\s+",
            header=0,
            names=in_column_names,
            skip_blank_lines=True,
            skiprows=1,
            dtype=dict(zip(self.IN_COLUMN_NAMES, self.IN_COLUMN_TYPES)),
        )

        # Convert epoch in datetime
        current_data["epoch"] = pd.to_datetime(
            current_data[["Year", "Month", "Day", "Hours", "Minutes", "Seconds"]]
        )

        # Remove time columns
        current_data = current_data.drop(
            ["Year", "Month", "Day", "Hours", "Minutes", "Seconds"], axis=1
        )

        # Reorder columns to merge to global dataframe
        current_data = current_data[out_column_names]

        # Add dataframe to the array
        self.output_dt.append(current_data)

    def perform_reading(self, data_dir: str, n_files) -> List[pd.DataFrame]:
        """Perform reading for everyfile in the directory"""

        # get the start time
        st = time.time()
        Logger.getLogger().debug("Start reading...")

        files = sorted(
            [
                os.path.join(data_dir, filename)
                for filename in os.listdir(data_dir)
                if filename.endswith(".txt") and filename.startswith("AI_Multipath_")
            ]
        )
        for file in files[: len(files) if n_files == -1 else n_files]:
            self.__read_directory(file)

        # end time
        et = time.time()
        Logger.getLogger().info(
            f"Reading finished in {str(dt.timedelta(seconds=et-st))}"
        )

        return self.output_dt
