import os
from io import StringIO
from typing import Union

import dvc.api
import numpy as np
import pandas as pd

from navutils.logger import Logger
from rlnav.data.process import DataProcessor


class Reader:
    """Parse a txt file into a pandas dataframe and concatenate it with the global dataframe"""

    OUT_COLUMN_NAMES = [
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
        "delta_cmc",
        "crc",
        "delta_time",
    ]  # Attributes to print
    IN_COLUMN_NAMES = [
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
        "delta_cmc",
        "crc",
        "delta_time",
    ]  # Attributes to read
    IN_COLUMN_TYPES = [
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
        np.float32,
        np.float32,
        np.float32,
    ]  # Attributes to read

    @staticmethod
    def read_file(
        repo_dir,
        file_path,
    ):
        """Perform parsing, casting and concatenating"""
        file = os.path.join(repo_dir, file_path)

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.READER,
            f"Reading {file}",
        )

        in_column_names = Reader.IN_COLUMN_NAMES
        out_column_names = Reader.OUT_COLUMN_NAMES

        # Read txt and convert it to pandas dataframe
        current_data = pd.read_csv(
            file,
            sep="\s+",
            header=0,
            names=in_column_names,
            skip_blank_lines=True,
            skiprows=1,
            dtype=dict(zip(Reader.IN_COLUMN_NAMES, Reader.IN_COLUMN_TYPES)),
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

        return current_data

    @staticmethod
    def read_and_process(
        repo_dir,
        file_path,
        config_signals,
        drop_nan_rows: bool = True,
        drop_outliers: Union[bool, int, float] = False,
    ) -> pd.DataFrame:
        df = Reader.read_file(repo_dir, file_path)
        return DataProcessor(
            config_signals, drop_nan_rows, drop_outliers
        ).preprocess_batch(df)
