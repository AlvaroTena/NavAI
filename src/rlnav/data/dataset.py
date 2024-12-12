import copy
import time
from collections import defaultdict

import dill as pickle
import numpy as np
import pandas as pd
from tf_agents.specs import array_spec

import pewrapper.types.constants as pe_const
import rlnav.types.constants as const
from rlnav.data.process import DataProcessor
from rlnav.data.transform import ColumnTransformer
from rlnav.types.utils import get_global_sat_idx


class RLDataset:

    def __init__(self, config_signals, transformer_path, window_size=1):
        """
        Initializes the dataset with the ColumnTransformer and the desired sequence length.

        Parameters:
        - transformer_path: Path to the .pkl file of the pretrained ColumnTransformer.
        - window_size: Size of the temporary window for sequences.
        """
        self.processor = DataProcessor(config_signals)
        with open(transformer_path, "rb") as transformer:
            self.transformer: ColumnTransformer = pickle.load(transformer)
        self.window_size = window_size

        self.data_buffer = pd.DataFrame()
        self.reset_times()

    def reset_times(self):
        self._times = {}
        self._times["process_data"] = []

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        times.update(self.processor.get_times())
        return times

    def process_data(
        self, raw_dataframe: pd.DataFrame, observation_spec: array_spec.BoundedArraySpec
    ):
        """
        Processes and transforms the data in the buffer.
        """
        start = time.time()
        if not raw_dataframe.empty:
            self.ingest_data(raw_dataframe)

        # Transformar los datos
        transformed_data = self.transform_features()

        observation = self.transform_to_observation_spec(
            transformed_data, observation_spec
        )

        self._times["process_data"].append(time.time() - start)
        return observation

    def ingest_data(self, dataframe: pd.DataFrame):
        """
        Ingests new data, preparing it for transformation.

        Parameters:
        - dataframe: pandas DataFrame with the new data.
        """
        processed_data = self.processor.preprocess_batch(dataframe, verbose=False)

        if not processed_data.empty:
            self.data_buffer = pd.concat([self.data_buffer, processed_data])

            unique_epochs = self.data_buffer.index.get_level_values("epoch").unique()[
                -self.window_size :
            ]
            self.data_buffer = self.data_buffer[
                self.data_buffer.index.get_level_values("epoch").isin(unique_epochs)
            ]

    def transform_features(self):
        """
        Applies the feature transformation to the entire dataframe.

        Parameters:
        - dataframe: dataframe to transform.

        Returns:
        - Transformed data.
        """
        if self.data_buffer.empty:
            return pd.DataFrame()

        timeseries_buffer = self.processor.process_timeseries(
            self.data_buffer[const.PREPROCESSED_TIMESERIES_LIST]
        )
        elevation_buffer = self.processor.process_elevation(
            self.data_buffer["elevation"]
        )

        transformed_df = self.data_buffer.drop(
            columns=const.PREPROCESSED_TIMESERIES_LIST + ["elevation"]
        )
        transformed_df = pd.concat(
            [transformed_df, timeseries_buffer, elevation_buffer], axis="columns"
        )

        transformed_df = self.transformer.transform(
            transformed_df[const.PROCESSED_FEATURE_LIST]
        )
        return transformed_df

    def transform_to_observation_spec(
        self,
        transformed_df: pd.DataFrame,
        observation_spec: array_spec.BoundedArraySpec,
    ):
        """
        Converts the transformed DataFrame to the form expected by observation_spec.

        Parameters:
        - transformed_df: Pandas DataFrame with the transformed data.

        Returns:
        - An array of NumPy that meets the observation_spec.
        """
        observation_array = np.full(
            (self.window_size,) + observation_spec.shape,
            fill_value=const.SENTINEL_VALUE,
            dtype=np.float32,
        )
        observation_array[:, :, 0] = 0

        if not transformed_df.empty:
            unique_epochs = transformed_df.index.get_level_values("epoch").unique()
            epoch_to_idx = {
                epoch: idx
                for idx, epoch in enumerate(
                    unique_epochs[-self.window_size :].to_numpy()
                )
            }

            # Convertir las características a un array de NumPy
            features_array = transformed_df[const.PROCESSED_FEATURE_LIST].to_numpy()
            features_array = np.hstack(
                (
                    np.ones((features_array.shape[0], 1), dtype=np.float32),
                    features_array,
                )
            )

            # Obtener índices para cons_idx, sat_idx, y freq_idx
            cons_idx = transformed_df.index.get_level_values("cons_idx").to_numpy()
            sat_idx = transformed_df.index.get_level_values("sat_idx").to_numpy()
            freq_idx = transformed_df.index.get_level_values("freq_idx").to_numpy()
            epochs = transformed_df.index.get_level_values("epoch").to_numpy()

            # Llenar el array de observación utilizando los índices para la ubicación directa
            for epoch, cons, sat, freq, features in zip(
                epochs, cons_idx, sat_idx, freq_idx, features_array
            ):
                if epoch in epoch_to_idx:
                    observation_array[
                        epoch_to_idx[epoch], get_global_sat_idx(cons, sat, freq), :
                    ] = features
        if self.window_size == 1:
            observation_array = observation_array[0, ...]

        return observation_array
