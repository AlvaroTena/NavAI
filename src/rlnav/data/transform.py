import os

import dill as pickle
import pandas as pd
from neptune import Run
from neptune.utils import stringify_unsupported
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from navutils.logger import Logger
from rlnav.data.process import DataProcessor


class ColumnTransformer:

    def __init__(self, transformers=None):
        if transformers is not None:
            self.transformers = {item[0]: item[1] for item in transformers}
        else:
            self.transformers = {}

    def fit(self, X, y=None):
        """
        Ajusta los transformers a las columnas del DataFrame.

        :param X: DataFrame de pandas con los datos a transformar
        :param y: No se utiliza
        :return: self
        """
        for column in X:
            if column in self.transformers:
                transformer = self.transformers[column]
                transformer.fit(X[[column]])
        return self

    def transform(self, X: pd.DataFrame):
        """
        Aplica los transformers ajustados a las columnas del DataFrame.

        :param X: DataFrame de pandas con los datos a transformar
        :return: DataFrame transformado
        """
        df_transformado = pd.DataFrame(index=X.index)
        for column in X:
            if column in self.transformers:
                transformer = self.transformers[column]
                try:
                    df_transformado[column] = transformer.transform(X[[column]]).ravel()

                except:
                    Logger.log_message(
                        Logger.Category.ERROR,
                        Logger.Module.MAIN,
                        f"Try to transform a column with not-fitted transformer",
                    )

            else:
                df_transformado[column] = X[[column]]

        return df_transformado

    def fit_transform(self, X, y=None):
        """
        Ajusta los transformers a las columnas del DataFrame y luego los aplica.

        :param X: DataFrame de pandas con los datos a transformar
        :param y: No se utiliza
        :return: DataFrame transformado
        """
        return self.fit(X).transform(X)


def read_and_transform(input_path, output_dir, npt_run: Run):
    transforms_dir = os.path.join(output_dir, "transforms")
    os.makedirs(transforms_dir, exist_ok=True)

    files = [
        os.path.join(input_path, file)
        for file in os.listdir(input_path)
        if file.endswith(".parquet")
    ]

    Logger.log_message(
        Logger.Category.DEBUG,
        Logger.Module.MAIN,
        f"Timestamp data transform",
    )

    df = pd.DataFrame()
    for file in files:
        df_columns = pd.read_parquet(file, columns=["doy", "seconds_of_day"])
        df = pd.concat([df, df_columns], axis="index")
        del df_columns

    df = DataProcessor.process_timeseries(df)

    for column_name in df.columns:
        df_file_path = os.path.join(
            transforms_dir, f"transformed_{column_name}.parquet"
        )
        pd.DataFrame(df[column_name]).to_parquet(df_file_path)

        npt_run[f"transformed_data/{column_name}"].track_files(df_file_path)

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f"Timestamp data transformed saved: {df_file_path}",
        )

    columns2Transforms = [
        ("delta_time", MinMaxScaler()),
        ("code", StandardScaler()),
        ("phase", StandardScaler()),
        ("doppler", MinMaxScaler()),
        ("snr", MinMaxScaler()),
        ("elevation", MinMaxScaler()),
        ("residual", RobustScaler()),
        ("iono", RobustScaler()),
        ("delta_cmc", RobustScaler()),
        ("crc", MinMaxScaler()),
    ]
    npt_run["data_transformers/columns"] = stringify_unsupported(columns2Transforms)

    ct = ColumnTransformer(columns2Transforms)

    for column_name, _ in columns2Transforms:
        Logger.log_message(
            Logger.Category.DEBUG, Logger.Module.MAIN, f"Transform {column_name}"
        )

        df = pd.DataFrame()
        for file in files:
            df_columns = pd.read_parquet(file, columns=[column_name])
            df = pd.concat([df, df_columns], axis="index")
            del df_columns

        if column_name == "elevation":
            df = DataProcessor.process_elevation(df)

        # Ajustar y transformar los datos de la columna actual
        df = ct.fit_transform(pd.DataFrame(df[column_name]))

        # Guardar la columna transformada en un file temporal
        df_file_path = os.path.join(
            transforms_dir, f"transformed_{column_name}.parquet"
        )
        df.to_parquet(df_file_path)

        npt_run[f"transformed_data/{column_name}"].track_files(df_file_path)

        Logger.log_message(
            Logger.Category.DEBUG,
            Logger.Module.MAIN,
            f"{column_name} data transformed saved: {df_file_path}",
        )

    # Serializar la función de transformación
    with open(
        (transform_fn_path := os.path.join(output_dir, "transform_fn.pkl")), "wb"
    ) as f:
        pickle.dump(ct, f, recurse=True)

    npt_run["data_transformers/transform_fn"].upload(transform_fn_path)

    return ct
