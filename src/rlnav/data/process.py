import copy
import datetime as dt
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
import scipy.constants as sc_const
from scipy import stats

import rlnav.types.constants as const
from navutils.logger import Logger


class DataProcessor:
    def __init__(
        self,
        config_signals,
        drop_nan_rows=True,
        drop_outliers: Union[bool, int, float] = False,
    ):
        self.config_signals = config_signals
        self.drop_nan_rows = drop_nan_rows
        self.drop_outliers = (
            drop_outliers if isinstance(drop_outliers, (int, float)) else 3
        )
        self.reset_times()

    def reset_times(self):
        self._times = {}
        self._times["preprocess_batch"] = []

    def get_times(self):
        times = copy.deepcopy(self._times)
        self.reset_times()
        return times

    def preprocess_batch(self, df: pd.DataFrame, verbose=True) -> pd.DataFrame:
        start = time.time()
        if df.empty:
            return None  # If there are not data to preprocess return None object

        if verbose:
            st = time.time()
            name = f"AI_Multipath_{df['epoch'].iloc[0].strftime('%Y%m%d_%H%M%S')}"
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.MAIN,
                f"Preprocessing {(len_df:=len(df))} rows from {name}",
            )

        bad_data = df[((df["code"] < 0) | (df["phase"] < 0) | (df["snr"] < 32.0))].index
        if len(bad_data) > 0:
            if verbose:
                Logger.log_message(
                    Logger.Category.DEBUG,
                    Logger.Module.MAIN,
                    f"{(n_bad:=len(bad_data))} ({(n_bad/len_df):.1%}) rows will be dropped as bad data",
                )
            df.drop(bad_data, inplace=True)

        # Categorization
        df["cons"] = self._preprocess_extract_cons(df["sat_id"])
        df["sat_id"] = self._preprocess_sat_id(df["sat_id"])

        df["doy"], df["seconds_of_day"] = self._preprocess_timeseries(df["epoch"])

        # Cons, sat_id and freq as index values
        df["cons_idx"], df["sat_idx"], df["freq_idx"] = self._preprocess_indexes(
            df[["cons", "sat_id", "freq"]]
        )

        df["elevation"] = np.where(df["elevation"] >= 5, df["elevation"], np.nan)

        # Handling NaN values and outliers
        if self.drop_nan_rows:
            if verbose:
                Logger.log_message(
                    Logger.Category.DEBUG,
                    Logger.Module.MAIN,
                    f"{(n_nan:=df.isna().any(axis=1).sum())} ({(n_nan/len_df):.1%}) rows from {name} will be dropped as NaN values",
                )
            df.dropna(inplace=True)

        if self.drop_outliers:
            z_scores = np.abs(
                stats.zscore(
                    df[
                        [
                            "code",
                            "phase",
                            "doppler",
                            "snr",
                            "elevation",
                            "residual",
                            "iono",
                            "delta_cmc",
                            "crc",
                        ]
                    ]
                )
            )
            threshold = self.drop_outliers
            outliers = (z_scores > threshold).any(axis=1)
            if verbose:
                Logger.log_message(
                    Logger.Category.DEBUG,
                    Logger.Module.MAIN,
                    f"{outliers.sum()} rows from {name} will be dropped as outliers",
                )
            df.drop(index=df[outliers].index, inplace=True)

        df.reset_index(drop=True, inplace=True)
        if verbose:
            et = time.time()
            Logger.log_message(
                Logger.Category.DEBUG,
                Logger.Module.MAIN,
                f"Preprocessed {name} (num rows: {len(df)})({str(dt.timedelta(seconds=(et-st)))})",
            )

        df["epoch"] = df["epoch"].astype("datetime64[us]")
        df.set_index(
            ["epoch", "cons_idx", "sat_idx", "freq_idx"], drop=True, inplace=True
        )
        df.sort_index(inplace=True)

        self._times["preprocess_batch"].append(time.time() - start)
        return df[const.PREPROCESSED_FEATURE_LIST]

    def _preprocess_extract_cons(self, sat_id: pd.Series) -> pd.Series:
        return pd.Categorical(sat_id.str[0], categories=["G", "E", "B"], ordered=False)

    def _preprocess_sat_id(self, sat_id: pd.Series) -> pd.Series:
        return pd.Categorical(
            sat_id.str[1:],
            categories=[str(i).zfill(2) for i in range(1, 64)],
            ordered=False,
        )

    def _preprocess_timeseries(
        self, df_timeseries: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        return df_timeseries.dt.dayofyear.astype(dtype=np.int32), (
            (df_timeseries.dt.hour * sc_const.hour)
            + (df_timeseries.dt.minute * sc_const.minute)
            + df_timeseries.dt.second
            # + (df_timeseries.dt.microsecond / 10e5)
        ).astype(dtype=np.int32)

    def _preprocess_indexes(
        self, df_indexes: pd.DataFrame
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        for signals in self.config_signals:
            mask1 = np.isclose(
                df_indexes["freq"], const.SIGNAL_FREQ.SIGNALS[signals[0]]
            )
            df_indexes.loc[mask1, "freq"] = 0
            mask2 = np.isclose(
                df_indexes["freq"], const.SIGNAL_FREQ.SIGNALS[signals[1]]
            )
            df_indexes.loc[mask2, "freq"] = 1

        return (
            df_indexes["cons"].cat.codes,
            df_indexes["sat_id"].cat.codes,
            df_indexes["freq"].astype(np.uint8, copy=False),
        )

    @staticmethod
    def process_timeseries(df_timeseries: pd.DataFrame):
        df_processed = pd.DataFrame(index=df_timeseries.index)
        df_processed["DOY_sin"] = np.sin(
            (df_timeseries["doy"] * sc_const.day) * (2 * np.pi / sc_const.Julian_year)
        )
        df_processed["DOY_cos"] = np.cos(
            (df_timeseries["doy"] * sc_const.day) * (2 * np.pi / sc_const.Julian_year)
        )
        df_processed["Day_sin"] = np.sin(
            df_timeseries["seconds_of_day"] * (2 * np.pi / sc_const.day)
        )
        df_processed["Day_cos"] = np.cos(
            df_timeseries["seconds_of_day"] * (2 * np.pi / sc_const.day)
        )

        return df_processed

    @staticmethod
    def process_elevation(elevation: pd.DataFrame):
        return pd.DataFrame(np.deg2rad(elevation))
