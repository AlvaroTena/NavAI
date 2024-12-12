import datetime as dt
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from mlnav.types.constant import *
from mlnav.utils.logger import Logger


def preprocess_df(
    df: pd.DataFrame,
    drop_nan_rows: bool = False,
    drop_outliers: Union[bool, int, float] = False,
) -> Tuple[dt.datetime, pd.DataFrame]:
    st = time.time()

    # If there are not DataFrames to preprocess return None object
    if df.empty:
        return None, df

    Logger.getLogger().debug(
        f"Processing AI_Multipath_{df['epoch'].iloc[0].strftime('%Y%m%d_%H%M%S')}..."
    )

    concat_df = df
    Logger.getLogger().debug(f"Loaded {len(concat_df)} rows")

    bad_data = concat_df[
        ((concat_df["code"] < 0) | (concat_df["phase"] < 0) | (concat_df["snr"] < 32.0))
    ].index
    if len(bad_data) > 0:
        Logger.getLogger().debug(f"{len(bad_data)} rows will be dropped as bad data")
        concat_df.drop(bad_data, inplace=True)

    # Sort DataFrame by sat_id (this will help us to compute code_rate_cons later)
    concat_df = concat_df.sort_values("sat_id", ignore_index=True)

    concat_df["cons"] = pd.Categorical(
        concat_df["sat_id"].str[0], categories=["G", "E", "B"], ordered=False
    )

    concat_df["sat_id"] = pd.Categorical(
        concat_df["sat_id"].str[1:],
        categories=[str(i).zfill(2) for i in range(1, 64)],
        ordered=False,
    )

    # Create pseudorange consistency rate columm
    # Divide into groups and calculate difference
    now_df = pd.concat(
        [
            group[1][
                ["epoch", "cons", "sat_id", "freq", "code", "phase", "doppler"]
            ].sort_values("epoch")
            for group in concat_df.groupby(["cons", "sat_id", "freq"], observed=False)
        ],
        ignore_index=False,
        sort=False,
    )
    prev_df = pd.concat(
        [
            group[1][["epoch", "cons", "sat_id", "freq", "code", "phase", "doppler"]]
            .sort_values("epoch")
            .shift(periods=1)
            for group in concat_df.groupby(["cons", "sat_id", "freq"], observed=False)
        ],
        ignore_index=False,
        sort=False,
    )
    # Variables for code_rate_cons
    time_var = (
        pd.to_datetime(now_df["epoch"]) - pd.to_datetime(prev_df["epoch"])
    ).dt.total_seconds()
    code_var = np.where(time_var < 1.0, now_df["code"] - prev_df["code"], np.nan)
    cmc = now_df["code"] - now_df["phase"]
    prev_cmc = prev_df["code"] - prev_df["phase"]
    # Reorder DataFrame by epoch (original order)
    concat_df["code_rate_cons"] = np.where(
        time_var < 1.0,
        abs(code_var - now_df["doppler"] * time_var).astype(np.float32).sort_index(),
        np.nan,
    )
    concat_df["delta_cmc"] = np.where(
        time_var < 1.0,
        abs((cmc - prev_cmc) / time_var),
        np.nan,
    )

    concat_df = concat_df.sort_values("epoch", ignore_index=True)
    column_names = (
        ["epoch", "cons", "sat_id"]
        + concat_df.columns.tolist()[2:-3]
        + concat_df.columns.tolist()[-2:]
    )
    concat_df = concat_df[column_names]

    # Freq as categorical value
    maskL1 = (np.isclose(concat_df["freq"], FREQ_L1_GPS)) & (concat_df["cons"] == "G")
    maskL2 = (np.isclose(concat_df["freq"], FREQ_L2_GPS)) & (concat_df["cons"] == "G")
    maskL5 = (np.isclose(concat_df["freq"], FREQ_L5_GPS)) & (concat_df["cons"] == "G")

    maskE1 = (np.isclose(concat_df["freq"], FREQ_E1_GAL)) & (concat_df["cons"] == "E")
    maskE5A = (np.isclose(concat_df["freq"], FREQ_E5A_GAL)) & (concat_df["cons"] == "E")
    maskE5B = (np.isclose(concat_df["freq"], FREQ_E5B_GAL)) & (concat_df["cons"] == "E")
    maskE5AB = (np.isclose(concat_df["freq"], FREQ_E5AB_GAL)) & (
        concat_df["cons"] == "E"
    )
    maskE6 = (np.isclose(concat_df["freq"], FREQ_E6_GAL)) & (concat_df["cons"] == "E")

    maskB1I = (np.isclose(concat_df["freq"], FREQ_B1I_BDS)) & (concat_df["cons"] == "B")
    maskB1C = (np.isclose(concat_df["freq"], FREQ_B1C_BDS)) & (concat_df["cons"] == "B")
    maskB1A = (np.isclose(concat_df["freq"], FREQ_B1A_BDS)) & (concat_df["cons"] == "B")
    maskB2A = (np.isclose(concat_df["freq"], FREQ_B2A_BDS)) & (concat_df["cons"] == "B")
    maskB3I = (np.isclose(concat_df["freq"], FREQ_B3I_BDS)) & (concat_df["cons"] == "B")

    concat_df.drop(["freq"], axis=1, inplace=True)

    concat_df.loc[maskL1, "freq"] = f"{FREQ_L1_GPS=}".split("=")[0]
    concat_df.loc[maskL2, "freq"] = f"{FREQ_L2_GPS=}".split("=")[0]
    concat_df.loc[maskL5, "freq"] = f"{FREQ_L5_GPS=}".split("=")[0]

    concat_df.loc[maskE1, "freq"] = f"{FREQ_E1_GAL=}".split("=")[0]
    concat_df.loc[maskE5A, "freq"] = f"{FREQ_E5A_GAL=}".split("=")[0]
    concat_df.loc[maskE5B, "freq"] = f"{FREQ_E5B_GAL=}".split("=")[0]
    concat_df.loc[maskE5AB, "freq"] = f"{FREQ_E5AB_GAL=}".split("=")[0]
    concat_df.loc[maskE6, "freq"] = f"{FREQ_E6_GAL=}".split("=")[0]

    concat_df.loc[maskB1I, "freq"] = f"{FREQ_B1I_BDS=}".split("=")[0]
    concat_df.loc[maskB1C, "freq"] = f"{FREQ_B1C_BDS=}".split("=")[0]
    concat_df.loc[maskB1A, "freq"] = f"{FREQ_B1A_BDS=}".split("=")[0]
    concat_df.loc[maskB2A, "freq"] = f"{FREQ_B2A_BDS=}".split("=")[0]
    concat_df.loc[maskB3I, "freq"] = f"{FREQ_B3I_BDS=}".split("=")[0]

    freqs = []
    freqs.append(f"{FREQ_L1_GPS=}".split("=")[0])
    freqs.append(f"{FREQ_L2_GPS=}".split("=")[0])
    freqs.append(f"{FREQ_L5_GPS=}".split("=")[0])
    freqs.append(f"{FREQ_E1_GAL=}".split("=")[0])
    freqs.append(f"{FREQ_E5A_GAL=}".split("=")[0])
    freqs.append(f"{FREQ_E5B_GAL=}".split("=")[0])
    freqs.append(f"{FREQ_E5AB_GAL=}".split("=")[0])
    freqs.append(f"{FREQ_E6_GAL=}".split("=")[0])
    freqs.append(f"{FREQ_B1I_BDS=}".split("=")[0])
    freqs.append(f"{FREQ_B1C_BDS=}".split("=")[0])
    freqs.append(f"{FREQ_B1A_BDS=}".split("=")[0])
    freqs.append(f"{FREQ_B2A_BDS=}".split("=")[0])
    freqs.append(f"{FREQ_B3I_BDS=}".split("=")[0])

    concat_df["freq"] = pd.Categorical(
        concat_df["freq"], categories=freqs, ordered=False
    )

    concat_df["elevation"] = np.where(
        concat_df["elevation"] >= 5, concat_df["elevation"], np.nan
    )

    # Returns dataframe without rows that contains NaN values
    if drop_nan_rows:
        Logger.getLogger().debug(
            f"{concat_df.isna().any(axis=1).sum()} rows will be dropped as NaN values"
        )
        concat_df.dropna(inplace=True)

    if drop_outliers:
        index = concat_df[
            (
                np.abs(
                    stats.zscore(
                        concat_df[
                            [
                                "code",
                                "phase",
                                "doppler",
                                "snr",
                                "elevation",
                                "residual",
                                "iono",
                                "delta_cmc",
                                "code_rate_cons",
                            ]
                        ]
                    )
                )
                > drop_outliers
                if isinstance(drop_outliers, (int, float))
                else 3
            ).any(axis=1)
        ].index
        Logger.getLogger().debug(f"{len(index)} rows will be dropped as outliers.")
        concat_df.drop(
            index=index,
            inplace=True,
        )

    concat_df.reset_index(drop=True, inplace=True)
    Logger.getLogger().debug(f"{len(concat_df)} rows processed.")

    et = time.time()
    Logger.getLogger().info(
        f"AI_Multipath_{df['epoch'].iloc[0].strftime('%Y%m%d_%H%M%S')} processing finished in {str(dt.timedelta(seconds=(et-st)))}"
    )

    return (df["epoch"].iloc[-1] - df["epoch"].iloc[0]).to_pytimedelta(), concat_df
