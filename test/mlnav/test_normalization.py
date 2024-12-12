import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler

from mlnav.data.reader import Reader
from mlnav.features.normalization import normalize_column
from mlnav.features.prepro import preprocess_df

cons_categories = ["E", "G"]
sat_id_categories = [str(i).zfill(2) for i in range(1, 41)]
freq_categories = ["FREQ_L1_GPS", "FREQ_L2_GPS", "FREQ_E1_GAL", "FREQ_E5B_GAL"]
# List of tuples (column name, transformer, columns expected) with:
#   + column name: Name to identify the transformer
#   + transformer: estimator to apply to each column
#   + columns: list of names of the columns to apply each transformer
transformers = [
    ("epoch", "drop", ["epoch"]),
    (
        "cons",
        OneHotEncoder(
            categories=[cons_categories],
            dtype=np.uint8,
            handle_unknown="ignore",
            sparse_output=False,
        ),
        cons_categories,
    ),
    ("sat_id", "drop", sat_id_categories),
    (
        "freq",
        OneHotEncoder(
            categories=[freq_categories],
            dtype=np.uint8,
            handle_unknown="ignore",
            sparse_output=False,
        ),
        freq_categories,
    ),
    ("code", MinMaxScaler((-1, 1)), ["code"]),
    ("phase", MinMaxScaler((-1, 1)), ["phase"]),
    ("doppler", MinMaxScaler((-1, 1)), ["doppler"]),
    ("snr", MinMaxScaler((-1, 1)), ["snr"]),
    ("elevation", MinMaxScaler((-1, 1)), ["elevation"]),
    ("residual", MinMaxScaler((-1, 1)), ["residual"]),
    ("iono", MinMaxScaler((-1, 1)), ["iono"]),
    ("cmc", MinMaxScaler((-1, 1)), ["cmc"]),
    ("code_rate_cons", MinMaxScaler((-1, 1)), ["code_rate_cons"]),
]
transformers = [transformer for transformer in transformers if transformer[1] != "drop"]


def test_nan_values():
    """Test if the normalization does not touch NaN values"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/prepro_tests/cmc_cons_file/", n_files=-1
    )
    df = pd.concat([preprocess_df(df)[1] for df in df_list])
    df = pd.concat(
        [
            normalize_column(pd.DataFrame(df[transformer[0]]), transformer)[0]
            for transformer in transformers
        ],
        axis=1,
    )
    print([transformer for transformer in transformers])
    """Test NaN values (after normlizing data)"""
    assert df["code_rate_cons"].isnull().values.any()


def test_normalize_data():
    """Tests if the nomalization normlizes data correctly"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/prepro_tests/cmc_cons_file/", n_files=-1
    )
    df = pd.concat([preprocess_df(df, True)[1] for df in df_list])
    df = pd.concat(
        [
            normalize_column(pd.DataFrame(df[transformer[0]]), transformer)[0]
            for transformer in transformers
        ],
        axis=1,
    )
    """Test if any value is less than -1 or greater than 1"""
    assert (df > 1).any().all() == False
    assert (df < -1).any().all() == False
