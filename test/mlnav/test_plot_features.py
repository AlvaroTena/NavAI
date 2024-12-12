import os

import pandas as pd

from mlnav.data.reader import Reader
from mlnav.features.prepro import preprocess_df
from mlnav.report.plot_features import plot_features


def test_plot_features():
    """Tests if plot_features saves kde and box features for a non corrupt file"""
    columns = [
        "code",
        "phase",
        "doppler",
        "snr",
        "elevation",
        "residual",
        "iono",
        "cmc",
        "code_rate_cons",
    ]
    filename_box = "Files_Test/prepro_tests/plot_files/test_plot_box_features.png"
    filename_kde = "Files_Test/prepro_tests/plot_files/test_plot_kde_features.png"
    files = [filename_box, filename_kde]
    filenames = [filename_box, filename_kde]
    df_list = Reader().perform_reading(
        data_dir="Files_Test/prepro_tests/plot_files/", n_files=-1
    )
    df = pd.concat([preprocess_df(df)[1] for df in df_list])
    plot_features(df, columns)
    for file in files:
        assert file
