import os

import numpy as np
import pandas as pd
import pytest

from mlnav.data.reader import Reader
from mlnav.features.prepro import *

UNIT_TEST_PATH = os.path.dirname(os.path.realpath(__file__))


def test_default_case():
    """Tests if the prepro returns a pandas dataframe"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/one_file/", n_files=-1
    )
    for df in df_list:
        df1 = preprocess_df(df)[1]
        df2 = pd.DataFrame()
        if type(df) != type(df2):
            pytest.fail("Prepro failed in return", pytrace=True)


def test_one_file():
    """Tests if the prepro returns the correct dataframe shape of reading 1 file"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/one_file/", n_files=-1
    )
    df = preprocess_df(df_list[0])
    assert df[1].shape == (15, 13)


def test_multiple_files():
    """Tests if the prepro returns the correct dataframe shape of reading multiple (5) files"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/multiple_files/", n_files=-1
    )
    for df in df_list:
        df_prepro = preprocess_df(df)
        if df_prepro[1].shape != (15, 13):
            pytest.fail("Wrong dataframe shape", pytrace=True)


def test_multipath_files():
    """Tests if the reader returns the correct dataframe shape of reading multiple (5) files with multipath"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/multipath_files/", n_files=-1
    )
    for df_in in df_list:
        df_out = preprocess_df(df_in)
        if df_out[1].shape != (15, 14):
            pytest.fail("Wrong dataframe shape", pytrace=True)


def test_empty_directory():
    """Tests if the prepro returns the correct dataframe shape of reading an empty directory"""
    df_in = pd.DataFrame({})
    df_out = preprocess_df(df_in)
    assert df_out == None


def test_cmc_values():
    """Tests if the prepro computes correctly the cmc values"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/prepro_tests/cmc_cons_file/", n_files=1
    )
    df = preprocess_df(df_list[0], drop_nan_rows=False)[1]
    """Tests first 10 rows"""
    assert df["cmc"][0] == -49
    assert df["cmc"][1] == -122
    assert df["cmc"][2] == -119
    assert df["cmc"][3] == -86
    assert df["cmc"][4] == -129
    assert df["cmc"][5] == -6
    assert df["cmc"][6] == -21
    assert df["cmc"][7] == -98
    assert df["cmc"][8] == -11
    assert df["cmc"][9] == -26


def test_code_rate_cons_values():
    """Tests if the prepro computes correctly the code_rate_cons values"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/prepro_tests/cmc_cons_file/", n_files=1
    )
    df = preprocess_df(df_list[0], drop_nan_rows=False)[1]

    """Tests rows 2-11"""
    assert df["code_rate_cons"][2] == pytest.approx(15934560256)
    assert df["code_rate_cons"][3] == pytest.approx(30.086134704508176)
    assert df["code_rate_cons"][4] == pytest.approx(46.23882530880654)
    assert np.isnan(df["code_rate_cons"][5]) == True
    assert np.isnan(df["code_rate_cons"][6]) == True
    assert df["code_rate_cons"][7] == pytest.approx(57.5813539198436)
    assert df["code_rate_cons"][8] == pytest.approx(2.6041991583577714)
    assert df["code_rate_cons"][9] == pytest.approx(71.00665283203125)
    assert np.isnan(df["code_rate_cons"][10]) == True
    assert df["code_rate_cons"][11], pytest.approx(45.27241685988921)
    """Test NaN values"""
    for i in range(0, 15, 5):
        assert np.isnan(df["code_rate_cons"][i]) == True


def test_drop_nan_rows():
    """Tests if the prepro drops correctly all rows with NaN values"""
    df_list = Reader().perform_reading(
        data_dir="Files_Test/prepro_tests/plot_files/", n_files=-1
    )

    for df_in in df_list:
        df_out = preprocess_df(df_in, drop_nan_rows=True)
        if df_out[1].isnull().values.any() != False:
            pytest.fail("Null values incorrectly deleted", pytrace=True)
