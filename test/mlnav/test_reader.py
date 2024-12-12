import pandas as pd
import pytest

from mlnav.data.reader import Reader


def test_default_case():
    """Tests if the reader returns a pandas dataframe"""
    df = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/one_file", n_files=-1
    )
    df2 = pd.DataFrame()
    assert type(df[0]) == type(df2)


def test_one_file():
    """Tests if the reader returns the correct dataframe shape of reading 1 file"""
    df = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/one_file", n_files=-1
    )
    assert df[0].shape == (15, 10)


def test_multiple_files():
    """Tests if the reader returns the correct dataframe shape of reading multiple (5) files"""
    dfs = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/multiple_files", n_files=-1
    )
    for df in dfs:
        if df.shape != (15, 10):
            pytest.fail("Incorrect output", pytrace=True)


def test_multipath_files():
    """Tests if the reader returns the correct dataframe shape of reading multiple (5) files with multipath"""
    dfs = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/multipath_files", n_files=-1
    )
    for df in dfs:
        if df.shape != (15, 11):
            pytest.fail("Incorrect output", pytrace=True)


def test_empty_directory():
    """Tests if the reader returns the correct dataframe shape of reading an empty directory"""
    df = Reader().perform_reading(
        data_dir="Files_Test/reader_tests/empty_directory", n_files=-1
    )
    assert df == []
