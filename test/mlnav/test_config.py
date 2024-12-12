from navutils.config import load_config

config = load_config("Files_Test/config_tests/params.yaml")


def test_string():
    assert config.test.string == "hello"


def test_number():
    assert config.other.number == 3


def test_bool():
    assert config.other.boolean == True
