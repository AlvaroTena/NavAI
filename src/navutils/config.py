from os import path
from typing import Text

import yaml
from box import ConfigBox


def load_config(config_path: Text) -> ConfigBox:
    """Loads yaml config in instance of ConfigBox.
    Args:
        config_path {Text}: path to config
    Returns:
        box.ConfigBox
    """

    with open(config_path) as config_file:

        config = yaml.safe_load(config_file)
        config = ConfigBox(config)
        config.base_path = path.abspath(path.join(config_path, "../../../"))

    return config
