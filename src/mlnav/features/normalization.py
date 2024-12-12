import datetime as dt
from time import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from mlnav.utils.logger import Logger


def normalize_column(column: pd.DataFrame, transformer: List[Tuple]):
    st = time()
    Logger.getLogger().debug(f"Normalizing {transformer[0]} column...")

    ct = ColumnTransformer(
        transformers=[(transformer[0], transformer[1], [transformer[0]])]
    )
    norm_column = ct.fit_transform(column)

    et = time()
    Logger.getLogger().info(
        f"{transformer[0]} normalization finished in {str(dt.timedelta(seconds=(et-st)))}"
    )

    return (
        pd.DataFrame(
            norm_column,
            columns=transformer[2],
            dtype=(
                np.int8
                if np.isclose(norm_column, np.round(norm_column), atol=1e-5).all()
                else np.float32
            ),
        ),
        ct,
    )
