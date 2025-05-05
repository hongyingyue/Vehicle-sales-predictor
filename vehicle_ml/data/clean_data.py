import logging
import time

import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    def __init__(self) -> None:
        pass

    def clean(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Start to clean data")
        start_time = time.time()
        start_shape = data.shape

        data = data
        logger.info(
            f"Finish cleaning data, elapsed in {time.time() - start_time:.0f}s, shape from {start_shape} to {data.shape}"
        )
        return

    def __call__(self, data):
        return self.clean(data)

    def __repr__(self):
        return "clean-data"
