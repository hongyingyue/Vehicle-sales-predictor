import logging
import time
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataIngester:
    """
    Base class for data ingestion. Extend this class to implement specific ingestion logic.
    """

    def __init__(self, input_data_path: str, transforms: Optional[List[callable]] = None) -> None:
        self.input_data_path = input_data_path
        self.transforms = transforms or []

    def ingest(self) -> pd.DataFrame:
        logger.info("Start to ingest data")
        start_time = time.time()
        data = pd.read_csv(self.input_data_path)
        data["Date"] = data["Date"].astype(str)
        logger.info(
            f"Finish ingest data, elapsed in {time.time() - start_time:.2f}s, shape {data.shape}, memory {data.memory_usage().sum() / 1e9:.3f}GB"
        )

        for transform in self.transforms:
            transform_start_time = time.time()
            data = transform(data)
            logger.info(f"Finish {transform} data in {time.time() - transform_start_time:.1f}s, shape: {data.shape}")
        return data
