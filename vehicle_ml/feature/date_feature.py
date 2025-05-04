import logging
from typing import List, Optional

import pandas as pd

from ..utils import timer

logger = logging.getLogger(__name__)


@timer("add datetime feature")
def add_datetime_feature(
    data: pd.DataFrame, time_column, date_type_list=None, feature_columns: Optional[List[str]] = None
):
    if time_column not in data.columns:
        raise ValueError(f"'{time_column}' column not found in dataframe.")

    data[time_column] = pd.to_datetime(data[time_column], errors="coerce")
    available_features = {
        "year": data[time_column].dt.year,
        "month": data[time_column].dt.month,
        "day": data[time_column].dt.day,
        "hour": data[time_column].dt.hour,
        "weekofyear": data[time_column].dt.isocalendar().week,
        "dayofweek": data[time_column].dt.dayofweek,
        "is_wknd": data[time_column].dt.dayofweek // 5,
        "quarter": data[time_column].dt.quarter,
        "is_month_start": data[time_column].dt.is_month_start.astype(int),
        "is_month_end": data[time_column].dt.is_month_end.astype(int),
    }

    if date_type_list is None:
        date_type_list = list(available_features.keys())

    prefix = prefix or time_column
    for feature in date_type_list:
        if feature not in available_features:
            raise ValueError(f"Unsupported date_type: '{feature}'")

        feature_col_name = f"{prefix}_{feature}"
        data[feature_col_name] = available_features[feature]

    return data
