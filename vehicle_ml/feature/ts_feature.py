import logging
from typing import List, Optional, Union

import pandas as pd

from ..utils import timer

logger = logging.getLogger(__name__)


@timer("add lagging feature")
def add_lagging_feature(
    data: pd.DataFrame,
    groupby_column: Union[str, List[str]],
    value_columns: List[str],
    lags: List[int],
    feature_columns: Optional[List[str]] = None,
):
    # note that the data should be sorted by time already
    # the lagging feature could be further developed use f1 - f1_lag, or f1 / f1_lag

    if not isinstance(groupby_column, (str, list)):
        raise TypeError(f"'groupby_column' must be a string or a list of strings, but got {type(groupby_column)}.")

    if not isinstance(value_columns, (list, tuple)):
        raise TypeError(f"'value_columns' must be a list of strings, but got {type(value_columns)}.")

    feature_columns: List[str] = feature_columns if feature_columns is not None else []
    for column in value_columns:
        if column not in data.columns:
            raise ValueError(f"Value column '{column}' not found in DataFrame.")

        for lag in lags:
            feature_col_name = f"{column}_lag{lag}"
            feature_columns.append(feature_col_name)
            logger.debug(
                f"Creating lagging feature: {feature_col_name} for column '{column}' with lag {lag} and groupby '{groupby_column}'."
            )
            data[feature_col_name] = data.groupby(groupby_column)[column].shift(lag)
    return data


@timer("add rolling feature")
def add_rolling_feature(
    data: pd.DataFrame,
    groupby_column: Union[str, List[str]],
    value_columns: List[str],
    periods: List[int],
    agg_funs=["mean"],
    feature_columns: Optional[List[str]] = None,
    prefix: Optional[str] = None,
):
    if not isinstance(groupby_column, (str, list)):
        raise TypeError(f"'groupby_column' must be a string or a list of strings, but got {type(groupby_column)}.")
    if not isinstance(value_columns, list):
        raise TypeError(f"'value_columns' must be a list of strings, but got {type(value_columns)}.")

    feature_columns: List[str] = feature_columns or []
    for column in value_columns:
        if column not in data.columns:
            raise ValueError(f"Value column '{column}' not found in DataFrame.")

        for period in periods:
            for agg_fun in agg_funs:
                prefix = prefix or column
                feature_col_name = f"{prefix}_roll{period}_{agg_fun}"
                logger.debug(
                    f"Creating rolling feature: {feature_col_name} for column '{column}' with period {period}, "
                    f"aggregation '{agg_fun}', and groupby '{groupby_column}'."
                )
                feature_columns.append(feature_col_name)
                data[feature_col_name] = data.groupby(groupby_column)[column].transform(
                    lambda x: x.rolling(period).agg(agg_fun)
                )

    return data


@timer("add lagging rolling feature")
def add_lagging_rolling_feature(
    data: pd.DataFrame,
    groupby_column: Union[str, List[str]],
    value_columns: List[str],
    lags,
    period: List[int],
    agg_funs=["mean"],
    feature_columns: Optional[List[str]] = None,
    prefix: Optional[str] = None,
):
    # note that the data should be sorted by time already
    if not isinstance(groupby_column, (str, list)):
        raise TypeError(f"'groupby_column' must be a string or a list of strings, but got {type(groupby_column)}.")
    if not isinstance(value_columns, list):
        raise TypeError(f"'value_columns' must be a list of strings, but got {type(value_columns)}.")

    feature_columns: List[str] = feature_columns or []

    for column in value_columns:
        if column not in data.columns:
            raise ValueError(f"Value column '{column}' not found in DataFrame.")

        for lag in lags:
            for agg_fun in agg_funs:
                prefix = prefix or column
                feature_col_name = f"{prefix}_lag{lag}_roll{period}_by{groupby_column}_{agg_fun}"
                feature_columns.append(feature_col_name)
                logger.debug(
                    f"Creating lagging rolling feature: {feature_col_name} for column '{column}' with "
                    f"lag {lag}, rolling period {period}, aggregation '{agg_fun}', and groupby '{groupby_column}'."
                )
                data[feature_col_name] = data.groupby(groupby_column)[column].transform(
                    lambda x: x.shift(lag).rolling(period).agg(agg_fun)
                )
    return data
