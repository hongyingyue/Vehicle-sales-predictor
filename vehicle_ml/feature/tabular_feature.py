"""
function: rank, nunique
"""

import logging
from typing import List, Optional

import numpy as np

from ..utils import timer

logger = logging.getLogger(__name__)


def add_count_feature():
    return


@timer("add num-num feature")
def add_num_num_feature(data, num_features, fun_list=None, feature_columns: Optional[List[str]] = None):
    if fun_list is None:
        fun_list = ["ratio", "multi", "add", "diff"]

    valid_funs = {"ratio", "multi", "add", "diff"}
    for fun in fun_list:
        if fun not in valid_funs:
            raise ValueError(f"Unsupported cross operation: '{fun}'")

    for f1 in num_features:
        for f2 in num_features:
            if f1 == f2:
                continue
            if "ratio" in fun_list:
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(data[f2].values != 0, data[f1].values / data[f2].values, np.nan)
                    data[f"{f1}_{f2}_ratio"] = ratio
            if "multi" in fun_list:
                data[f"{f1}_{f2}_multi"] = data[f1].values * data[f2].values
            if "add" in fun_list:
                data[f"{f1}_{f2}_add"] = data[f1].values + data[f2].values
            if "diff" in fun_list:
                data[f"{f1}_{f2}_diff"] = data[f1].values - data[f2].values
    return data


@timer("add cat-num feature")
def add_cat_num_feature(data, amount_feas, category_feas, fun_list=None):
    if fun_list is None:
        fun_list = ["median", "mean", "max", "min", "std"]

    valid_funs = {"median", "mean", "max", "min", "std"}
    for fun in fun_list:
        if fun not in valid_funs:
            raise ValueError(f"Unsupported aggregation function: '{fun}'")

    for f in amount_feas:
        for cate in category_feas:
            if f == cate:
                continue
            group = data.groupby(cate)[f]
            for fun in fun_list:
                feature_name = f"{cate}_{f}_{fun}"
                try:
                    data[feature_name] = group.transform(fun)
                except Exception as e:
                    raise RuntimeError(f"Error computing '{fun}' for '{cate}' and '{f}': {e}")

    return data


def add_cat_cat_feature():
    return


def add_target_encoding_feature():
    return
