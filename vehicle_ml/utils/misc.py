import logging
import math
import os
import time
import random
from contextlib import contextmanager

import numpy as np
import psutil

logger = logging.getLogger(__name__)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def reduce_mem_usage(df, verbose: bool = True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    cols_ = [col for col in list(df) if col not in ["cid", "vid"]]
    for col in cols_:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        logger.info(
            f"Memory usage from {start_mem:5.2f}MB to {end_mem:5.2f}Mb with {100 * (start_mem - end_mem) / start_mem}% reduction"
        )
    return df


@contextmanager
def timer(name: str):
    """elapsed time tracer
    @timer("data")
    def load_data():
        pass
    """
    start_time = time.time()
    msg = f"[{name}] start"
    try:
        logger.info(msg)
    except NameError:
        print(msg)
    yield

    msg = f"[{name}] finished in {time.time() - start_time:.2f} s"
    try:
        logger.info(msg)
    except NameError:
        print(msg)


@contextmanager
def memory_tracer(name: str):
    """memory tracer
    @memory_tracer("data")
    def load_data():
        pass
    """
    start_time = time.time()
    p = psutil.Process(os.getpid())
    start_memory = p.memory_info().rss / 2.0**30
    yield

    end_memory = p.memory_info().rss / 2.0**30
    delta = end_memory - start_memory
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    msg = f"[{end_memory:.1f}GB({sign}{delta:.1f}GB):{time.time() - start_time:.1f}sec] {name} "
    try:
        logger.info(msg)
    except NameError:
        print(msg)
