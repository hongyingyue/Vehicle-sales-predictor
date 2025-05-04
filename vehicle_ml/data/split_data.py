import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter:
    def __init__(self, split_type: str = "random") -> None:
        self.split_type = split_type

    def split(self, data):
        return data


class DatetimeSplitter:
    def __init__(self, time_column: str, test_date: List) -> None:
        self.time_column = time_column
        self.test_date = test_date

    def split(self, data: pd.DataFrame):
        test_index = data[self.time_column].isin(self.test_date)
        train_index = ~test_index
        train = data[train_index]
        test = data[test_index]
        return train, test


class StratifiedLeaveOneGroupOut:
    """
    Stratified Leave-One-Group-Out cross-validator.

    Provides train/test indices to split data according to a third-party
    provided group of samples. This variation aims to create folds that
    are stratified with respect to the target variable within each group.

    Warning: This cross-validator assumes that the input data 'x' is
             already sorted by the 'groups' column. The index of 'x'
             should be continuous from the beginning.

    Parameters
    ----------
    n_splits : int
        Number of splits to generate. This determines the size of the
        test set relative to the group size.

    random_state : int, default=2020
        Random state to ensure reproducibility.
    """

    def __init__(self, n_splits: int, random_state: Optional[int] = 2020):
        """Initializes the StratifiedLeaveOneGroupOut object."""
        if not isinstance(n_splits, int) or n_splits <= 0:
            raise ValueError("n_splits must be a positive integer.")
        self.n_splits = n_splits
        self.random_state = random_state
        np.random.seed(self.random_state)

    def split(
        self, x: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Generates indices to split data into training and test sets.

        Parameters
        ----------
        x : pd.DataFrame
            Training data. Assumed to be sorted by the 'groups' column.
            The index should be continuous.
        y : pd.Series, optional
            The target variable for stratification. Must have the same
            length as 'x'. Defaults to None.
        groups : pd.Series
            Group labels for the samples in 'x'. Must have the same
            length as 'x'.

        Yields
        ------
        train_index : list of int
            The training set indices for the fold.
        test_index : list of int
            The test set indices for the fold.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        if y is not None and len(y) != len(x):
            raise ValueError("The length of 'y' must be the same as 'x'.")
        if len(groups) != len(x):
            raise ValueError("The length of 'groups' must be the same as 'x'.")

        unique_groups = np.unique(groups)
        for idx in range(self.n_splits):
            train_index, test_index = [], []
            for group in unique_groups:
                group_indices = x.index[groups == group].tolist()
                group_data = x.loc[group_indices]

                if len(group_data) < (self.n_splits + 1):
                    raise ValueError(
                        f"Group '{group}' has fewer samples than n_splits + 1. "
                        f"Consider reducing n_splits or handling small groups differently."
                    )

                block_size = len(group_data) // (self.n_splits + 1)
                test_start_offset = block_size * (idx + 1)
                test_end_offset = test_start_offset + block_size

                test_indices_in_group = group_data.iloc[test_start_offset:test_end_offset].index.tolist()
                train_indices_in_group = [i for i in group_indices if i not in test_indices_in_group]

                train_index.extend(train_indices_in_group)
                test_index.extend(test_indices_in_group)

            yield train_index, test_index

    def get_n_splits(self) -> int:
        return self.n_splits


class BlockedLeaveOneGroupOut:
    """
    Blocked Leave-One-Group-Out cross-validator.

    Provides train/test indices to split data according to a third-party
    provided group of samples. This cross-validator divides each group
    into `n_splits + 1` contiguous blocks and iteratively uses one block
    as the test set while the remaining blocks form the training set.

    Warning: This cross-validator assumes that the input data 'x' is
             already sorted by the 'groups' column. The index of 'x'
             should be continuous from the beginning.

    Parameters
    ----------
    n_splits : int
        Number of splits to generate. This determines how many times
        each group will have a distinct block held out as the test set.

    random_state : int, default=2020
        Random state for potential future randomization (currently not used
        in the core logic but included for consistency).
    """

    def __init__(self, n_splits: int, random_state: Optional[int] = 2020):
        """Initializes the BlockedLeaveOneGroupOut object."""
        if not isinstance(n_splits, int) or n_splits <= 0:
            raise ValueError("n_splits must be a positive integer.")
        self.n_splits = n_splits
        self.random_state = random_state
        np.random.seed(self.random_state)

    def split(self, x: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None):
        """
        Generates indices to split data into training and test sets.

        Parameters
        ----------
        x : pd.DataFrame
            Training data. Assumed to be sorted by the 'groups' column.
            The index should be continuous.
        y : pd.Series, optional
            The target variable (not directly used in this splitting strategy).
            Defaults to None.
        groups : pd.Series
            Group labels for the samples in 'x'. Must have the same
            length as 'x'.

        Yields
        ------
        train_index : list of int
            The training set indices for the fold.
        test_index : list of int
            The test set indices for the fold.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None.")
        if len(groups) != len(x):
            raise ValueError("The length of 'groups' must be the same as 'x'.")

        unique_groups = np.unique(groups)
        for idx in range(self.n_splits):
            train_index, test_index = [], []
            for group in unique_groups:
                temp = x.loc[x[groups] == group]
                n_group_samples = len(temp)

                if n_group_samples < (self.n_splits + 1):
                    raise ValueError(
                        f"Group '{group}' has fewer samples ({n_group_samples}) than n_splits + 1 ({self.n_splits + 1}). "
                        f"Consider reducing n_splits or handling small groups differently."
                    )

                block_size = n_group_samples // (self.n_splits + 1)
                test_start_offset = block_size * (idx + 1)
                test_end_offset = test_start_offset + block_size

                # Get indices for the test block within the current group
                test_indices_in_group = temp.iloc[test_start_offset:test_end_offset].index.tolist()

                # Get indices for the training blocks (all other blocks) within the current group
                train_indices_in_group = [i for i in temp.index if i not in test_indices_in_group]

                train_index.extend(train_indices_in_group)
                test_index.extend(test_indices_in_group)

            yield train_index, test_index

    def get_n_splits(self) -> int:
        return self.n_splits
