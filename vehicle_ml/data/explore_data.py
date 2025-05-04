import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class DataExplorer:
    def __init__(self) -> None:
        pass

    def explore(self, train_df, selected_columns, target_column, test_df=None):
        return


def vis_scatter():
    """ """
    return


def vis_hist(data, col, bins=100):
    """ """
    plt.hist(data[col], bins=bins)
    plt.xlabel(col)
    return


def vis_hist_by_class(self, value_col: str, class_col: str, bins=100, alpha=0.5, title=None):
    # hist for classification, different label
    unique_classes = self.data[class_col].dropna().unique()
    for cls in sorted(unique_classes):
        subset = self.data[self.data[class_col] == cls]
        plt.hist(subset[value_col], bins=bins, alpha=alpha, label=f"{class_col}={cls}")

    plt.xlabel(value_col)
    plt.ylabel("Count")
    plt.legend()
    if title:
        plt.title(title)
    plt.grid(True)
    plt.show()
    return


def vis_time_hist(data):
    plt.hist(data, bins=pd.date_range("2019-04-01", "2019-11-01", freq="d"), rwidth=0.74, color="#ffd700")
    return


def vis_time_trend(data):
    return


def vis_train_test_venn(train, test):
    from matplotlib_venn import venn2

    if not isinstance(train, set):
        train = set(train)
    if not isinstance(test, set):
        test = set(test)

    common_val = train & test
    train_val = train - common_val
    test_val = test - common_val
    print(f"train unique: {len(train_val)}")
    print(f"test unique: {len(test_val)}")
    print(f"common unique: {len(common_val)}")
    return venn2(
        subsets=(
            len(train_val),
            len(test_val),
            len(common_val),
        ),
        set_labels=("train", "test"),
    )
