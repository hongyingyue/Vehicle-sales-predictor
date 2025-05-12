import logging

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def explore_data(train_df, selected_columns, target_column, test_df=None):
    return


def vis_scatter(data, x, y, c=None):
    """
    num_column is for regression target
    https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html#pandas.DataFrame.plot.scatter
    """
    if not c:
        plt.scatter(data[x], data[y], alpha=0.7)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"Scatter: {y} vs {x}")
    else:
        data.plot.scatter(x=x, y=y, c=c, colormap="viridis")
    return


def vis_hist(data, column, class_column=None, bins=100):
    """
    class_column is for classification problem
    """
    plt.figure(figsize=(6, 4))
    if not class_column:
        plt.hist(data[column], bins=bins, alpha=0.7)
    else:
        unique_class = data[class_column].dropna().unique()
        for cls in sorted(unique_class):
            samples = data.loc[data[class_column] == cls]
            plt.hist(samples[column], bins=bins, label=f"{class_column}={cls}")
        plt.legend()
    plt.xlabel(column)
    plt.ylabel("frequency")
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
