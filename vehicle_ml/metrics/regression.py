import numpy as np


def get_mae(y_true, y_pred):
    score = np.mean(np.abs(y_true - y_pred))
    return score


def get_mse(y_true, y_pred):
    # mean square error
    score = np.mean((y_true - y_pred) ** 2)
    return score


def get_rmse(y_true, y_pred):
    score = np.sqrt(get_mse(y_true, y_pred))
    return score


def get_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    idx = y_true > 0
    y_true = y_true[idx]
    y_pred = y_pred[idx]

    score = np.abs(y_pred - y_true) / np.abs(y_true)
    return np.mean(score)


def get_smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    score = np.abs(y_pred - y_true) / ((np.abs(y_true) + np.abs(y_pred)) / 2)
    score = np.where(np.isnan(score), 0, score)
    return score
