import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mean_absolute_percentage_error(y_true, y_pred, eps: float = 1e-8):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


def evaluate(targets: np.ndarray, predictions: np.ndarray):
    """
    evaluate model performance
    :param targets: np.ndarray, shape [n_samples, 12, n_nodes, n_features]
    :param predictions: [n_samples, 12, n_nodes, n_features]
    :return: a dict [str -> float]
    """
    assert targets.shape == predictions.shape and targets.shape[1] == 12
    n_samples = targets.shape[0]
    scores = dict()
    y_true, y_pred = np.reshape(targets[:, :3], (n_samples, -1)), np.reshape(predictions[:, :3], (n_samples, -1))
    scores['MAE-15min'] = mean_absolute_error(y_true, y_pred)
    scores['MAE-30min'] = mean_absolute_error(y_true, y_pred)
    scores['MAE-60min'] = mean_absolute_error(y_true, y_pred)
    y_true, y_pred = np.reshape(targets[:, :6], (n_samples, -1)), np.reshape(predictions[:, :6], (n_samples, -1))
    scores['RMSE-15min'] = mean_squared_error(y_true, y_pred)
    scores['RMSE-30min'] = mean_squared_error(y_true, y_pred)
    scores['RMSE-60min'] = mean_squared_error(y_true, y_pred)
    y_true, y_pred = np.reshape(targets, (n_samples, -1)), np.reshape(predictions, (n_samples, -1))
    scores['MAPE-15min'] = mean_absolute_percentage_error(y_true, y_pred)
    scores['MAPE-30min'] = mean_absolute_percentage_error(y_true, y_pred)
    scores['MAPE-60min'] = mean_absolute_percentage_error(y_true, y_pred)
    return scores
