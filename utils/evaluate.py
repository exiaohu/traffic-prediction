from collections import defaultdict

import numpy as np


def evaluate(predictions: np.ndarray, targets: np.ndarray):
    """
    evaluate model performance
    :param predictions: [n_samples, 12, n_nodes, n_features]
    :param targets: np.ndarray, shape [n_samples, 12, n_nodes, n_features]
    :return: a dict [str -> float]
    """
    assert targets.shape == predictions.shape and targets.shape[1] == 12
    n_samples = targets.shape[0]
    scores = defaultdict(dict)
    y_true, y_pred = np.reshape(targets[:, :3], (n_samples, -1)), np.reshape(predictions[:, :3], (n_samples, -1))
    scores['masked MAE']['15min'] = masked_mae_np(y_pred, y_true, null_val=0.)
    scores['masked RMSE']['15min'] = masked_rmse_np(y_pred, y_true, null_val=0.)
    scores['masked MAPE']['15min'] = masked_mape_np(y_pred, y_true, null_val=0.) * 100.
    scores['MAE']['15min'] = mae_np(y_pred, y_true)
    scores['RMSE']['15min'] = rmse_np(y_pred, y_true)
    y_true, y_pred = np.reshape(targets[:, :6], (n_samples, -1)), np.reshape(predictions[:, :6], (n_samples, -1))
    scores['masked MAE']['30min'] = masked_mae_np(y_pred, y_true, 0.)
    scores['masked RMSE']['30min'] = masked_rmse_np(y_pred, y_true, 0.)
    scores['masked MAPE']['30min'] = masked_mape_np(y_pred, y_true, 0.) * 100.
    scores['MAE']['30min'] = mae_np(y_pred, y_true)
    scores['RMSE']['30min'] = rmse_np(y_pred, y_true)
    y_true, y_pred = np.reshape(targets, (n_samples, -1)), np.reshape(predictions, (n_samples, -1))
    scores['masked MAE']['60min'] = masked_mae_np(y_pred, y_true, 0.)
    scores['masked RMSE']['60min'] = masked_rmse_np(y_pred, y_true, 0.)
    scores['masked MAPE']['60min'] = masked_mape_np(y_pred, y_true, 0.) * 100.
    scores['MAE']['60min'] = mae_np(y_pred, y_true)
    scores['RMSE']['60min'] = rmse_np(y_pred, y_true)
    return scores


def masked_rmse_np(preds, labels, null_val=np.nan):
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(preds, labels)).astype('float32')
        mse = np.nan_to_num(mse * mask)
        return np.mean(mse)


def masked_mae_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        return np.mean(mae)


def masked_mape_np(preds, labels, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def mae_np(y_pred, y_true):
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def mse_np(y_pred, y_true):
    return np.mean(np.square(np.subtract(y_true, y_pred)))


def rmse_np(y_pred, y_true):
    return np.sqrt(mse_np(y_true, y_pred))
