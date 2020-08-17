import numpy as np
import torch
from sklearn.metrics import accuracy_score


def accuracy(y_true, pred_log_probas):
    y_true = np.array(y_true)
    if isinstance(pred_log_probas, torch.Tensor):
        pred_log_probas = pred_log_probas.detach().numpy()
    else:
        pred_log_probas = np.array(pred_log_probas)
    y_true = y_true.reshape((-1,))
    y_pred = np.argmax(pred_log_probas, axis=-1).reshape((-1,))
    return accuracy_score(y_true, y_pred)
