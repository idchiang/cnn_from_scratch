import numpy as np


def calc_accuracy(a_true, a_pred):
    y_true = np.argmax(a_true, axis=1)
    y_pred = np.argmax(a_pred, axis=1)
    return np.sum(y_true == y_pred) / len(y_true)
