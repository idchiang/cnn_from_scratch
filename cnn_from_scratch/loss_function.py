"""
# Loss functions
"""
# Log Loss (for classification)
# y_true, y_pred AREN'T interchangeable!!

import numpy as np


def log_loss(y_true, y_pred, epsilon=1e-15):
    # Sanity checks: might implement them elsewhere
    assert len(y_true) == len(y_pred)
    assert np.all((y_true == 0) | (y_true == 1)), "y_true isn't binary"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "y_pred outside (0.0, 1.0)"
    if y_pred.dtype not in [np.float32, np.float64, float]:
        y_pred = y_pred.astype(np.float64)
    epsilon = np.finfo(y_pred.dtype).eps
    # clip to avoid y == 0 & y == 1
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred_clipped) +
                    (1 - y_true) * np.log(1 - y_pred_clipped))
    return loss
