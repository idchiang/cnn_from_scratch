"""
# Loss functions
"""
import numpy as np

# Log Loss (for classification)
# y_true, y_pred AREN'T interchangeable!!
def log_loss(y_true, y_pred, epsilon=1e-16):
    m = len(y_true)
    # Sanity checks: might implement them elsewhere
    assert m == len(y_pred)
    assert np.all((y_true == 0) + (y_true == 1))
    assert np.all(y_pred >= 0)
    assert np.all(y_pred <= 1)
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss_sum = -np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
    return loss_sum / m
