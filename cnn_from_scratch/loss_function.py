"""
# Loss functions
"""
# Log Loss (for classification)
# y_true, y_pred AREN'T interchangeable!!

import numpy as np


def log_loss(y_true, y_pred):
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


def log_loss_derivative(y_true, y_pred):
    # Sanity checks: might implement them elsewhere
    assert len(y_true) == len(y_pred)
    assert np.all((y_true == 0) | (y_true == 1)), "y_true isn't binary"
    assert np.all((y_pred >= 0) & (y_pred <= 1)), "y_pred outside (0.0, 1.0)"
    assert False, "log_loss_derivative() not implemented yet"


"""
Functions for setting loss
"""
loss_functions_dict = {
    'log_loss': [log_loss, log_loss_derivative],
}


def set_loss(input_str):
    if input_str.lower() in loss_functions_dict:
        return loss_functions_dict[input_str.lower()][0]
    else:
        raise ValueError(
            f"Loss function {input_str} not implemented yet! Allowed inputs are: {loss_functions_dict.keys()}")


def set_loss_derivative(input_str):
    if input_str.lower() in loss_functions_dict:
        return loss_functions_dict[input_str.lower()][1]
    else:
        raise ValueError(
            f"Loss function {input_str} not implemented yet! Allowed inputs are: {loss_functions_dict.keys()}")
