"""
# Loss functions
"""
# Log Loss (for classification)
# y_true, y_pred AREN'T interchangeable!!

import numpy as np


def log_loss(a_true, a_pred):
    # Sanity checks: might implement them elsewhere
    assert len(a_true) == len(a_pred)
    assert np.all((a_true == 0) | (a_true == 1)), "y_true isn't binary"
    assert np.all((a_pred >= 0) & (a_pred <= 1)), "y_pred outside (0.0, 1.0)"
    if a_pred.dtype not in [np.float32, np.float64, float]:
        a_pred = a_pred.astype(np.float64)
    epsilon = np.finfo(a_pred.dtype).eps
    # clip to avoid y == 0 & y == 1
    a_pred_clipped = np.clip(a_pred, epsilon, 1 - epsilon)
    loss = -np.mean(a_true * np.log(a_pred_clipped) +
                    (1 - a_true) * np.log(1 - a_pred_clipped))
    return loss


def log_loss_derivative(a_true, a_pred):
    # Sanity checks: might implement them elsewhere
    assert len(a_true) == len(a_pred)
    assert np.all((a_true == 0) | (a_true == 1)), "y_true isn't binary"
    assert np.all((a_pred >= 0) & (a_pred <= 1)), "y_pred outside (0.0, 1.0)"
    if a_pred.dtype not in [np.float32, np.float64, float]:
        a_pred = a_pred.astype(np.float64)
    epsilon = np.finfo(a_pred.dtype).eps
    # clip to avoid y == 0 & y == 1
    a_pred_clipped = np.clip(a_pred, epsilon, 1 - epsilon)
    dL_da = (-a_true / a_pred_clipped + (1 - a_true) /
             (1 - a_pred_clipped)) / len(a_true)
    return dL_da


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
