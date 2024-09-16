# test_loss_derivative_function.py

import pytest
import numpy as np
from ..loss_function import *

LossFunc_dict = {
    'log': LogLoss(),
    # 'mse': MSELoss()
}


def log_loss_derivative2(y_true, y_pred):
    return -y_true / y_pred + (1 - y_true) / (1 - y_pred)


def mse_loss_derivative2(y_true, y_pred):
    return 2 * (y_pred - y_true) / len(y_true)


def expected_loss_derivative(input_str, y_true, y_pred):
    expected_derivative_functions_dict = {
        'log': log_loss_derivative2,
        'mse': mse_loss_derivative2
    }
    if input_str.lower() in LossFunc_dict:
        res = expected_derivative_functions_dict[input_str.lower()](
            y_true, y_pred)
        return res
    else:
        raise ValueError(
            f"Loss function {input_str} not implemented yet! Allowed inputs are: {expected_derivative_functions_dict.keys()}")


# Test cases
input_str_list = ['log']
y_true_list = [np.array([0, 1, 1]), np.array([1, 0, 1])]
y_pred_list = [np.array([0.1, 0.9, 0.8]), np.array([0.6, 0.4, 0.7])]
test_cases_loss_derivative = []
for input_str in input_str_list:
    for y_true in y_true_list:
        for y_pred in y_pred_list:
            test_cases_loss_derivative.append([input_str, y_true, y_pred])

for i, test_case in enumerate(test_cases_loss_derivative):
    input_str, y_true, y_pred = test_case
    loss_func = LossFunc_dict[input_str.lower()]
    # Compute the expected derivative
    expected = expected_loss_derivative(input_str, y_true, y_pred)
    test_cases_loss_derivative[i].append(expected)


def test_loss_derivative():
    for i, (input_str, y_true, y_pred, expected) in enumerate(test_cases_loss_derivative):
        loss_func = LossFunc_dict[input_str.lower()]
        result = loss_func.derivative(y_true, y_pred)
        try:
            assert np.allclose(
                result, expected, atol=1e-5), f"Test case {i+1} failed: {result} != {expected}"
        except AssertionError:
            if np.all(np.isnan(result) == np.isnan(expected)):
                pass
            else:
                raise AssertionError(
                    f"Test case {i+1} failed: {result} != {expected}")
        if __name__ == '__main__':
            print(f"Test case {i+1} passed.")
