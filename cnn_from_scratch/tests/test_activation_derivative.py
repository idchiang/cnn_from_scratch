# test_derivative_function.py

import pytest
import numpy as np
from scipy.special import expit as sigmoid_activation2
from ..activation_function import *

ActFunc_dict = {
    'linear': LinearActFunc(),
    'sigmoid': SigmoidActFunc(),
    'relu': ReLUActFunc()
}


def linear_derivative2(input_value, activation_value):
    return np.ones_like(input_value)


def sigmoid_derivative2(input_value, activation_value):
    return sigmoid_activation2(input_value) * (1 - sigmoid_activation2(input_value))


def relu_derivative2(input_value, activation_value):
    return np.where(input_value > 0, 1, 0)


def expected_derivative(input_str, input_value, activation_value):
    expected_derivative_functions_dict = {
        'linear': linear_derivative2,
        'sigmoid': sigmoid_derivative2,
        'relu': relu_derivative2
    }
    if input_str.lower() in ActFunc_dict:
        res = expected_derivative_functions_dict[input_str.lower()](
            input_value, activation_value)
        return res
    else:
        raise ValueError(
            f"Activation function {input_str} not implemented yet! Allowed inputs are: {expected_derivative_functions_dict.keys()}")


# Test cases
input_str_list = ['linear', 'sigmoid', 'relu']
input_value_list = [-1e9, -1, 0, 1, 1e9, np.nan]
test_cases_derivative = []
for input_str in input_str_list:
    for input_value in input_value_list:
        activation_value = ActFunc_dict[input_str.lower()].compute(input_value)
        test_cases_derivative.append(
            [input_str, input_value, activation_value])
input_str_list2 = ['lInear', 'Sigmoid', 'ReLU']
for input_str in input_str_list2:
    activation_value = ActFunc_dict[input_str.lower()].compute(
        input_value_list[0])
    test_cases_derivative.append(
        [input_str, input_value_list[0], activation_value])

for i, test_case in enumerate(test_cases_derivative):
    input_str, input_value, activation_value = test_case
    expected = expected_derivative(input_str, input_value, activation_value)
    test_cases_derivative[i].append(expected)


def test_derivative():
    for i, (input_str, input_value, activation_value, expected) in enumerate(test_cases_derivative):
        print(i, input_str, input_value, activation_value, expected)
        func = ActFunc_dict[input_str.lower()]
        result = func.derivative(input_value, activation_value)
        try:
            assert np.isclose(
                result, expected, atol=1e-5), f"Test case {i+1} failed: {result} != {expected}"
        except AssertionError:
            if np.isnan(result) and np.isnan(expected):
                pass
            else:
                raise AssertionError(
                    f"Test case {i+1} failed: {result} != {expected}")
        if __name__ == '__main__':
            print(f"Test case {i+1} passed.")
