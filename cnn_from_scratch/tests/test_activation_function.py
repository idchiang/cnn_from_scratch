# !pytest tests/test_activation_function.py
import pytest
import numpy as np
from scipy.special import expit as sigmoid_activation2
from ..activation_function import *

ActFunc_dict = {
    'linear': LinearActFunc(),
    'sigmoid': SigmoidActFunc(),
    'relu': ReLUActFunc()
}


def linear_activation2(input_value):
    return 1.0 * input_value


def relu_activation2(input_value):
    return np.maximum(0, input_value)


def expected_activation(input_str, input_value):
    expected_activation_functions_dict = {
        'linear': linear_activation2,
        'sigmoid': sigmoid_activation2,
        'relu': relu_activation2
    }
    if input_str.lower() in ActFunc_dict:
        res = expected_activation_functions_dict[input_str.lower()](
            input_value)
        return res
    else:
        raise ValueError(
            f"Activation function {input_str} not implemented yet! Allowed inputs are: {expected_activation_functions_dict.keys()}")


# Test cases
input_str_list = ['linear', 'sigmoid', 'relu']
input_value_list = [-1e9, -1, 0, 1, 1e9, np.nan]
test_cases_activation = []
for input_str in input_str_list:
    for input_value in input_value_list:
        test_cases_activation.append([input_str, input_value])
input_str_list2 = ['lInear', 'Sigmoid', 'ReLU']
for input_str in input_str_list2:
    test_cases_activation.append([input_str, input_value_list[0]])

for i, test_case in enumerate(test_cases_activation):
    input_str, input_value = test_case
    expected = expected_activation(input_str, input_value)
    test_cases_activation[i].append(expected)


def test_activation():
    for i, (input_str, input_value, expected) in enumerate(test_cases_activation):
        func = ActFunc_dict[input_str.lower()]
        result = func.compute(input_value)
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
