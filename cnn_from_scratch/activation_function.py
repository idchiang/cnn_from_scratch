"""
# Activation functions
"""
# not sure if it's better to make them as functions or objects. advantage of functions: simple; advantage of objects: unified parameters (but there aren't too many parameters)
# Let's do functions for now

import numpy as np

"""
Linear
"""


def linear_activation(input_value):
    return input_value


def linear_derivative(result, output):
    return 1


"""
Sigmoid
"""


def sigmoid_activation(input_value):
    if input_value >= 0:
        return 1 / (1 + np.exp(-input_value))
    else:
        return np.exp(input_value) / (1 + np.exp(input_value))


def sigmoid_derivative(result, output):
    return output * (1 - output)


"""
Rectified Linear Unit
"""


def relu_activation(input_value):
    if input_value <= 0:
        return 0
    else:
        return input_value


def relu_derivative(result, output):
    if result <= 0:
        return 0
    else:
        return 1


"""
Functions for setting activation & derivative
"""
activation_functions_dict = {
    'linear': [linear_activation, linear_derivative],
    'sigmoid': [sigmoid_activation, sigmoid_derivative],
    'relu': [relu_activation, relu_derivative]
}


def set_activation(input_str):
    if input_str.lower() in activation_functions_dict:
        return activation_functions_dict[input_str.lower()][0]
    else:
        raise ValueError(
            f"Activation function {input_str} not implemented yet! Allowed inputs are: {activation_functions_dict.keys()}")


def set_derivative(input_str):
    if input_str.lower() in activation_functions_dict:
        return activation_functions_dict[input_str.lower()][1]
    else:
        raise ValueError(
            f"Activation function {input_str} not implemented yet! Allowed inputs are: {activation_functions_dict.keys()}")
