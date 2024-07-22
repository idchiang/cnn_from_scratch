import numpy as np
"""
# Activation functions
"""
# not sure if it's better to make them as functions or objects. advantage of functions: simple; advantage of objects: unified parameters (but there aren't too many parameters)
# Let's do functions for now

def set_activation(input_str):
    if input_str == 'linear':
        return linear_act
    elif input_str == 'sigmoid':
        return sigmoid_act
    elif input_str == 'ReLU':
        return ReLU_act
    else:
        assert False, f"activation function {input_str} not implemented yet!!!"

# linear
def linear_act(input_value):
    return input_value

# sigmoid
def sigmoid_act(input_value):
    return 1 / (1 + np.exp(-input_value))

# linear action
def ReLU_act(input_value):
    result = input_value.copy()
    result[result <= 0] = 0
    return result
