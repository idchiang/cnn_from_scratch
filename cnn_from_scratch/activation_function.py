"""
# Activation functions
"""
# okay, let's make them as classes for easier management

import numpy as np
from abc import ABC, abstractmethod

# output := activate_function(result). For simple CNN, result = wx+b.


class ActivationFunction(ABC):
    # def __init__(self):
    #     self.name = self.__class__.__name__

    @abstractmethod
    def compute(self, input_value):
        pass

    @abstractmethod
    def derivative(self, result, output):
        pass


"""
Linear
"""


class LinearActFunc(ActivationFunction):
    def compute(self, input_value):
        return input_value

    def derivative(self, result, output):
        return 1


"""
Sigmoid
"""


class SigmoidActFunc(ActivationFunction):
    def compute(self, input_value):
        if input_value >= 0:
            return 1 / (1 + np.exp(-input_value))
        else:
            return np.exp(input_value) / (1 + np.exp(input_value))

    def derivative(self, result, output):
        return output * (1 - output)


"""
Rectified Linear Unit
"""


class ReLUActFunc(ActivationFunction):
    def compute(self, input_value):
        if input_value <= 0:
            return 0
        else:
            return input_value

    def derivative(self, result, output):
        if result <= 0:
            return 0
        else:
            return 1
