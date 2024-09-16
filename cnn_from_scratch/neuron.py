"""
# Neuron objects (DenseNeuron-focused for now)
"""
import numpy as np
from abc import ABC, abstractmethod
import warnings
from .activation_function import *

# Neuron superclass.


class Neuron(ABC):
    def __init__(self, input_dim, name='N0', learning_rate=1e-2, act_func=None, quiet=True):
        if act_func is None:
            act_func = SigmoidActFunc()
        assert act_func.__class__.__base__ is ActivationFunction, f"Neuron.__init__() in {name}: act_func must be a subclass of ActivationFunction()"
        # Set input dimension & activation function
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.name = name
        self.act_func = act_func
        self.backpropagation_count = 0
        # Initialize I/O variables
        self.current_input = np.full(input_dim, np.nan)
        self.current_result = np.nan
        self.current_output = np.nan  # output = activation_function(result)
        self.derivative = np.nan
        self.quiet = quiet
        # Initialize weights
        self.w = np.random.randn(input_dim) / np.sqrt(input_dim)
        self.b = np.random.randn()
        # Initialize deltas
        self.delta_w = np.zeros(input_dim)
        self.delta_b = 0

    @abstractmethod
    def compute_output(self, input_data):
        pass

    @abstractmethod
    def backpropagation(self, dL_da):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

# Neuron for Dense Layer.
# Not sure if we need other types of neurons. Just keep the name clear.


class DenseNeuron(Neuron):
    def __init__(self, input_dim, name='N0', learning_rate=1e-2, act_func=None, quiet=True):
        super().__init__(input_dim=input_dim, name=name,
                         learning_rate=learning_rate, act_func=act_func, quiet=quiet)
        # Initialize weights
        self.w = np.random.randn(input_dim) / np.sqrt(input_dim)
        self.b = 0.
        # Initialize deltas
        self.delta_w = np.zeros(input_dim)
        self.delta_b = 0.

    def compute_output(self, input_data):
        assert len(
            input_data) == self.input_dim, f"DenseNeuron.compute_output() for {self.name}: Input dimension doesn't match"
        self.current_input = input_data
        self.current_result = np.dot(self.w, input_data) + self.b
        self.current_output = self.act_func.compute(self.current_result)
        # derivative is fixed per forward calculation
        self.derivative = self.act_func.derivative(
            self.current_result, self.current_output)
        return self.current_output

    def backpropagation(self, dL_da):
        """
        Math notes:

        L = Loss function = L(a_pred, a_true)
        a = sigma(y), sigma is the activation function
        y = wx+b, x is input from previous layer (a_i-1)
        dL_da = dL/da

        Want: dL/db, dL/dw and dL/dx (d is actually partial...)

        dL/db = dL/da * dsigma(y)/dy * dy/b = dL/da * sigma'
        dL/dw = dL/da * dsigma(y)/dy * dy/dw = dL/da * sigma' * x = dL/db * x
        dL/dx = dL/da * dsigma(y)/dy * dy/dx = dL/da * sigma' * w = dL/db * w

        The returned dL/dx should be combined in the "layer" object before sending back to the previous layer.
        """
        delta_b = dL_da * self.derivative

        self.delta_w += delta_b * self.current_input
        self.delta_b += delta_b
        delta_x = delta_b * self.w
        self.backpropagation_count += 1
        return delta_x

    def update_parameters(self):
        if self.backpropagation_count == 0:
            warnings.warn(
                f"No DenseNeuron.backpropagation() for {self.name} performed before DenseNeuron.update_parameters()")
            return
        # update w & b
        if not self.quiet:
            print(f"DenseNeuron.update_parameters() in {self.name}:", type(
                self.delta_w), self.delta_w)
            print(f"DenseNeuron.update_parameters() in {self.name}:", type(
                self.learning_rate), self.learning_rate)
            print(f"DenseNeuron.update_parameters() in {self.name}:", type(
                self.backpropagation_count), self.backpropagation_count)
        self.w -= self.learning_rate * self.delta_w / \
            float(self.backpropagation_count)
        self.b -= self.learning_rate * self.delta_b / \
            float(self.backpropagation_count)
        # reset backpropagation stuff
        self.delta_w = np.zeros(self.input_dim)
        self.delta_b = 0
        self.backpropagation_count = 0

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def reset_parameters(self):
        self.__init__(input_dim=self.input_dim, name=self.name, learning_rate=self.learning_rate,
                      act_func=self.act_func, quiet=self.quiet)
