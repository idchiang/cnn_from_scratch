"""
# Neuron objects (DenseNeuron-focused for now)
"""
import numpy as np
from .activation_function import set_activation, set_derivative


class Neuron():
    pass

# Neuron for Dense Layer.
# Not sure if we need other types of neurons. Just keep the name clear.


class DenseNeuron(Neuron):
    def __init__(self, input_dim, name='N0', learning_rate=1e-2, activation_function_str='sigmoid', quiet=True):
        # Set input dimension & activation function
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.name = 'N'
        self.activation_funtion_str = activation_function_str
        self.activation_function = set_activation(activation_function_str)
        self.derivative_function = set_derivative(activation_function_str)
        # Initialize weights
        self.w = np.random.randn(input_dim) / np.sqrt(input_dim)
        self.b = np.random.randn()
        # Initialize deltas
        self.delta_w = np.zeros(input_dim)
        self.delta_b = 0
        self.backpropagation_count = 0
        # Initialize I/O variables
        self.current_input = np.full(input_dim, np.nan)
        self.current_result = np.nan
        self.current_output = np.nan  # output = activation_function(result)
        self.derivative = np.nan
        self.quiet = quiet

    def compute_output(self, input_data):
        assert len(
            input_data) == self.input_dim, f"DenseNeuron.compute_output() for {self.name}: Input dimension doesn't match"
        self.current_input = input_data
        self.current_result = np.dot(self.w, input_data) + self.b
        self.current_output = self.activation_function(self.current_result)
        # derivative is fixed per forward calculation
        self.derivative = self.derivative_function(
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
        assert self.backpropagation_count > 0, f"No DenseNeuron.backpropagation() for {self.name} performed before DenseNeuron.update_parameters()"
        # update w & b
        self.w -= self.learning_rate * self.delta_w / self.backpropagation_count
        self.b -= self.learning_rate * self.delta_b / self.backpropagation_count
        # reset backpropagation stuff
        self.delta_w = np.zeros(self.input_dim)
        self.delta_b = 0
        self.backpropagation_count = 0

    def reset_parameters(self):
        self.__init__(self.input_dim, self.learning_rate,
                      self.activation_function_str)
