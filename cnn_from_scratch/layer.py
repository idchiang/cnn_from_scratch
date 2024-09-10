"""
# Layer objects (DenseLayer focused for now)
"""
import numpy as np
from .neuron import DenseNeuron

# Layer superclass. Pass for now.


class Layer():
    pass

# Main type of layer for our simple CNN.
# Note: use consistent names with DenseNeuron().


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, name='L0', learning_rate=1e-2, activation_function_str='sigmoid', quiet=True):
        self.name = name
        # Sanity checks
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(
                f"DenseLayer.__init__() in {self.name}: input_dim must be a positive integer")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(
                f"DenseLayer.__init__() in {self.name}: output_dim must be a positive integer")
        if type(activation_function_str) in {list, tuple, np.ndarray}:
            if len(activation_function_str) != output_dim:
                raise ValueError(
                    f"DenseLayer.__init__() in {self.name}: when using arrays, len(activation_function_str) must == output_dim")
        # Basics
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.activation_funtion_str = activation_function_str
        # Initialize Neurons
        if type(activation_function_str) is str:
            activation_function_str_arr = [
                activation_function_str] * output_dim
        else:
            activation_function_str_arr = activation_function_str
        for i, af_str in enumerate(activation_function_str_arr):
            self.neurons.append(DenseNeuron(input_dim=input_dim, learning_rate=learning_rate,
                                name=f"{name}_N{i+1}", activation_function_str=af_str), quiet=quiet)
        # I/O variables
        self.current_input = np.full(input_dim, np.nan)
        self.current_output = np.full(output_dim, np.nan)
        self.quiet = quiet

    def compute_output(self, input_data):
        assert len(
            input_data) == self.input_dim, f"DenseLayer.compute_output() for {self.name}: Input dimension doesn't match"
        self.current_input = input_data
        for i, neuron in enumerate(self.neurons):
            self.current_output[i] = neuron.compute_output(input_data)
        return self.current_output

    def backpropagation(self, dL_da_arr):
        delta_x_arr = np.zeros(self.input_dim)
        for i, neuron in enumerate(self.neurons):
            delta_x_arr += neuron.backpropagation(dL_da_arr[i])
        return delta_x_arr

    def update_parameters(self):
        for i, neuron in enumerate(self.neurons):
            neuron.update_parameters()

    def reset_parameters(self):
        for i, neuron in enumerate(self.neurons):
            neuron.reset_parameters()


"""
class InputLayer(Layer):
    def __init__(self):
        assert False, "InputLayer() not implemented yet!!!!"


class OutputLayer(Layer):
    def __init__(self):
        assert False, "OutputLayer() not implemented yet!!!!"
"""
