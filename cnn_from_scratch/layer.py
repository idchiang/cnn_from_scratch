"""
# Layer objects (DenseLayer focused for now)
"""
import numpy as np
from abc import ABC, abstractmethod
from .neuron import DenseNeuron
from .activation_function import *

# Layer superclass.


class Layer(ABC):
    def __init__(self, input_dim, output_dim, name='L0', learning_rate=1e-2, act_func=None, quiet=True):
        self.name = name
        if act_func is None:
            act_func = SigmoidActFunc()
        # Sanity checks
        if not isinstance(input_dim, int) or input_dim <= 0:
            raise ValueError(
                f"DenseLayer.__init__() in {self.name}: input_dim must be a positive integer")
        if not isinstance(output_dim, int) or output_dim <= 0:
            raise ValueError(
                f"DenseLayer.__init__() in {self.name}: output_dim must be a positive integer")
        if type(act_func) in {list, tuple, np.ndarray}:
            if len(act_func) != output_dim:
                raise ValueError(
                    f"DenseLayer.__init__() in {self.name}: when using arrays, len(act_func) must == output_dim")
        # I/O variables
        self.current_input = np.full(input_dim, np.nan)
        self.current_output = np.full(output_dim, np.nan)
        self.quiet = quiet
        # Basics
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.act_func = act_func
        self.neurons = []

    @abstractmethod
    def compute_output(self, input_data):
        pass

    @abstractmethod
    def backpropagation(self, dL_da_arr):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def reset_parameters(self):
        pass

# Main type of layer for our simple CNN.
# Note: use consistent names with DenseNeuron().

# activation(w_matrix * x_arr + offset_b) = output_arr


class DenseLayer(Layer):
    def __init__(self, input_dim, output_dim, name='L0', learning_rate=1e-2, act_func=SigmoidActFunc(), quiet=True):
        super().__init__(input_dim=input_dim, output_dim=output_dim, name=name, learning_rate=learning_rate,
                         act_func=act_func, quiet=quiet)
        # Initialize Neurons
        if type(act_func) not in {list, tuple, np.ndarray}:
            act_func_arr = [
                act_func] * output_dim
        else:
            act_func_arr = act_func
        for i, af in enumerate(act_func_arr):
            self.neurons.append(DenseNeuron(input_dim=input_dim, learning_rate=learning_rate,
                                name=f"{name}_N{i+1}", act_func=af, quiet=quiet))

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

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for i, neuron in enumerate(self.neurons):
            neuron.set_learning_rate(learning_rate)

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
