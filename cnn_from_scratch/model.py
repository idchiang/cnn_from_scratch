"""
# CNN Model objects (Simple CNN focused for now)
"""
import numpy as np
import warnings
from .layer import DenseLayer
from .loss_function import *
from .activation_function import *

# Main type of CNNModel for our simple CNN.
# Note: use consistent names with DenseNeuron() & DenseLayer().


class CNN_Model():
    def __init__(self, model_input_dim, model_output_dim, name='M0', learning_rate=1e-2, loss_func=None, quiet=True):
        self.name = name
        if loss_func is None:
            loss_func = LogLoss()
        # assert loss_func.__class__.__base__ is LossFunction, f"CNN_Model.__init__() in {name}: loss_func must be a subclass of LossFunction()"
        # Sanity checks
        if not isinstance(model_input_dim, int) or model_input_dim <= 0:
            raise ValueError(
                f"CNN_Model.__init__() in {self.name}: input_dim must be a positive integer")
        if not isinstance(model_output_dim, int) or model_output_dim <= 0:
            raise ValueError(
                f"CNN_Model.__init__() in {self.name}: output_dim must be a positive integer")
        if not learning_rate > 0:
            raise ValueError(
                f"CNN_Model.__init__() in {self.name}: learning_rate must be positive")
        # Basics
        self.model_input_dim = model_input_dim
        self.model_output_dim = model_output_dim
        self.learning_rate = learning_rate
        self.layers = []
        self.model_validated = False
        # I/O variables
        self.current_model_input = np.full(model_input_dim, np.nan)
        self.current_model_output = np.full(model_output_dim, np.nan)
        # Loss function
        self.loss_func = loss_func
        self.quiet = quiet

    def insert_layer(self, layer, idx=-1):
        if (type(idx) is not int) or (idx < -1) or (idx > len(self.layers)):
            raise ValueError(
                f"CNN_Model.insert_layer() in {self.name}: idx must be a int, with -1 <= idx <= len(self.layers)")
        if (type(layer)) is not DenseLayer:
            raise TypeError(
                f"CNN_Model.insert_layer() in {self.name}: layer must be a DenseLayer() object. Current input type: {type(layer)}")
        self.layers.insert(idx, layer)
        self.model_validated = False

    def remove_layer(self, idx):
        if (type(idx) is not int) or (idx < -1) or (idx > len(self.layers)):
            raise ValueError(
                f"CNN_Model.remove_layer() in {self.name}: idx must be a int, with -1 <= idx <= len(self.layers)")
        self.layers.pop(idx)
        self.model_validated = False

    def validate_model(self):
        # (1) check if all layers are connected with right I/O dimensions; (2) check if all layers have action function set up(?)
        validate_failed = False
        if len(self.layers) <= 0:
            validate_failed = True
        if (self.layers[0].input_dim != self.model_input_dim) or (self.layers[-1].output_dim != self.model_output_dim):
            validate_failed = True
        for i in range(len(self.layers) - 1):
            if self.layers[i].output_dim != self.layers[i + 1].input_dim:
                validate_failed = True
        if not validate_failed:
            self.model_validated = True

    def compute_output(self, model_input_data):
        if not self.model_validated:
            self.validate_model()
        if not self.model_validated:
            warnings.warn(
                f"CNN_Model.compute_output() in {self.name}: model validation failed. there is something wrong in the layer I/O dimensions.", UserWarning)
            return
        assert len(
            model_input_data) == self.model_input_dim, f"CNN_Model.compute_output() for {self.name}: Input dimension doesn't match"
        self.current_model_input = model_input_data
        prev_output = model_input_data
        # if not self.quiet:
        #     print(
        #         f"CNN_Model.compute_output() in {self.name}: prev_output={prev_output}")
        for i, layer in enumerate(self.layers):
            prev_output = layer.compute_output(prev_output)
            # if not self.quiet:
            #     print(
            #         f"CNN_Model.compute_output() in {self.name}: prev_output={prev_output}")
        self.current_model_output = prev_output
        return self.current_model_output

    def backpropagation(self, a_true):
        # to do: calculate dL_da_arr from a_true and a_pred=self.current_model_output
        prev_dL_da_arr = self.loss_func.derivative(
            a_true, self.current_model_output)
        for i, layer in enumerate(self.layers[::-1]):
            prev_dL_da_arr = layer.backpropagation(prev_dL_da_arr)

    def naive_model(self, hidden_dims=[50, 50], hidden_act_func=None,
                    output_act_func=None):
        if hidden_act_func is None:
            hidden_act_func = SigmoidActFunc()
        if output_act_func is None:
            output_act_func = SigmoidActFunc()
        # hidden_dims = [num1, num2, num3, ......]
        self.layers = []
        for dim in hidden_dims:
            if not isinstance(dim, int) or dim <= 0:
                raise ValueError(
                    f"CNN_Model.naive_model() in {self.name}: elements in hidden_dims must be positive integers")
        if type(hidden_dims) is np.ndarray:
            hidden_dims = hidden_dims.tolist()
        dims = [self.model_input_dim] + hidden_dims + [self.model_output_dim]
        if not self.quiet:
            print(f"CNN_Model.naive_model() in {self.name}: dims={dims}")
        act_func_arr = [
            hidden_act_func] * (len(dims) - 2) + [output_act_func]
        for i in range(len(dims) - 1):
            layer = DenseLayer(input_dim=dims[i], output_dim=dims[i+1],
                               name=f'{self.name}_L{i+1}', learning_rate=self.learning_rate, act_func=act_func_arr[i], quiet=self.quiet)
            self.insert_layer(layer, i)
        self.validate_model()

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(learning_rate)

    def update_parameters(self):
        for i, layer in enumerate(self.layers):
            layer.update_parameters()

    def reset_parameters(self):
        for i, layer in enumerate(self.layers):
            layer.reset_parameters()
