import pytest
import numpy as np
from ..layer import DenseLayer
from ..activation_function import *

# Only flow check now.


def test_layer():
    input_dim, output_dim = 200, 10
    x = np.random.rand(input_dim)
    n_backward = 10
    my_layer = DenseLayer(input_dim=input_dim, output_dim=output_dim)
    y = my_layer.compute_output(x)
    assert len(y) == output_dim
    for i in range(n_backward):
        dL_da_arr = np.random.rand(output_dim)
        xp = my_layer.backpropagation(dL_da_arr)
        assert len(xp) == input_dim
    assert my_layer.neurons[0].backpropagation_count == n_backward
    my_layer.update_parameters()
    assert my_layer.neurons[0].backpropagation_count == 0
    my_layer.reset_parameters()
    my_layer2 = DenseLayer(input_dim=input_dim, output_dim=output_dim, act_func=[
                           ReLUActFunc()]*output_dim)
    my_layer2.reset_parameters()
