import pytest
import numpy as np
from ..layer import DenseLayer
from ..model import CNN_Model
# Only flow check now.


def test_model():
    input_dim, output_dim = 200, 10
    x = np.random.rand(input_dim)
    n_backward = 10
    my_model = CNN_Model(model_input_dim=input_dim, model_output_dim=output_dim,
                         quiet=False)
    my_model.naive_model(
        hidden_dims=[130, 70, 40], hidden_activation_function_str='ReLU')
    my_model.reset_parameters()
    assert len(my_model.layers) == 4
    y = my_model.compute_output(x)
    assert len(y) == output_dim
    for i in range(n_backward):
        a_true = np.random.randint(2, size=output_dim, dtype=int)
        my_model.backpropagation(a_true)
    assert my_model.layers[0].neurons[0].backpropagation_count == n_backward
    my_model.update_parameters()
    assert my_model.layers[0].neurons[0].backpropagation_count == 0
    #
    my_model2 = CNN_Model(model_input_dim=input_dim, model_output_dim=output_dim,
                          )
    my_model2.naive_model(
        hidden_dims=[130, 70], hidden_activation_function_str='linear', output_activation_function_str='ReLU')
    my_layer = DenseLayer(input_dim=input_dim, output_dim=output_dim)
    my_model2.insert_layer(my_layer, 0)
    my_model2.remove_layer(0)
    my_model2.validate_model()


if __name__ == '__main__':
    test_model()
