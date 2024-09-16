# test_neuron.py

import pytest
import numpy as np
from ..neuron import DenseNeuron
from ..activation_function import SigmoidActFunc

# Helper function to create a DenseNeuron with predefined settings


def create_test_neuron(input_dim=3, learning_rate=1e-2):
    return DenseNeuron(
        input_dim=input_dim,
        name='TestNeuron',
        learning_rate=learning_rate,
        act_func=SigmoidActFunc(),
        quiet=True
    )


def test_initialization():
    neuron = create_test_neuron()
    assert neuron.input_dim == 3
    assert neuron.learning_rate == 1e-2
    assert isinstance(neuron.act_func, SigmoidActFunc)
    assert neuron.current_input is not None
    assert neuron.w.shape[0] == 3
    assert neuron.b is not None


def test_compute_output():
    neuron = create_test_neuron()
    input_data = np.array([0.5, -0.2, 0.1])
    output = neuron.compute_output(input_data)

    # Expected output can be manually calculated if needed
    # For Sigmoid activation function:
    expected_result = neuron.act_func.compute(
        np.dot(neuron.w, input_data) + neuron.b)

    assert np.isclose(output, expected_result, atol=1e-5)


def test_backpropagation():
    neuron = create_test_neuron()
    input_data = np.array([0.5, -0.2, 0.1])
    output = neuron.compute_output(input_data)

    dL_da = 0.1  # Dummy gradient
    delta_x = neuron.backpropagation(dL_da)

    # Expected calculations for gradients:
    expected_derivative = neuron.act_func.derivative(
        neuron.current_result, neuron.current_output)

    expected_delta_b = dL_da * expected_derivative
    expected_delta_w = expected_delta_b * input_data
    expected_delta_x = expected_delta_b * neuron.w

    assert np.allclose(neuron.delta_b, expected_delta_b, atol=1e-5)
    assert np.allclose(neuron.delta_w, expected_delta_w, atol=1e-5)
    assert np.allclose(delta_x, expected_delta_x, atol=1e-5)


def test_update_parameters():
    neuron = create_test_neuron()
    input_data = np.array([0.5, -0.2, 0.1])
    neuron.compute_output(input_data)

    dL_da = 0.1
    neuron.backpropagation(dL_da)

    neuron.update_parameters()

    assert neuron.w is not None
    assert neuron.b is not None
    assert np.allclose(neuron.delta_w, np.zeros(neuron.input_dim), atol=1e-5)
    assert np.isclose(neuron.delta_b, 0, atol=1e-5)
    assert neuron.backpropagation_count == 0


def test_reset_parameters():
    neuron = create_test_neuron()
    input_data = np.array([0.5, -0.2, 0.1])
    neuron.compute_output(input_data)

    dL_da = 0.1
    neuron.backpropagation(dL_da)

    neuron.update_parameters()

    neuron.reset_parameters()

    assert neuron.delta_w is not None
    assert neuron.delta_b == 0
    assert neuron.backpropagation_count == 0


if __name__ == '__main__':
    pytest.main()
