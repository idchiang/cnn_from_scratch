import pytest
from ..datasets import load_mnist


def test_mnist():
    x_train, y_train, x_test, y_test = load_mnist()
    assert x_train.shape == (60000, 784)
    assert y_train.shape == (60000, 10)
    assert x_test.shape == (10000, 784)
    assert y_test.shape == (10000, 10)
    assert x_train.max() == 1.0
    assert x_train.min() == 0.0
    assert x_test.max() == 1.0
    assert x_test.min() == 0.0
    assert y_train.max() == 1.0
    assert y_train.min() == 0.0
    assert y_test.max() == 1.0
    assert y_test.min() == 0.0
