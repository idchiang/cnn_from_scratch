import pytest
import numpy as np
from sklearn.metrics import log_loss as sklearn_log_loss
from ..loss_function import log_loss

# Test cases
test_cases_log_loss = [
    [np.array([1., 0.]), np.array([1., 0.])],
    [np.array([1., 0.]), np.array([1., 1.])],
    [np.array([1., 0.]), np.array([0., 0.])],
    [np.array([1., 0., 1, 0]), np.array([1., 0., 1., 0.])],
    [np.array([1., 1., 0, 0]), np.array([0.9, 0.8, 0.4, 0.1])],
    [np.array([1., 0., 1, 0]), np.array([0.8, 0.3, 0.7, 0.2])],
    [np.array([1., 0., 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1]),
     np.array([0.85, 0.45, 0.93, 0.45, 0.08, 0.0, 0.06, 0.37, 0.89, 0.71, 0.11, 0.72, 0.84, 0.88, 0.21, 0.59, 0.28, 0.06, 0.54, 0.78])],
    [np.array([1., 0., 1., 0.]), np.array([0., 1., 0., 1.])]
]

# Add expected values to test cases
for i, test_case in enumerate(test_cases_log_loss):
    y_true, y_pred = test_case
    expected = sklearn_log_loss(y_true, y_pred)
    test_cases_log_loss[i].append(expected)

# Note:
#    np.finfo(np.float32).eps = 1.1920929e-07
#    np.finfo(np.float64).eps = 2.220446049250313e-16


def test_log_loss():
    for i, (y_true, y_pred, expected) in enumerate(test_cases_log_loss):
        result = log_loss(y_true, y_pred)
        assert np.isclose(
            result, expected, atol=1e-7), f"Test case {i+1} failed: {result} != {expected}"
        if __name__ == '__main__':
            print(f"Test case {i+1} passed.")


if __name__ == '__main__':
    test_log_loss()
