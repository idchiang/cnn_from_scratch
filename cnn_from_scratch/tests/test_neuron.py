import pytest
import numpy as np
from ..activation_function import *
from ..neuron import DenseNeuron

# Only flow check now.


def test_neuron():
    n1 = DenseNeuron(input_dim=10, act_func=SigmoidActFunc())
    print('# Check if DenseNeuron.__init__() set things as expected...')
    for key in n1.__dict__:
        print('     ', key, '=', n1.__dict__[key])

    print('# Check if compute_output() works')
    x = np.ones(10)
    act_func = SigmoidActFunc()
    exp = act_func.compute(np.dot(n1.w, x) + n1.b)
    print('         ', n1.compute_output(x))
    print('     Exp:', exp)

    w = n1.w.copy()
    b = float(n1.b)

    print('# Check if backpropagation() works')
    n_count = 3
    dL_da_arr = np.random.randn(n_count)
    for i, dL_da in enumerate(dL_da_arr):
        print(f'## Propagate {i+1}, dL/da={dL_da}')
        delta_x = n1.backpropagation(dL_da)
        print('         self.delta_w = ', n1.delta_w)
        print('         self.delta_b = ', n1.delta_b)
        print('              delta_x = ', delta_x)
        print('  backpropagation_count = ', n1.backpropagation_count)
    assert n1.backpropagation_count == n_count
    print('w unchanged?', w == n1.w)
    print('b unchanged?', b == n1.b)
    print('         w = ', n1.w)
    print('         b = ', n1.b)
    print('# Check if update_parameters() works')
    n1.update_parameters()
    print('         w = ', n1.w)
    print('         b = ', n1.b)
    print('         self.delta_w = ', n1.delta_w)
    print('         self.delta_b = ', n1.delta_b)
    assert n1.backpropagation_count == 0
    n1.reset_parameters()
