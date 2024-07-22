"""
# Neuron object. (DenseLayer-focused)
"""

# Neuron for Dense Layer.
# Not sure if we need other types of neurons. Just keep the name clear. 
class DenseNeuron():
    def __init__(self, input_dim, activation_funtion='sigmoid'):
        pass
    """
    attribute input_dim
    attribute self.w (np.ndarray, input_dim -> 1), self.b (float)
    attribute self.delta_w, self.delta_b  # save for batch update
    attribute action_funtion
    attribute current_input = None  # needed for backpropagation
    attribute current_output = None
    method __init__(input_dim, activation_funtion)
    method compute_data(input_data)
    method backpropagation(diff)  # within a batch, update self.delta_w, self.delta_b
    method update_parameters()  # at the end of a batch, update self.delta_w, self.delta_b with self.delta_w, self.delta_b
    method reset_parameters()  # details differ for different layers
    """