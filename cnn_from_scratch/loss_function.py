"""
# Loss functions
"""
# Log Loss (for classification)
# y_true, y_pred AREN'T interchangeable!!

import numpy as np
from abc import ABC, abstractmethod


class LossFunction(ABC):
    @abstractmethod
    def compute(self, a_true, a_pred):
        pass

    @abstractmethod
    def derivative(self, a_true, a_pred):
        pass


class LogLoss(LossFunction):
    def compute(self, a_true, a_pred):
        # Sanity checks: might implement them elsewhere
        self.sanity_checks(a_true, a_pred)
        # clip to avoid y == 0 & y == 1
        a_pred_clipped = self.clip_a_pred(a_pred)
        loss = -np.mean(a_true * np.log(a_pred_clipped) +
                        (1 - a_true) * np.log(1 - a_pred_clipped))
        return loss

    def derivative(self, a_true, a_pred):
        # Sanity checks: might implement them elsewhere
        self.sanity_checks(a_true, a_pred)
        # clip to avoid y == 0 & y == 1
        a_pred_clipped = self.clip_a_pred(a_pred)
        dL_da = (-a_true / a_pred_clipped + (1 - a_true) /
                 (1 - a_pred_clipped)) / len(a_true)
        return dL_da

    def sanity_checks(self, a_true, a_pred):
        assert len(a_true) == len(a_pred)
        assert np.all((a_true == 0) | (a_true == 1)), "y_true isn't binary"
        assert np.all((a_pred >= 0) & (a_pred <= 1)
                      ), "y_pred outside (0.0, 1.0)"

    def clip_a_pred(self, a_pred):
        if a_pred.dtype not in [np.float32, np.float64, float]:
            a_pred = a_pred.astype(np.float64)
        epsilon = np.finfo(a_pred.dtype).eps
        # clip to avoid y == 0 & y == 1
        a_pred_clipped = np.clip(a_pred, epsilon, 1 - epsilon)
        return a_pred_clipped
