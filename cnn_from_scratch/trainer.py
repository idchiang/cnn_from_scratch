"""
# CNN Model Trainer object (Simple CNN focused for now)
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
from .model import CNN_Model
from .loss_function import *
# Main type of model trainer for our simple CNN.


class CNN_trainer():
    def __init__(self, model=None, loss_func=None, name='T0', quiet=True):
        # Sanity checks
        if loss_func is None:
            loss_func = LogLoss()
        assert loss_func.__class__.__base__ is LossFunction, f"CNN_Model.__init__() in {name}: loss_func must be a subclass of LossFunction()"
        # Inputs
        self.name = name
        self.quiet = quiet
        self.model = None
        if model is not None:
            self.set_model(model)
        self.loss_func = loss_func
        # Basics
        self.model_validated = False
        self.data_validated = False
        self.cross_validated = False
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        # History stuff
        self.reset_history()

    def set_data(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.validate_data()

    def validate_data(self):
        self.data_validated = False
        try:
            if len(self.x_train) != len(self.y_train):
                warnings.warn(
                    f"CNN_trainer.validate_data() in {self.name}: length of x_train and y_train don't match.", UserWarning)
            elif len(self.x_test) != len(self.y_test):
                warnings.warn(
                    f"CNN_trainer.validate_data() in {self.name}: length of x_test and y_test don't match.", UserWarning)
            elif self.x_train.shape[1] != self.x_test.shape[1]:
                warnings.warn(
                    f"CNN_trainer.validate_data() in {self.name}: dimension of x_train and x_test don't match.", UserWarning)
            elif self.y_train.shape[1] != self.y_test.shape[1]:
                warnings.warn(
                    f"CNN_trainer.validate_data() in {self.name}: dimension of y_train and y_test don't match.", UserWarning)
            elif (len(self.x_train.shape) != 2) or (len(self.y_train.shape) != 2) or (len(self.x_test.shape) != 2) or (len(self.y_test.shape) != 2):
                warnings.warn(
                    f"CNN_trainer.validate_data() in {self.name}: only 2-dim input allowed for x_train, y_train, x_test and y_test don't match.", UserWarning)
            else:
                self.data_validated = True
        except:
            warnings.warn(
                f"CNN_trainer.validate_data() in {self.name}: unexpected error", UserWarning)

    def set_model(self, model):
        self.model = model
        self.validate_model()

    def validate_model(self):
        try:
            self.model.validate_model()
            self.model_validated = self.model.model_validated
        except:
            self.model_validated = False

    def naive_model(self, model_input_dim, model_output_dim, hidden_dims=[50, 50],
                    hidden_act_func=None,
                    output_act_func=None):
        model = CNN_Model(model_input_dim=model_input_dim,
                          model_output_dim=model_output_dim,
                          name='NaiveCNN',
                          loss_func=self.loss_func,
                          quiet=self.quiet)
        model.naive_model(hidden_dims=hidden_dims, hidden_act_func=hidden_act_func,
                          output_act_func=output_act_func)
        self.set_model(model)

    def cross_validate(self):
        self.cross_validated = False
        if not self.data_validated:
            self.validate_data()
        if not self.data_validated:
            warnings.warn(
                f"CNN_trainer.cross_validate() in {self.name}: data validation failed. missing data or dimension mismatch.", UserWarning)
            return
        if not self.model_validated:
            self.validate_model()
        if not self.data_validated:
            warnings.warn(
                f"CNN_trainer.cross_validate() in {self.name}: model validation failed. mismatched dimension.", UserWarning)
            return
        # validate data, validate model, then cross-validate
        if self.model.model_input_dim != self.x_train.shape[1]:
            warnings.warn(
                f"CNN_trainer.cross_validate() in {self.name}: model_input_dim mismatch between model and data", UserWarning)
        elif self.model.model_output_dim != self.y_train.shape[1]:
            warnings.warn(
                f"CNN_trainer.cross_validate() in {self.name}: model_output_dim mismatch between model and data", UserWarning)
        else:
            self.cross_validated = True

    def train_model(self, n_epoch=20, batch_size=200, learning_rate=1e-2, reset_history=False, do_acc=False):
        # Sanity check
        if not self.cross_validated:
            self.cross_validate()
        if not self.cross_validated:
            raise ValueError(
                f"CNN_trainer.train_model() in {self.name}: cross validation failed")
        if batch_size > len(self.x_train):
            batch_size = len(self.x_train)
            warnings.warn(
                f"CNN_trainer.train_model() in {self.name}: batch_size larger than data length. Setting it to {len(self.x_train)}")
        if batch_size > 1:
            warnings.warn("CNN_trainer.train_model() in {self.name}: unsolved issue with batch_size > 1. Suggesting batch_size = 1 for now.")
        # Learning rate...
        self.model.set_learning_rate(learning_rate)
        # Reset?
        if reset_history:
            self.reset_history()
            # Initial Loss
            self.calc_print_status(0, n_epoch, do_acc=do_acc)
        N_train = len(self.x_train)
        perm_arr = np.random.permutation(N_train)
        # Train
        for j in range(n_epoch):
            # Shuffle?
            perm_arr = np.random.permutation(N_train)
            for i, i_train in enumerate(perm_arr):
                # Fowward computation & Backpropagation -> this part could be made parallel?
                self.model.compute_output(self.x_train[i_train])
                self.model.backpropagation(self.y_train[i_train])
                # Batch update?
                if (i + 1) % batch_size == 0:
                    self.model.update_parameters()
            if N_train % batch_size > 0:
                self.model.update_parameters()
            # Calc + print history
            self.calc_print_status(j + 1, n_epoch, do_acc=do_acc)
        # Visualize
        self.plot_history(do_acc=do_acc)

    def calc_print_status(self, i_history, n_epoch, do_acc=False):
        train_loss, train_acc = self.calculate_train_loss(do_acc=do_acc)
        test_loss, test_acc = self.calculate_test_loss(do_acc=do_acc)
        current_time = datetime.now()
        time_stamp = current_time.strftime("%H:%M:%S")
        #
        if do_acc:
            print(
                f"# CNN_trainer {self.name}, Epoch: {i_history}/{n_epoch} - {time_stamp}\n",
                f"         - Train loss = {train_loss:.3f}; Test loss = {test_loss:.3f}\n",
                f"         - Train acc = {train_acc:.1f}%; Test acc = {test_acc:.1f}%")
        else:
            print(
                f"# CNN_trainer {self.name}, Epoch: {i_history}/{n_epoch} - {time_stamp}\n",
                f"         - Train loss = {train_loss:.3f}; Test loss = {test_loss:.3f}")

    def calculate_train_loss(self, do_acc=False):
        total_loss = 0.
        total_correct = 0
        for i, x_train_i in enumerate(self.x_train):
            a_pred = self.model.compute_output(x_train_i)
            loss = self.loss_func.compute(self.y_train[i], a_pred)
            total_loss += loss
            if do_acc:
                total_correct += int(
                    np.argmax(self.y_train[i]) == np.argmax(a_pred))
        avg_loss = total_loss / len(self.x_train)
        self.history_train_loss.append(avg_loss)
        acc = np.nan
        if do_acc:
            acc = 100 * float(total_correct) / len(self.x_train)
            self.history_train_acc.append(acc)
        return avg_loss, acc

    def calculate_test_loss(self, do_acc=False):
        total_loss = 0.
        total_correct = 0
        for i, x_test_i in enumerate(self.x_test):
            a_pred = self.model.compute_output(x_test_i)
            loss = self.loss_func.compute(self.y_test[i], a_pred)
            total_loss += loss
            if do_acc:
                total_correct += int(
                    np.argmax(self.y_test[i]) == np.argmax(a_pred))
        avg_loss = total_loss / len(self.x_test)
        self.history_test_loss.append(avg_loss)
        acc = np.nan
        if do_acc:
            acc = 100 * float(total_correct) / len(self.x_test)
            self.history_test_acc.append(acc)
        return avg_loss, acc

    def plot_history(self, do_acc=False):
        if len(self.history_train_loss) == 0:
            warnings.warn(
                f"CNN_trainer.plot_history() in {self.name}: There is no history to plot!!", UserWarning)
            return
        plt.close('all')
        plt.ion()
        if do_acc:
            fig, axs = plt.subplots(nrows=2)
            ax = axs[0]
        else:
            fig, axs = plt.subplots()
            ax = axs
        # Loss history
        ax.plot(np.arange(len(self.history_train_loss)), self.history_train_loss,
                color='tab:blue', label='Train loss')
        ax.plot(np.arange(len(self.history_train_loss)), self.history_test_loss,
                color='tab:orange', label='Test loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('N')
        ax.set_title(self.model.name)
        ax.legend()
        # Accuracy history
        if do_acc:
            ax = axs[1]
            ax.plot(np.arange(len(self.history_train_loss)), np.array(self.history_train_acc),
                    color='tab:blue', label='Train accuracy')
            ax.plot(np.arange(len(self.history_train_loss)), np.array(self.history_test_acc),
                    color='tab:orange', label='Test accuracy')
            ax.set_ylabel('Accuracy [%]')
            ax.set_xlabel('N')
            ax.legend()
        fig.show()

    def reset_history(self):
        self.history_train_loss = []
        self.history_test_loss = []
        self.history_train_acc = []
        self.history_test_acc = []

    def predict(self, x_input):
        if len(x_input.shape) == 1:
            assert len(
                x_input) == self.model.model_input_dim, f"CNN_trainer.predict() in {self.name}: x_input wrong dimension"
        elif len(x_input.shape) == 2:
            assert x_input.shape[
                1] == self.model.model_input_dim, f"CNN_trainer.predict() in {self.name}: x_input wrong dimension"
        else:
            assert False, f"CNN_trainer.predict() in {self.name}: Only 1D or 2D inputs allowed"
        #
        if len(x_input.shape) == 1:
            a_pred = self.model.compute_output(x_input)
        elif len(x_input.shape) == 2:
            a_pred = []
            for x_input_elem in x_input:
                a_pred.append(self.model.compute_output(x_input_elem))
            a_pred = np.array(a_pred)
        return a_pred

    def savefile(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def loadfile(self, filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)
