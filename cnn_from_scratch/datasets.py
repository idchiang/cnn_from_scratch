# x_train, y_train, x_test, y_test = load_mnist()
import os, sys
import platform
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def load_mnist(overwrite=False, filename=''):
    if filename == '':
        if platform.system() == 'Windows':
            filename = 'C:/Users/jiang/workspace/cnn_from_scratch/data/mnist.npz'
        elif platform.system() == 'macOS':
            filename = '/Users/idchiang/working/cnn_from_scratch/data/mnist.npz'
    if overwrite or (not os.path.isfile(filename)):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28 * 28).astype(float) / 255.0
        x_test = x_test.reshape(-1, 28 * 28).astype(float) / 255.0
        y_train = keras.utils.to_categorical(y_train, num_classes=10)
        y_test = keras.utils.to_categorical(y_test, num_classes=10)
        np.savez(filename, x_train=x_train, x_test=x_test,
                 y_train=y_train, y_test=y_test)
    else:
        data = np.load(filename)
        x_train = data['x_train']
        y_train = data['y_train']
        x_test = data['x_test']
        y_test = data['y_test']
    return x_train, y_train, x_test, y_test


def show_mnist_digit(x, y=None, ax=None, overwrite_title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if overwrite_title is None:
        try:
            label = str(np.argmax(y))
        except:
            label = ''
    else:
        label = overwrite_title
    image = x.reshape(28, 28)
    ax.imshow(image, cmap='Greys')
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(labelleft=False, labelbottom=False)


def mnist_model_examination(trainer, x_data, y_data):
    plt.close('all')
    plt.ion()
    fig, axs = plt.subplots(5, 5, figsize=(8, 8))
    idxs = np.random.randint(len(x_data), size=25)
    for q in range(25):
        i, j = q // 5, q % 5
        a_pred = trainer.predict(x_data[idxs[q]])
        y_pred = np.argmax(a_pred)
        y_true = np.argmax(y_data[idxs[q]])
        title = f"true: {y_true}; pred: {y_pred}"
        show_mnist_digit(x_data[idxs[q]], y_data[idxs[q]],
                         axs[i, j], overwrite_title=title)
    fig.tight_layout()
