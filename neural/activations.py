import numpy as np


# logistic
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_d(x):
    x1 = logistic(x)
    return x1 * (1 - x1)


def tanh_d(x):
    return 1 - np.tanh(x) ** 2


def relu(x):
    return np.maximum(0, x)


def relu_d(x):
    return (x > 0).astype(float)


def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)


def leaky_relu_d(x):
    return np.where(x > 0, 1, 0.01)


def elu(x):
    x = np.asarray(x)
    return np.where(x > 0, x, np.exp(x) - 1)


def elu_d(x):
    x = np.asarray(x)
    return np.where(x > 0, 1.0, np.exp(x))


# dictionary of activation functions
# every field returns a tuple with
# 1. the activation function
# 2. its derivative

activations = {
    "linear": (lambda x: x, lambda x: np.ones_like(x)),
    "logistic": (logistic, logistic_d),
    "tanh": (np.tanh, tanh_d),
    "relu": (relu, relu_d),
    "leaky_relu": (leaky_relu, leaky_relu_d),
    "elu": (elu, elu_d),
}
