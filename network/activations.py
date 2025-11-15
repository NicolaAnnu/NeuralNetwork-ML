import numpy as np


# linear (for regression)
def linear(x):
    return x


def linear_d(_):
    return 1


# logistic
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_d(x):
    x1 = logistic(x)
    return x1 * (1 - x1)


# hyperbolic tangent
def tanh(x):
    return np.tanh(x)


def tanh_d(x):
    return 1 - np.tanh(x) ** 2


activations = {
    "linear": (linear, linear_d),
    "logistic": (logistic, logistic_d),
    "tanh": (tanh, tanh_d),
}
