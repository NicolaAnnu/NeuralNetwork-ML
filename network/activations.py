import numpy as np


# logistic
def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_d(x):
    x1 = logistic(x)
    return x1 * (1 - x1)


def tanh_d(x):
    return 1 - np.tanh(x) ** 2


# dictionary of activation functions
# every field returns a tuple with
# 1. the activation function
# 2. its derivative
# 3. the hypothesis function associated with that activation (for predictions)

activations = {
    "linear": (lambda x: x, lambda _: 1, lambda x: x),
    "logistic": (logistic, logistic_d, np.round),
    "tanh": (np.tanh, tanh_d, np.sign),
}
