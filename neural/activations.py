from typing import Protocol

import numpy as np


class Activation(Protocol):
    def __call__(self, x: np.ndarray) -> np.ndarray: ...

    def derivative(self, x: np.ndarray) -> np.ndarray: ...

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray: ...


class Linear(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = 1 / np.sqrt(fan_in)
        return np.random.uniform(-limit, limit, (fan_in, fan_out))


class Logistic(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        x1 = self(x)
        return x1 * (1 - x1)

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = 1 / np.sqrt(fan_in)
        return np.random.uniform(-limit, limit, (fan_in, fan_out))


class Tanh(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        limit = 1 / np.sqrt(fan_in)
        return np.random.uniform(-limit, limit, (fan_in, fan_out))


class Relu(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))


class LeakyRelu(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, 0.01 * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0.01)

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))


class Elu(Activation):
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, np.exp(x) - 1)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1.0, np.exp(x))

    def init_weights(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.normal(0, np.sqrt(2 / fan_in), (fan_in, fan_out))


activations = {
    "linear": Linear(),
    "logistic": Logistic(),
    "tanh": Tanh(),
    "relu": Relu(),
    "leaky_relu": LeakyRelu(),
    "elu": Elu(),
}
