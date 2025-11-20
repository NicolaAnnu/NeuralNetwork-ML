import numpy as np

from neural.activations import activations
from neural.neuron import Neuron


class Layer:
    def __init__(
        self,
        n_units: int,
        activation: str = "logistic",
        learning_rate: float = 0.01,
    ) -> None:
        self.n_units = n_units
        self.units = [Neuron(activation, learning_rate) for _ in range(n_units)]
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        for u in self.units:
            u.init_weights(n)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], self.n_units))
        for i, u in enumerate(self.units):
            out[:, i] = u(X)

        return out

    def backward(self, error: np.ndarray) -> np.ndarray:
        deltas = np.zeros((error.shape[0], self.n_units))
        for i, u in enumerate(self.units):
            deltas[:, i] = u.backward(error[:, i])

        return deltas

    def weights(self):
        return np.stack([u.W for u in self.units], axis=1)
