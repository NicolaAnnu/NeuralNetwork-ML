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
        out = np.zeros(self.n_units)
        for i, u in enumerate(self.units):
            out[i] = u.forward(X)

        return out

    def backward(self, error: np.ndarray) -> np.ndarray:
        deltas = np.zeros(self.n_units)
        for i, u in enumerate(self.units):
            deltas[i] = u.backward(error)

        return deltas

    def weights(self) -> np.ndarray:
        weights_per_unit = len(self.units[0].W)
        W = np.zeros((self.n_units, weights_per_unit))
        for i, u in enumerate(self.units):
            W[i] = u.W

        return W
