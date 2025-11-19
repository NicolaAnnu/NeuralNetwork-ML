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
        self.units = [Neuron(activation, learning_rate) for _ in range(n_units)]
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        for u in self.units:
            u.init_weights(n)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros(len(self.units))
        for i, u in enumerate(self.units):
            out[i] = u.forward(X)

        return out

    def backward(self, delta: np.ndarray) -> np.ndarray:
        deltas = np.zeros(len(self.units))
        for i, u in enumerate(self.units):
            old_W = u.W.copy()
            deltas[i] = u.backward(delta)
            deltas[i] = np.sum(deltas[i] * old_W)
            delta = np.array([deltas[i]])

        return deltas
