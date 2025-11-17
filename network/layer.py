import numpy as np

from network.activations import activations
from network.neuron import Neuron


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
        out = np.zeros((X.shape[0], len(self.units)))
        for i, u in enumerate(self.units):
            out[:, i] = u(X)

        return out

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[:, 0]

    def update_weights(self, errors: np.ndarray) -> np.ndarray:
        deltas = np.zeros((len(self.units), errors.size))
        for i, u in enumerate(self.units):
            deltas[i] = u.update_weights(errors)

        return deltas[0, :]
