import numpy as np

from neural.activations import activations


class Layer:
    def __init__(
        self,
        n_units: int,
        activation: str = "logistic",
        learning_rate: float = 0.01,
    ) -> None:
        self.n_units = n_units
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        self.W = np.random.normal(0, 1, (n, self.n_units))
        self.b = np.zeros(self.n_units)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out_prev = X
        self.net = X.T @ self.W + self.b

    def backward(self, error: np.ndarray) -> np.ndarray:
        deltas = np.zeros((error.shape[0], self.n_units))
        for i, u in enumerate(self.units):
            deltas[:, i] = u.backward(error[:, i])

        return deltas

    def weights(self):
        return np.stack([u.W for u in self.units], axis=1)
