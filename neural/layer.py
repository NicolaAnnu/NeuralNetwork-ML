import numpy as np

from neural.activations import activations


class Layer:
    def __init__(
        self,
        units: int,
        activation: str = "logistic",
        learning_rate: float = 0.01,
    ) -> None:
        self.units = units
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        self.W = np.random.normal(0, 1, (n, self.units))
        self.b = np.zeros(self.units)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out = X
        self.net = X @ self.W + self.b

        return self.activation[0](self.net)

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        delta = dloss * self.activation[1](self.net)

        # compute gradients
        weights_gradient = self.out.T @ delta
        bias_gradient = np.sum(delta, axis=0)

        # update weights and bias through learning rule
        self.W -= self.learning_rate * weights_gradient
        self.b -= self.learning_rate * bias_gradient

        return delta @ self.W.T
