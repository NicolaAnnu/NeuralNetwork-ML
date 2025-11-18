import numpy as np

from neural.activations import activations


class Neuron:
    def __init__(
        self,
        activation: str = "logistic",
        learning_rate: float = 0.01,
    ) -> None:
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        # randomly initialize weights and bias
        self.W = np.random.normal(0, 0.01, n)
        self.b = 0.0

    def __call__(self, X: np.ndarray) -> np.ndarray:
        # save for backpropagation
        self.out_prev = X
        self.net = self.b + X @ self.W

        return self.activation[0](self.net)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return self.__call__(X).reshape(-1, 1)

    def update_weights(self, error: np.ndarray) -> np.ndarray:
        delta = error * self.activation[1](self.net)

        # compute gradients
        weights_gradient = self.out_prev.T @ delta
        bias_gradient = np.sum(delta)

        # update weights and bias through learning rule
        self.W -= self.learning_rate * weights_gradient
        self.b -= self.learning_rate * bias_gradient

        return np.sum(delta, keepdims=True)
