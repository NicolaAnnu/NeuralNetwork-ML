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
        self.W = np.random.normal(0, 1, n)
        self.b = 0.0

    def forward(self, X: np.ndarray) -> np.ndarray:
        # save for backpropagation
        self.out_prev = X
        self.net = self.b + X @ self.W

        out = np.zeros(1)
        out[0] = self.activation[0](self.net)

        return out

    def __call__(self, X: np.ndarray) -> float:
        return self.activation[0](self.b + X @ self.W)

    def backward(self, error: np.ndarray) -> float:
        delta = np.outer(error, self.activation[1](self.net))[0]

        # compute gradients
        weights_gradient = self.out_prev * delta
        bias_gradient = delta

        # update weights and bias through learning rule
        self.W -= self.learning_rate * weights_gradient
        self.b -= self.learning_rate * bias_gradient

        return delta
