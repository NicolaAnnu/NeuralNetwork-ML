import numpy as np

from neural.activations import activations


class Layer:
    def __init__(
        self,
        units: int,
        activation: str = "logistic",
        learning_rate: float = 0.01,
        lam: float = 0.0001,
        alpha: float = 0.5,
    ) -> None:
        self.units = units
        self.activation = activations[activation]
        self.learning_rate = learning_rate
        self.lam = lam
        self.alpha = alpha

    def init_weights(self, n: int) -> None:
        self.W = np.random.normal(0, 1, (n, self.units))
        self.b = np.zeros(self.units)

        # for momentum
        self.weight_gradient_old = np.zeros_like(self.W)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out = X
        self.net = X @ self.W + self.b

        return self.activation[0](self.net)

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        delta = dloss * self.activation[1](self.net)

        # compute the delta for the previous layer
        delta_out = delta @ self.W.T

        # compute gradients
        weights_gradient = self.out.T @ delta
        bias_gradient = np.sum(delta, axis=0)

        # regularization term
        penalty = 2 * self.lam * self.W

        # momentum
        momentum = self.alpha * self.weight_gradient_old

        # update weights and bias through learning rule
        self.W -= self.learning_rate * (weights_gradient + momentum) + penalty
        self.b -= self.learning_rate * bias_gradient

        self.weight_gradient_old = weights_gradient

        return delta_out
