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
        if self.activation == activations["relu"]:
            # He initialization
            self.W = np.random.normal(0, np.sqrt(2 / n), (n, self.units))
        else:
            # Glorot-Xavier initialization
            limit = 1 / np.sqrt(n)
            self.W = np.random.uniform(-limit, limit, (n, self.units))

        self.b = np.zeros(self.units)

        # for momentum
        self.momentum_w = np.zeros_like(self.W)
        self.momentum_b = np.zeros_like(self.b)

    def store_best(self) -> None:
        self.best_W = self.W.copy()
        self.best_b = self.b.copy()

    def load_best(self) -> None:
        self.W = self.best_W.copy()
        self.b = self.best_b.copy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out = X
        self.net = X @ self.W + self.b

        return self.activation[0](self.net)

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        delta = dloss * self.activation[1](self.net)

        # compute the delta for the previous layer
        delta_out = delta @ self.W.T

        # compute gradients
        gradient_w = self.out.T @ delta / delta.shape[0]
        gradient_b = np.sum(delta, axis=0) / delta.shape[0]

        # compute delta w and b for momentum
        delta_w = self.learning_rate * gradient_w
        delta_b = self.learning_rate * gradient_b

        # regularization term
        penalty = 2 * self.lam * self.W

        # momentum terms
        self.momentum_w = self.alpha * self.momentum_w + delta_w
        self.momentum_b = self.alpha * self.momentum_b + delta_b

        # update weights and bias
        self.W -= self.momentum_w + penalty
        self.b -= self.momentum_b

        return delta_out
