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
        Activation = activations[activation]
        self.activation = Activation()
        self.learning_rate = learning_rate
        self.lam = lam
        self.alpha = alpha

    def init_weights(self, n: int) -> None:
        # init weights accordingly to activation function
        self.W = self.activation.init_weights(n, self.units)
        self.b = np.zeros(self.units)

        # init momentum inertia to 0
        self.momentum_w = np.zeros_like(self.W)
        self.momentum_b = np.zeros_like(self.b)

    def store_best(self) -> None:
        # store best weights
        self.best_W = self.W.copy()
        self.best_b = self.b.copy()

    def load_best(self) -> None:
        # restore best weights
        self.W = self.best_W.copy()
        self.b = self.best_b.copy()

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.out = X  # save the output of the net for backprop
        self.net = X @ self.W + self.b

        return self.activation(self.net)

    def backward(self, dloss: np.ndarray) -> np.ndarray:
        delta = dloss * self.activation.derivative(self.net)

        # compute the delta for the previous layer
        delta_out = delta @ self.W.T

        # compute gradients
        gradient_w = self.out.T @ delta / delta.shape[0]
        gradient_b = np.sum(delta, axis=0) / delta.shape[0]

        # compute delta w and b for momentum
        delta_w = self.learning_rate * gradient_w
        delta_b = self.learning_rate * gradient_b

        # regularization term
        penalty = self.lam * self.W

        # momentum terms
        self.momentum_w = self.alpha * self.momentum_w + delta_w
        self.momentum_b = self.alpha * self.momentum_b + delta_b

        # update weights and bias
        self.W -= self.momentum_w + penalty
        self.b -= self.momentum_b

        return delta_out
