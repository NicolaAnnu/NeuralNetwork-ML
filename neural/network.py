import numpy as np

from neural.layer import Layer


class Network:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        batch_size: int = 50,
        max_iter: int = 200,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = [
            Layer(n_units=n_units, activation=activation, learning_rate=learning_rate)
            for n_units in hidden_layer_sizes
        ]

        self.activation = activation
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter

    def forward(self, X: np.ndarray) -> np.ndarray:
        for l in self.layers[:-1]:
            X = l.forward(X)

        return self.layers[-1](X)

    def backward(self, delta: np.ndarray) -> None:
        for l in reversed(self.layers):
            delta = l.update_weights(delta)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_inputs = X.shape[1]
        for l in self.layers:
            l.init_weights(n_inputs)
            n_inputs = len(l.units)

        for epoch in range(self.max_iter):
            for i in range(len(y)):
                out = self.forward(X[i : i + self.batch_size, :])
                errors = out - y[i : i + self.batch_size]
                delta = 2 * errors / self.batch_size
                self.backward(delta)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
