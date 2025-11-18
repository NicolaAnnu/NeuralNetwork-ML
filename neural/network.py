import numpy as np

from neural.layer import Layer


class Network:
    def __init__(
        self,
        hidden_layer_sizes=(5,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        batch_size: int = 10,
        max_iter: int = 200,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = [
            Layer(n_units=n_units, activation=activation, learning_rate=learning_rate)
            for n_units in hidden_layer_sizes
        ]

        # initialize hidden layers weights
        for i in range(1, len(self.layers), 1):
            self.layers[i].init_weights(len(self.layers[i - 1].units))

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
        # initialize first layer weights
        self.layers[0].init_weights(X.shape[1])

        self.loss_curve = []
        for _ in range(self.max_iter):
            epoch_loss = 0.0
            for i in range(len(y)):
                out = self.forward(X[i : i + self.batch_size, :])
                errors = out - y[i : i + self.batch_size]
                delta = 2 * errors / self.batch_size
                self.backward(delta)

                epoch_loss += np.sum(errors**2) / self.batch_size
            self.loss_curve.append(epoch_loss)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    @property
    def loss(self):
        return self.loss_curve[-1]


class Classifier(Network):
    def __init__(
        self,
        hidden_layer_sizes=(5,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        batch_size: int = 10,
        max_iter: int = 200,
    ) -> None:
        super().__init__(
            hidden_layer_sizes, activation, learning_rate, batch_size, max_iter
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        output = Layer(1, activation="logistic", learning_rate=self.learning_rate)
        output.init_weights(len(self.layers[-1].units))
        self.layers.append(output)

        super().fit(X, y)


class Regressor(Network):
    def __init__(
        self,
        hidden_layer_sizes=(5,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        batch_size: int = 10,
        max_iter: int = 200,
    ) -> None:
        super().__init__(
            hidden_layer_sizes, activation, learning_rate, batch_size, max_iter
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        output = Layer(1, activation="linear", learning_rate=self.learning_rate)
        output.init_weights(len(self.layers[-1].units))
        self.layers.append(output)

        super().fit(X, y)
