import numpy as np
import numpy.linalg as la

from neural.layer import Layer


class Network:
    def __init__(
        self,
        hidden_layer_sizes=(5,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        lam: float = 0.0001,  # regularization
        alpha: float = 0.5,  # momentum
        batch_size: int = 10,
        max_iter: int = 200,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = [
            Layer(
                units=units,
                activation=activation,
                learning_rate=learning_rate,
                lam=lam,
            )
            for units in hidden_layer_sizes
        ]

        # initialize hidden layers weights
        for i in range(1, len(self.layers), 1):
            self.layers[i].init_weights(self.layers[i - 1].units)

        self.activation = activation
        self.learning_rate = learning_rate
        self.lam = lam
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_iter = max_iter

    def forward(self, X: np.ndarray) -> np.ndarray:
        for l in self.layers:
            X = l.forward(X)

        return X

    def backward(self, dloss: np.ndarray) -> None:
        for l in reversed(self.layers):
            dloss = l.backward(dloss)

    def fit(self, X, y):
        # initialize first layer weights
        self.layers[0].init_weights(X.shape[1])

        self.loss_curve = []
        for _ in range(self.max_iter):
            epoch_loss = 0.0
            for i in range(0, len(y), self.batch_size):
                out = self.forward(X[i : i + self.batch_size, :])
                error = out - y[i : i + self.batch_size].reshape(-1, 1)

                # penalty term (regularization)
                penalty = 0.0
                for l in self.layers:
                    penalty += l.lam * la.norm(l.W, ord=2) ** 2

                dloss = 2 * error + penalty
                self.backward(dloss)

            self.loss_curve.append(epoch_loss / (len(y) / self.batch_size))

    @property
    def loss(self) -> float:
        return self.loss_curve[-1]


class Classifier(Network):
    def __init__(
        self,
        hidden_layer_sizes=(5,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        lam: float = 0.0001,
        alpha: float = 0.5,
        batch_size: int = 10,
        max_iter: int = 200,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            lam=lam,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        output = Layer(
            units=1,
            activation="logistic",
            learning_rate=self.learning_rate,
            lam=self.lam,
        )
        output.init_weights(self.layers[-1].units)
        self.layers.append(output)

        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.round(self.forward(X)[:, 0])


class Regressor(Network):
    def __init__(
        self,
        hidden_layer_sizes=(5,),
        activation: str = "logistic",
        learning_rate: float = 0.1,
        lam: float = 0.0001,
        alpha: float = 0.5,
        batch_size: int = 10,
        max_iter: int = 200,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            lam=lam,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        output = Layer(
            units=1,
            activation="linear",
            learning_rate=self.learning_rate,
            lam=self.lam,
        )
        output.init_weights(self.layers[-1].units)
        self.layers.append(output)

        super().fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[:, 0]
