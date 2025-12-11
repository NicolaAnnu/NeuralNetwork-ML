import numpy as np

from neural.layer import Layer


class Network:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "logistic",
        learning_rate: float = 0.01,
        lam: float = 0.0001,  # regularization
        alpha: float = 0.5,  # momentum
        tol: float = 1e-5,
        batch_size: int = 10,
        shuffle: bool = False,
        max_iter: int = 200,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.layers = [
            Layer(
                units=units,
                activation=activation,
                learning_rate=learning_rate,
                lam=lam,
                alpha=alpha,
            )
            for units in hidden_layer_sizes
        ]

        self.activation = activation
        self.learning_rate = learning_rate
        self.lam = lam
        self.alpha = alpha
        self.tol = tol
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_iter = max_iter

    def init_weights(self, input_size):
        for layer in self.layers:
            layer.init_weights(input_size)
            input_size = layer.units

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = X.copy()
        for layer in self.layers:
            out = layer.forward(out)

        return out

    def backward(self, dloss: np.ndarray) -> None:
        for layer in reversed(self.layers):
            dloss = layer.backward(dloss)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: None | np.ndarray = None,
        y_val: None | np.ndarray = None,
    ):
        # initialize weights
        self.init_weights(X.shape[1])

        if self.batch_size == -1:
            self.batch_size = X.shape[0]

        # indices to shuffle samples
        indices = np.arange(X.shape[0])

        self.loss_curve = []
        self.val_loss_curve = []

        best_loss = np.inf
        stop_counter = 0
        for _ in range(self.max_iter):
            # shuffle the indices
            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(0, y.shape[0], self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                out = self.forward(X[batch_idx])
                error = out - y[batch_idx]

                # backpropagation
                self.backward(2 * error / error.shape[1])

            # epoch loss
            out = self.forward(X)
            self.loss_curve.append(np.mean((out - y) ** 2))

            if X_val is not None:
                out = self.forward(X_val)
                self.val_loss_curve.append(np.mean((out - y_val) ** 2))

            # stopping criteria
            if abs(best_loss - self.loss_curve[-1]) < self.tol:
                stop_counter += 1
            else:
                best_loss = self.loss_curve[-1]
                stop_counter = 0

            if stop_counter == 10:
                break

    @property
    def loss(self) -> float:
        return self.loss_curve[-1]


class Classifier(Network):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "logistic",
        learning_rate: float = 0.01,
        lam: float = 0.0001,
        alpha: float = 0.5,
        tol: float = 1e-5,
        batch_size: int = 10,
        shuffle: bool = False,
        max_iter: int = 200,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            lam=lam,
            alpha=alpha,
            tol=tol,
            batch_size=batch_size,
            shuffle=shuffle,
            max_iter=max_iter,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: None | np.ndarray = None,
        y_val: None | np.ndarray = None,
    ):
        # add output layer with one logistic unit
        output = Layer(
            units=1,
            activation="logistic",
            learning_rate=self.learning_rate,
            lam=self.lam,
            alpha=self.alpha,
        )
        self.layers.append(output)

        if y_val is not None:
            super().fit(X, y.reshape(-1, 1), X_val, y_val.reshape(-1, 1))
        else:
            super().fit(X, y.reshape(-1, 1), X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.round(self.forward(X)[:, 0])


class Regressor(Network):
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "logistic",
        learning_rate: float = 0.01,
        lam: float = 0.0001,
        alpha: float = 0.5,
        tol: float = 1e-5,
        batch_size: int = 10,
        shuffle: bool = False,
        max_iter: int = 200,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            lam=lam,
            alpha=alpha,
            tol=tol,
            batch_size=batch_size,
            shuffle=shuffle,
            max_iter=max_iter,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: None | np.ndarray = None,
        y_val: None | np.ndarray = None,
    ):
        # see if output should be 1-Dimensional or N-Dimensional
        n_outputs = 1 if (len(y.shape) == 1) else y.shape[1]

        # add the output layer with
        output = Layer(
            units=n_outputs,
            activation="linear",
            learning_rate=self.learning_rate,
            lam=self.lam,
            alpha=self.alpha,
        )
        self.layers.append(output)

        super().fit(X, y, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
