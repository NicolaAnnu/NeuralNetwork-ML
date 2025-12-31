import numpy as np

from neural.convergence import methods
from neural.layer import Layer


class Network:
    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation: str = "logistic",
        learning_rate: float = 0.01,
        lam: float = 0.0001,
        alpha: float = 0.5,
        shuffle: bool = False,
        batch_size: int = 64,
        convergence: str = "loss_convergence",
        patience: int = 10,
        limit: float = -np.inf,
        max_iter: int = 500,
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
        self.shuffle = shuffle
        self.batch_size = batch_size

        criteria = methods[convergence]
        self.convergence = criteria(patience, limit)
        self.patience = patience
        self.limit = limit

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
        metric,
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
        self.score_curve = []

        if X_val is not None:
            self.val_loss_curve = []
            self.val_score_curve = []

        best_epoch = 0
        for epoch in range(self.max_iter):
            # shuffle the indices
            if self.shuffle:
                np.random.shuffle(indices)

            for i in range(0, y.shape[0], self.batch_size):
                batch_idx = indices[i : i + self.batch_size]
                out = self.forward(X[batch_idx])
                error = out - y[batch_idx]

                # backpropagation
                self.backward(2 * error / error.shape[1])

            # training loss and score
            out = self.forward(X)
            loss = np.mean((out - y) ** 2)
            self.loss_curve.append(loss)

            pred = self.predict(X)
            score = metric(y, pred)
            self.score_curve.append(score)

            # validation loss and score
            if X_val is not None:
                out = self.forward(X_val)
                val_loss = np.mean((out - y_val) ** 2)
                self.val_loss_curve.append(val_loss)

                pred = self.predict(X_val)
                val_score = metric(y_val, pred)
                self.val_score_curve.append(val_score)

            # convergence
            if not self.convergence.should_stop(loss, val_loss):
                if self.convergence.restore_weights and self.convergence.counter == 0:
                    best_epoch = epoch
                    for l in self.layers:
                        l.store_best()
            else:
                if self.convergence.restore_weights:
                    for l in self.layers:
                        l.load_best()

                    if best_epoch > 0:
                        self.loss_curve = self.loss_curve[:best_epoch]
                        self.val_loss_curve = self.val_loss_curve[:best_epoch]
                        self.score_curve = self.loss_curve[:best_epoch]
                        self.val_score_curve = self.val_loss_curve[:best_epoch]

                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

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
        shuffle: bool = False,
        batch_size: int = 64,
        convergence: str = "loss_convergence",
        patience: int = 10,
        limit: float = -np.inf,
        max_iter: int = 500,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            lam=lam,
            alpha=alpha,
            shuffle=shuffle,
            batch_size=batch_size,
            convergence=convergence,
            patience=patience,
            limit=limit,
            max_iter=max_iter,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric,
        X_val: None | np.ndarray = None,
        y_val: None | np.ndarray = None,
    ):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if y_val is not None:
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)

        # add output layer with one logistic unit
        output = Layer(
            units=1,
            activation="logistic",
            learning_rate=self.learning_rate,
            lam=self.lam,
            alpha=self.alpha,
        )
        self.layers.append(output)
        super().fit(X, y, metric, X_val, y_val)

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
        shuffle: bool = False,
        batch_size: int = 64,
        convergence: str = "loss_convergence",
        patience: int = 10,
        limit: float = -np.inf,
        max_iter: int = 500,
    ) -> None:
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            learning_rate=learning_rate,
            lam=lam,
            alpha=alpha,
            shuffle=shuffle,
            batch_size=batch_size,
            convergence=convergence,
            patience=patience,
            limit=limit,
            max_iter=max_iter,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        metric,
        X_val: None | np.ndarray = None,
        y_val: None | np.ndarray = None,
    ):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if y_val is not None:
            if len(y_val.shape) == 1:
                y_val = y_val.reshape(-1, 1)

        # add the output layer with
        output = Layer(
            units=y.shape[1],
            activation="linear",
            learning_rate=self.learning_rate,
            lam=self.lam,
            alpha=self.alpha,
        )
        self.layers.append(output)
        super().fit(X, y, metric, X_val, y_val)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)
