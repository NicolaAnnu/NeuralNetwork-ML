import numpy as np

from network.activations import activations
from network.neuron import Neuron


class Layer:
    def __init__(
        self,
        n_units: int,
        activation: str = "logistic",
        learning_rate: float = 0.01,
    ) -> None:
        self.units = [Neuron(activation, learning_rate) for _ in range(n_units)]
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        for u in self.units:
            u.init_weights(n)

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], len(self.units)))
        for i, u in enumerate(self.units):
            out[:, i] = u.output(X)

        return out

    def update_weights(self, X: np.ndarray, error: np.ndarray):
        deltas = np.zeros((X.shape[0], len(self.units)))
        for i, u in enumerate(self.units):
            deltas[:, i] = u.update_weights(X, error)


def init(layers: list[Layer], input_size: int):
    for l in layers:
        l.init_weights(input_size)
        input_size = len(l.units)


def backpropagation(layers: list[Layer], out: np.ndarray, error: np.ndarray) -> None:
    for l in reversed(layers):
        l.update_weights(out, error)


def forward(layers, X):
    for l in layers:
        out = l.forward(X)
        X = out

    return out


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    n_features = 5
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=100,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
    ]

    lr = 0.01
    layers = [
        Layer(n_units=1, activation="logistic", learning_rate=lr) for _ in range(1)
    ]
    layers.append(Layer(n_units=1, activation="logistic", learning_rate=lr))
    init(layers, X.shape[1])

    batch_size = 1
    for epoch in range(200):
        for i in range(0, len(y), batch_size):
            out = forward(layers, X)
            error = out - y[i : i + batch_size]
            backpropagation(layers, out, error)

    out = np.round(forward(layers, X))[:, 0]
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")
