import numpy as np

from network.activations import activations
from network.neuron import Neuron


class Layer:
    def __init__(
        self,
        n_units: int,
        activation: str = "logistic",
        learning_rate: float = 0.01,
        max_iter: int = 200,
    ) -> None:
        self.units = [Neuron(activation, learning_rate) for _ in range(n_units)]
        self.activation = activations[activation]
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def init_weights(self, n: int) -> None:
        for u in self.units:
            u.init_weights(n)

    def output(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], len(self.units)))
        for i, u in enumerate(self.units):
            out[:, i] = u.output(X)

        return out

    def update_weights(self, X: np.ndarray, error: np.ndarray):
        for u, e in zip(self.units, error):
            u.update_weights(X, e)


def connect_layers(layers: list[Layer], input_size: int):
    for l in layers:
        l.init_weights(input_size)
        input_size = len(l.units)


def forward(layers: list[Layer], X: np.ndarray) -> np.ndarray:
    for layer in layers:
        out = layer.output(X)
        X = out

    return out


def backward(layers: list[Layer], error: np.ndarray, X: np.ndarray):
    for i in range(len(layers)):
        layers[-(i - 1)].update_weights(X, error)


if __name__ == "__main__":
    from sklearn.datasets import make_classification

    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=100,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
    ]

    layers = [Layer(1) for _ in range(1)]
    connect_layers(layers, X.shape[1])
    out = forward(layers, X[0:1, :])

    error = out[0] - y[0:1]
    print(error)

    backward(layers, error, X[0])

    # cls0 = X[y == 0].T
    # cls1 = X[y == 1].T
    #
    # plt.figure(figsize=(5, 4), dpi=150)
    # plt.title("Example Dataset")
    #
    # plt.scatter(cls0[0], cls0[1], c="r", ec="k", label="class 0")
    # plt.scatter(cls1[0], cls1[1], c="b", ec="k", label="class 1")
    #
    # plt.xlabel("Feature 1")
    # plt.ylabel("Feature 2")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
