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

    def forward(self, X: np.ndarray) -> np.ndarray:
        out = np.zeros((X.shape[0], len(self.units)))
        for i, u in enumerate(self.units):
            out[:, i] = u.predict(X)

        return out


def forward(X: np.ndarray, layers: list[Layer]) -> np.ndarray:
    out = X
    for layer in layers:
        out = layer.forward(out)

    return out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
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

    layer = Layer(1)
    output = Layer(1)
    net = [layer, output]

    n_inputs = X.shape[1]
    for l in net:
        l.init_weights(n_inputs)
        n_inputs = len(l.units)

    out = forward(X, net)

    cls0 = X[y == 0].T
    cls1 = X[y == 1].T

    plt.figure(figsize=(5, 4), dpi=150)
    plt.title("Example Dataset")

    plt.scatter(cls0[0], cls0[1], c="r", ec="k", label="class 0")
    plt.scatter(cls1[0], cls1[1], c="b", ec="k", label="class 1")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
