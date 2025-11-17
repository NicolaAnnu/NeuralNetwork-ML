import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from network.neuron import Neuron


def forward(neurons: list[Neuron], X: np.ndarray) -> np.ndarray:
    for n in neurons[:-1]:
        X = n.forward(X)

    return neurons[-1](X)


def backward(neurons: list[Neuron], error: np.ndarray) -> None:
    delta = 2 * error / error.size
    for i, n in enumerate(reversed(neurons)):
        delta = n.update_weights(delta)

        if i > 0:
            delta = delta[:, None] * n.W[None, :]
            delta = delta[:, 0]


if __name__ == "__main__":
    n_features = 2
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=100,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
    ]

    lr = 0.3
    chain = [Neuron(activation="logistic", learning_rate=lr) for _ in range(4)]

    input_size = X.shape[1]
    for n in chain:
        n.init_weights(input_size)
        input_size = 1

    batch_size = 5
    for epoch in range(200):
        for i in range(len(y)):
            out = forward(chain, X[i : i + batch_size, :])
            error = out - y[i : i + batch_size]
            backward(chain, error)

    out = np.round(forward(chain, X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")
