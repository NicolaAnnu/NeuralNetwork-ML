import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from neural.layer import Layer


def forward(layers: list[Layer], X: np.ndarray) -> np.ndarray:
    for l in layers[:-1]:
        X = l.forward(X)

    return layers[-1](X)


def backward(layers: list[Layer], delta: np.ndarray) -> None:
    for l in reversed(layers):
        delta = l.backward(delta)


if __name__ == "__main__":
    n_features = 2
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=200,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            class_sep=2,
            random_state=0,
        )
    ]

    lr = 0.05
    hidden = Layer(n_units=8, activation="logistic", learning_rate=lr)
    output = Layer(n_units=1, activation="logistic", learning_rate=lr)

    hidden.init_weights(X.shape[1])
    output.init_weights(len(hidden.units))
    layers = [hidden, output]

    batch_size = 10
    for epoch in range(200):
        for i in range(len(y)):
            out = forward(layers, X[i : i + batch_size, :])
            errors = out - y[i : i + batch_size]
            delta = 2 * errors / batch_size
            backward(layers, delta)

    out = np.round(forward(layers, X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")

    import matplotlib.pyplot as plt

    plt.scatter(X.T[0], X.T[1], c=y)
    plt.show()
