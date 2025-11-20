import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from neural.neuron import Neuron


def forward(neurons: list[Neuron], X: np.ndarray) -> np.ndarray:
    for n in neurons:
        X = n.forward(X)

    return X[:, 0]


def backward(neurons: list[Neuron], error: np.ndarray) -> None:
    for n in reversed(neurons):
        old_W = n.W.copy()  # save the old weights
        delta = n.backward(error)
        error = np.outer(delta, old_W)[:, 0]


if __name__ == "__main__":
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=200,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=2,
            random_state=0,
        )
    ]

    lr = 0.3
    chain = [Neuron(activation="relu", learning_rate=lr) for _ in range(3)]
    chain.append(Neuron(activation="logistic", learning_rate=lr))
    chain[0].init_weights(X.shape[1])
    for u in chain[1:]:
        u.init_weights(1)

    out = np.round(forward(chain, X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")

    loss_curve = []
    batch_size = 100
    for epoch in range(500):
        epoch_loss = 0.0
        for i in range(0, len(y), batch_size):
            out = forward(chain, X[i : i + batch_size, :])
            error = out - y[i : i + batch_size]
            backward(chain, 2 * error)
            epoch_loss += np.sum(error**2) / batch_size
        loss_curve.append(epoch_loss / (len(y) / batch_size))

    out = np.round(forward(chain, X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")

    plt.plot(loss_curve)
    plt.show()
