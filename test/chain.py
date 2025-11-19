import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from neural.neuron import Neuron


def forward(neurons: list[Neuron], X: np.ndarray) -> float:
    for n in neurons:
        X = n.forward(X)

    return X[0]


def backward(neurons: list[Neuron], error: float) -> None:
    for n in reversed(neurons):
        old_W = n.W.copy()  # save the old weights
        delta = n.backward(np.array([error]))
        error = np.sum(delta * old_W)


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

    chain = [Neuron(activation="logistic", learning_rate=0.3) for _ in range(3)]
    chain[0].init_weights(X.shape[1])
    for u in chain[1:]:
        u.init_weights(1)

    loss_curve = []
    for epoch in range(1000):
        epoch_loss = 0.0
        for i in range(len(y)):
            out = forward(chain, X[i])
            backward(chain, 2 * (out - y[i]))
            epoch_loss += (out - y[i]) ** 2
        loss_curve.append(epoch_loss / y.size)

    out = np.array([np.round(forward(chain, x)) for x in X])
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")

    plt.plot(loss_curve)
    plt.show()
