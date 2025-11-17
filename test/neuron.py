import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from neural.neuron import Neuron

if __name__ == "__main__":
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=100,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
    ]

    neuron = Neuron(activation="logistic", learning_rate=0.1)
    neuron.init_weights(X.shape[1])

    batch_size = 1
    for epoch in range(200):
        for i in range(0, len(y), batch_size):
            out = neuron(X[i : i + batch_size, :])
            error = out - y[i : i + batch_size]
            error = 2 * error / batch_size
            delta = neuron.update_weights(error)

    out = np.round(neuron(X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")
