import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

from neural.neuron import Neuron

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

    neuron = Neuron(activation="logistic", learning_rate=0.1)
    neuron.init_weights(X.shape[1])

    loss_curve = []
    for epoch in range(50):
        epoch_loss = 0.0
        for i in range(len(y)):
            out = neuron(X[i])
            delta = neuron.update(2 * (out - y[i]))
            epoch_loss += (out - y[i]) ** 2
        loss_curve.append(epoch_loss / y.size)

    out = np.array([np.round(neuron(x)) for x in X])
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")
