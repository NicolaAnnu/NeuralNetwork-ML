import matplotlib.pyplot as plt
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
    batch_size = 100
    for epoch in range(200):
        epoch_loss = 0.0
        for i in range(0, len(y), batch_size):
            out = neuron(X[i : i + batch_size, :])
            error = out - y[i : i + batch_size]
            neuron.backward(2 * error)
            epoch_loss += np.sum(error**2) / batch_size
        loss_curve.append(epoch_loss / (len(y) / batch_size))

    plt.plot(loss_curve)
    plt.show()

    out = np.round(neuron(X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")
