import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.network import Classifier
from neural.validation import grid_search

if __name__ == "__main__":
    X, y = [np.array(i) for i in load_breast_cancer(return_X_y=True)]
    normalizer = StandardScaler()
    X = normalizer.fit_transform(X)
    X_train, X_test, y_train, y_test = [
        np.array(i) for i in train_test_split(X, y, test_size=0.2)
    ]

    hyperparams = {
        "hidden_layer_sizes": [(5,), (8,)],
        "activation": ["logistic", "tanh"],
        "learning_rate": [0.01, 0.03, 0.1],
        "lam": [0.0001, 0.0002],
        "alpha": [0.7, 0.9],
        "batch_size": [16, 32, 64],
        "shuffle": [False, True],
        "max_iter": [500, 1000],
    }

    net = grid_search(Classifier, hyperparams, X_train, y_train, 0.2, accuracy_score)

    print(net.__dict__)

    print(f"network loss: {net.loss:.2f}")

    plt.plot(net.loss_curve, label="network")
    plt.legend()
    plt.show()

    net_pred = net.predict(X_train)
    accuracy = np.mean(net_pred == y_train)
    print(f"network train accuracy: {accuracy:.2f}")

    net_pred = net.predict(X_test)
    accuracy = np.mean(net_pred == y_test)
    print(f"network test accuracy: {accuracy:.2f}")
