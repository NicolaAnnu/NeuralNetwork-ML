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
        "hidden_layer_sizes": [(5,), (10,)],
        "activation": ["tanh", "relu"],
        "learning_rate": [0.01, 0.03],
        "lam": [0.0, 0.0001],
        "alpha": [0.0, 0.5, 0.9],
        "batch_size": [16, 32],
        "shuffle": [False, True],
        "max_iter": [1000],
    }

    net, score = grid_search(
        model_type=Classifier,
        hyperparams=hyperparams,
        X=X_train,
        y=y_train,
        k=5,
        score_metric=accuracy_score,
        retrain=False,
    )
    for k in hyperparams.keys():
        print(f"{k}: {net.__dict__[k]}")

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
