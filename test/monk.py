import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from itertools import product
from neural.network import Classifier


def grid_search_params(hyperparameters):
    keys = list(hyperparameters.keys())
    values = list(hyperparameters.values())

    for instance in product(*values):
        params = dict(zip(keys, instance))
        yield params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    args = parser.parse_args()

    train = pd.read_csv(f"datasets/monks_train{args.id}.csv")
    X_train = train[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_train = train["class"].to_numpy()

    test = pd.read_csv(f"datasets/monks_test{args.id}.csv")
    X_test = test[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_test = test["class"].to_numpy()

    encoder = OneHotEncoder(sparse_output=False)
    X_train = encoder.fit_transform(X_train)
    X_test = np.asarray(encoder.transform(X_test))

    hyperparameters = {
        "learning_rate": [0.3, 0.1, 0.01],
        "activation": ["tanh", "logistic"],   
        "lam": [0, 0.0001, 0.001],
        "alpha": [0.0, 0.5, 0.9],
        "batch_size": [8, 16, 32],
        "max_iter": [1000],
        "tol": [1e-4, 1e-3, 1e-2],
        "topology": [(3,), (5,)],
        "shuffle": [True, False],
    }

    best_params = None
    best_accuracy = -np.inf
    best_net = None

    for params in grid_search_params(hyperparameters):
        print("Provo configurazione:", params)

        # Crea la rete con questi iperparametri
        net = Classifier(
            hidden_layer_sizes=params["topology"],
            activation=params["activation"],
            learning_rate=params["learning_rate"],
            lam=params["lam"],
            alpha=params["alpha"],
            batch_size=params["batch_size"],
            shuffle=params["shuffle"],
            max_iter=params["max_iter"],
        )

        # Allena
        net.fit(X_train, y_train)

        y_pred_train = net.predict(X_train)
        train_acc = np.mean(y_pred_train == y_train)
        print(f"Train accuracy: {train_acc:.3f}")

        # Valutazione su test
        y_pred_test = net.predict(X_test)
        test_acc = np.mean(y_pred_test == y_test)
        print(f"Test accuracy:  {test_acc:.3f}")

        # Salvo il migliore in base alla test accuracy
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_params = params
            best_net = net

    print("Migliori iperparametri:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print(f"Migliore test accuracy: {best_accuracy:.3f}")

    # Se vuoi, puoi anche plottare la loss dell'ultimo/best modello
    if hasattr(best_net, "loss_curve"):
        plt.plot(best_net.loss_curve, label="network loss")
        plt.legend()
        plt.show()
