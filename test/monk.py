import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neural.network import Classifier
from neural.validation import grid_search

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

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    hyperparams = {
        "hidden_layer_sizes": [(3,), (5,), (10,)],
        "activation": ["logistic", "tanh"],
        "learning_rate": [0.01, 0.03, 0.1],
        "lam": [0.0, 0.0001],
        "alpha": [0.0, 0.5, 0.9],
        "tol": [1e-4, 1e-6],
        "batch_size": [8, 16, 32],
        "shuffle": [False, True],
        "max_iter": [1000],
    }

    net = grid_search(
        Classifier,
        hyperparams,
        X_train,
        y_train,
        0.1,
        accuracy_score,
        retrain=False,
    )
    for k in hyperparams.keys():
        print(f"{k}: {net.__dict__[k]}")

    print(f"network loss: {net.loss:.2f}")
    plt.plot(net.loss_curve, label="network")
    plt.legend()
    plt.show()

    # Network
    net_pred = net.predict(X_train)
    accuracy = accuracy_score(y_train, net_pred)
    print(f"network train accuracy: {accuracy:.2f}")

    # Test set
    net_pred = net.predict(np.asarray(X_test))
    accuracy = accuracy_score(y_test, net_pred)
    print(f"network test accuracy: {accuracy:.2f}")
