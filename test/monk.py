import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

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

    hyperparams = {
        "hidden_layer_sizes": [(3,), (5,), (8,)],
        "activation": ["logistic", "tanh", "relu"],
        "learning_rate": [0.01, 0.03, 0.1],
        "lam": [0.00005, 0.0001, 0.0002],
        "alpha": [0.7, 0.9],
        "batch_size": [8, 16, 32],
        "shuffle": [False, True],
        "max_iter": [500, 1000],
    }

    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    net = grid_search(
        Classifier, hyperparams, X_train[indices], y_train[indices], 0.1, accuracy_score
    )

    for k in hyperparams.keys():
        print(f"{k}: {net.__dict__[k]}")

    net.fit(X_train, y_train)
    print(f"network loss: {net.loss:.2f}")

    plt.plot(net.loss_curve, label="network")
    plt.legend()
    plt.show()

    # Network
    net_pred = net.predict(X_train)
    accuracy = accuracy_score(y_train, net_pred)
    print(f"network train accuracy: {accuracy:.2f}")

    # Test set
    net_pred = net.predict(X_test)
    accuracy = accuracy_score(y_test, net_pred)
    print(f"network test accuracy: {accuracy:.2f}")
