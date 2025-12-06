import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neural.network import Classifier
from neural.validation import grid_search


def stats(net, score, hyperparams, X_train, X_test, y_train, y_test):
    print(f"validation score: {score:.4f}")

    for k in hyperparams.keys():
        print(f"{k}: {net.__dict__[k]}")

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.4f}")

    # training accuracy
    net_pred = net.predict(X_train)
    accuracy = accuracy_score(y_train, net_pred)
    print(f"train accuracy: {accuracy:.2f}")

    # test accuracy
    net_pred = net.predict(X_test)
    accuracy = accuracy_score(y_test, net_pred)
    print(f"test accuracy: {accuracy:.2f}")


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
        "hidden_layer_sizes": [(3,), (5,)],
        "activation": ["logistic", "tanh", "relu"],
        "learning_rate": [0.001, 0.003, 0.01, 0.03],
        "lam": [0.0, 0.0001],
        "alpha": [0.0, 0.7],
        "tol": [1e-5],
        "batch_size": [8, 16, 32],
        "shuffle": [False, True],
        "max_iter": [2000],
    }

    net, score = grid_search(
        model_type=Classifier,
        hyperparams=hyperparams,
        X=X_train,
        y=y_train,
        k=10,
        score_metric=accuracy_score,
        retrain=False,
    )
    stats(net, score, hyperparams, X_train, X_test, y_train, y_test)

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
