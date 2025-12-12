import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor
from neural.utils import save_stats, stats
from neural.validation import grid_search


def mean_euclidean_error(y_true, y_pred):
    return np.mean(np.linalg.norm(y_true - y_pred, axis=1))


def neg_mean_euclidean_error(y_true, y_pred):
    return -np.mean(np.linalg.norm(y_true - y_pred, axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", action="store_true", help="perform a grid search")
    args = parser.parse_args()

    # set headers
    names = ["ID"]
    features = [f"feature{i}" for i in range(12)]
    targets = [f"target{i}" for i in range(4)]

    names.extend(features)
    names.extend(targets)

    train = pd.read_csv(
        "datasets/ml_cup_train.csv",
        header=None,
        names=names,
        skiprows=7,
    )

    # get feature and target columns
    X = train.iloc[:, 1:13].to_numpy()
    y = train.iloc[:, 13:].to_numpy()

    # shuffle the entire dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # keep 10% for test
    test_size = int(np.round(X.shape[0] * 0.2))
    threshold = X.shape[0] - test_size
    X_train = X[:threshold]
    y_train = y[:threshold]
    X_test = X[threshold:]
    y_test = y[threshold:]

    hyperparams = {
        "hidden_layer_sizes": [(64,)],
        "activation": ["tanh", "relu"],
        "learning_rate": [0.001, 0.003, 0.01, 0.03],
        "lam": [0.0, 0.0001],
        "alpha": [0.0, 0.7, 0.9],
        "tol": [1e-5],
        "batch_size": [64],
        "shuffle": [False],
        "max_iter": [2000],
    }

    # if --gs argument is passed the grid search is performed
    if args.gs:
        net, score = grid_search(
            model_type=Regressor,
            hyperparams=hyperparams,
            X=X_train,
            y=y_train,
            k=5,
            score_metric=neg_mean_euclidean_error,
            scale=True,
            verbose=True,
        )

    # normalize train and test set
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    if not args.gs:
        with open(f"results/cup.json", "r") as fp:
            data = json.load(fp)
            params = data["parameters"]

            net = Regressor(**params)
            net.fit(X_train, y_train, X_test, y_test)

    # training accuracy
    y_pred = net.predict(X_train)
    y_train = y_scaler.inverse_transform(y_train)
    y_pred = y_scaler.inverse_transform(y_pred)
    train_score = mean_euclidean_error(y_train, y_pred)

    # test accuracy
    y_pred = net.predict(X_test)
    y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    test_score = mean_euclidean_error(y_test, y_pred)

    # if --gs passed save results of grid search to a file
    stats(net, hyperparams, train_score, test_score)
    if args.gs:
        save_stats(net, hyperparams, score, "results/cup.json")

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
