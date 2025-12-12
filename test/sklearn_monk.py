import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

from neural.utils import save_stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    parser.add_argument("--gs", action="store_true", help="perform a grid search")
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
        "hidden_layer_sizes": [(3,)],
        "activation": ["tanh"],
        "learning_rate_init": [0.001, 0.003, 0.01],
        "alpha": [0.0, 0.0001],
        "momentum": [0.0, 0.7],
        "batch_size": [8, 16],
        "shuffle": [False, True],
    }

    # if --gs argument is passed the grid search is performed
    if args.gs:
        net = MLPClassifier(solver="sgd", tol=1e-5, max_iter=2000)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(
            net,
            hyperparams,
            scoring="accuracy",
            n_jobs=-1,
            cv=kf,
            verbose=2,
        )

        grid.fit(X_train, y_train)

        net = grid.best_estimator_
        score = grid.best_score_

    else:  # otherwise reads parameters from file and perform a single training
        with open(f"results/sklearn_monk{args.id}.json", "r") as fp:
            data = json.load(fp)
            params = data["parameters"]

        net = MLPClassifier(**params)
        net.fit(X_train, y_train)

    # training accuracy
    net_pred = net.predict(X_train)
    train_score = accuracy_score(y_train, net_pred)
    print(f"train score: {train_score:.2f}")

    # test accuracy
    net_pred = net.predict(X_test)
    test_score = accuracy_score(y_test, net_pred)
    print(f"test score: {test_score:.2f}")

    # if --gs passed save results of grid search to a file
    if args.gs:
        save_stats(net, hyperparams, score, f"results/sklearn_monk{args.id}.json")

    plt.title("Loss Curve")
    plt.plot(net.loss_curve_, label="training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
