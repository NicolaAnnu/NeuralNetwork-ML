import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neural.metrics import mean_euclidean_error
from neural.network import Regressor
from neural.validation import grid_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gs", action="store_true", help="perform a grid search")
    parser.add_argument("--save", action="store_true", help="save results in a file")
    parser.add_argument(
        "--dask", type=str, default=None, help="perform a distributed grid search"
    )
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

    # if --gs argument is passed the grid search is performed
    if args.gs:
        hyperparams = {
            "hidden_layer_sizes": [(32, 32), (64, 64)],
            "activation": ["relu"],
            "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.05, 0.1],
            "lam": [0.0, 0.00005, 0.0001],
            "alpha": [0.0, 0.7, 0.9],
            "tol": [1e-5],
            "batch_size": [32, 64],
            "shuffle": [False, True],
            "max_iter": [3000],
        }

        results = grid_search(
            model_type=Regressor,
            hyperparams=hyperparams,
            X=X_train,
            y=y_train,
            k=10,
            metric="mee",
            scale=True,
            address=args.dask,
            verbose=True,
        )

        # save results to a json file
        if args.save:
            with open("results/cup.json", "w") as fp:
                json.dump(results, fp, indent=2)
    else:
        with open("results/cup.json", "r") as fp:
            results = json.load(fp)

    best = results[0]
    print(json.dumps(best["parameters"], indent=2))
    print(f"best grid search score: {best['score']:.2f}")

    params = best["parameters"]
    net = Regressor(**params)

    # normalize train and test set
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    # train the model
    net.fit(X_train, y_train, X_test, y_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.3f}")

    # training accuracy
    y_pred = net.predict(X_train)
    # y_train = y_scaler.inverse_transform(y_train)
    # y_pred = y_scaler.inverse_transform(y_pred)
    train_score = mean_euclidean_error(y_train, y_pred)
    print(f"train MEE: {train_score:.3f}")

    # test accuracy
    y_pred = net.predict(X_test)
    # y_test = y_scaler.inverse_transform(y_test)
    # y_pred = y_scaler.inverse_transform(y_pred)
    test_score = mean_euclidean_error(y_test, y_pred)
    print(f"test MEE: {test_score:.3f}")

    mtn = np.mean(np.linalg.norm(y_test, axis=1))
    print(f"mean target norm: {mtn:.2f}")
    print(f"percentage error: {test_score / mtn:.2f}")

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
