import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.metrics import mean_euclidean_error
from neural.network import Regressor
from neural.utils import dump_results, load_results, target_plot
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

    # stratified split
    y_scalar = np.linalg.norm(y, axis=1)
    bins = np.percentile(y_scalar, np.linspace(0, 100, 11))
    y_bins = np.digitize(y_scalar, bins[1:-1])
    X_train, X_test, y_train, y_test = [
        np.asarray(i)
        for i in train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=0,
            stratify=y_bins,
        )
    ]

    # if --gs argument is passed the grid search is performed
    if args.gs:
        hyperparams = {
            "hidden_layer_sizes": [(64, 64, 64)],
            "activation": ["relu", "leaky_relu"],
            "learning_rate": [0.001, 0.005, 0.01],
            "lam": [0.0, 0.0001],
            "alpha": [0.5, 0.9],
            "shuffle": [False, True],
            "batch_size": [16, 64],
            "convergence": ["train_loss", "early_stopping"],
            "tol": [0.0, 1e-5],
            "patience": [100],
            "max_iter": [2000],
        }

        results = grid_search(
            model=Regressor,
            hyperparams=hyperparams,
            X=X_train,
            y=y_train,
            k=10,
            metric=mean_euclidean_error,
            scale=True,
            address=args.dask,
            verbose=True,
        )

        # save results to a JSON file
        if args.save:
            dump_results("results/cup.json", results)
    else:
        results = load_results("results/cup.json")

    results = [r for r in results if r["loss"] != np.inf]
    best = sorted(results, key=lambda x: x["score"])[0]
    print(f"grid search score: {best['score']:.2f}")
    print(f"grid search std score: {best['std']:.2f}")
    print(f"grid search loss: {best['loss']:.2f}")

    # re-train the model
    params = best["parameters"]
    print(json.dumps(params, indent=2))

    net = Regressor(**params)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    net.fit(X_train, y_train, mean_euclidean_error, X_test, y_test)
    y_pred_train = net.predict(X_train)
    y_pred_test = net.predict(X_test)

    y_train = y_scaler.inverse_transform(y_train)
    y_test = y_scaler.inverse_transform(y_test)
    y_pred_train = y_scaler.inverse_transform(y_pred_train)
    y_pred_test = y_scaler.inverse_transform(y_pred_test)

    print(f"train MSE: {mean_squared_error(y_train, y_pred_train):.3f}")
    print(f"train MEE: {mean_euclidean_error(y_train, y_pred_train):.3f}")

    print(f"test MSE: {mean_squared_error(y_test, y_pred_test):.3f}")
    print(f"test MEE: {mean_euclidean_error(y_test, y_pred_test):.3f}")

    plt.figure(figsize=(8, 6))
    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.title("MEE Curve")
    plt.plot(net.score_curve, label="training")
    plt.plot(net.val_score_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    target_plot(y_test, y_pred_test)
