import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neural.metrics import mean_euclidean_error, r2
from neural.network import Regressor
from neural.utils import dump_results, load_results, plot_curve, retrain, target_plot
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

    # keep 20% for test
    test_size = int(np.round(X.shape[0] * 0.2))
    threshold = X.shape[0] - test_size
    X_train = X[:threshold]
    y_train = y[:threshold]
    X_test = X[threshold:]
    y_test = y[threshold:]

    # if --gs argument is passed the grid search is performed
    if args.gs:
        hyperparams = {
            "hidden_layer_sizes": [(64, 64, 64)],
            "activation": ["leaky_relu"],
            "learning_rate": [0.03, 0.05, 0.07],
            "lam": [0.0, 0.00001, 0.0001],
            "alpha": [0.9],
            "tol": [1e-5],
            "batch_size": [8, 16, 32],
            "shuffle": [False, True],
            "early_stopping": [False, True],
            "patience": [10, 20, 50],
            "max_iter": [3000],
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

    results = [r for r in results if r["score"] != -np.inf]
    best = sorted(results, key=lambda x: x["score"] + x["std"])[0]
    print(f"grid search score: {best['score']:.2f}")
    print(f"grid search std score: {best['std']:.2f}")

    loss_curves = []
    val_loss_curves = []
    err_curves = []
    val_err_curves = []
    losses = []
    train_mees = []
    train_r2s = []
    test_mees = []
    test_r2s = []

    # re-train the model
    loss_limit = -np.inf
    params = best["parameters"]
    if params["early_stopping"]:
        params["early_stopping"] = False  # always disable it for retraining
        loss_limit = best["loss"]

    print(json.dumps(params, indent=2))

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    nets = retrain(
        Regressor, params, X_train, y_train, loss_limit, X_test, y_test, 10, args.dask
    )

    for net in nets:
        loss_curves.append(net.loss_curve.copy())
        val_loss_curves.append(net.val_loss_curve.copy())
        err_curves.append(net.err_curve.copy())
        val_err_curves.append(net.val_err_curve.copy())
        losses.append(net.loss)

        # training
        y_pred = net.predict(X_train)
        y_train_raw = y_scaler.inverse_transform(y_train)
        y_pred = y_scaler.inverse_transform(y_pred)
        train_mees.append(mean_euclidean_error(y_train_raw, y_pred))
        train_r2s.append(r2(y_train_raw, y_pred))

        # test
        y_pred = net.predict(X_test)
        y_test_raw = y_scaler.inverse_transform(y_test)
        y_pred = y_scaler.inverse_transform(y_pred)
        test_mees.append(mean_euclidean_error(y_test_raw, y_pred))
        test_r2s.append(r2(y_test_raw, y_pred))

    epochs = [len(lc) for lc in loss_curves]
    print(f"mean convergence in {np.mean(epochs, dtype=int)} epochs")
    print(f"mean loss: {np.mean(losses):.3f}")
    print(f"mean train MEE: {np.mean(train_mees):.3f}")
    print(f"mean train R2: {np.mean(train_r2s):.3f}")

    print(f"min test MEE: {np.min(test_mees):.3f}")
    print(f"mean test MEE: {np.mean(test_mees):.3f}")
    print(f"max test MEE: {np.max(test_mees):.3f}")
    print(f"mean test R2: {np.mean(test_r2s):.3f}")

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plot_curve(loss_curves, "training")
    plot_curve(val_loss_curves, "test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("MEE Curve")
    plot_curve(err_curves, label="training")
    plot_curve(val_err_curves, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    target_plot(y_test, y_pred)
