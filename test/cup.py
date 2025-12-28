import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
            "hidden_layer_sizes": [(64, 32), (64, 64), (128, 64)],
            "activation": ["relu", "leaky_relu"],
            "learning_rate": [0.01, 0.03, 0.05, 0.07],
            "lam": np.concatenate(([0.0], np.logspace(-6, -4, 3))).tolist(),
            "alpha": [0.0, 0.7, 0.9],
            "tol": [1e-5],
            "batch_size": [16, 32, 64, 128],
            "shuffle": [False, True],
            "early_stopping": [False, True],
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

    best = sorted(results, key=lambda x: x["score"] + x["std"])[0]
    print(json.dumps(best["parameters"], indent=2))
    print(f"grid search score: {best['score']:.2f}")
    print(f"grid search std score: {best['std']:.2f}")

    # normalize train and test set
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    # re-train the model
    loss_limit = -np.inf
    params = best["parameters"]
    if params["early_stopping"]:
        params["early_stopping"] = False  # always disable it for retraining
        loss_limit = best["loss"]

    net = Regressor(**params)
    net.fit(X_train, y_train, loss_limit=loss_limit, X_val=X_test, y_val=y_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.3f}")

    # training
    y_pred = net.predict(X_train)
    y_train = y_scaler.inverse_transform(y_train)
    y_pred = y_scaler.inverse_transform(y_pred)
    train_score = mean_euclidean_error(y_train, y_pred)
    print(f"train MEE: {train_score:.3f}")

    mse_per_output = np.mean((y_train - y_pred) ** 2, axis=0)
    var_per_output = np.var(y_train, axis=0)
    r2_per_output = 1 - mse_per_output / var_per_output
    print(f"train R2: {np.mean(r2_per_output):.3f}")

    # test
    y_pred = net.predict(X_test)
    y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    test_score = mean_euclidean_error(y_test, y_pred)
    print(f"test MEE: {test_score:.3f}")

    rmse_per_output = np.sqrt(np.mean((y_test - y_pred) ** 2, axis=0))
    range_per_output = np.max(y_test, axis=0) - np.min(y_test, axis=0)
    nrmse = np.mean(rmse_per_output / range_per_output)
    print(f"mean NRMSE: {nrmse:.3f}")

    std_per_output = np.std(y_test, axis=0)
    print(f"std per output: {std_per_output}")
    nrmse_std = np.mean(rmse_per_output / std_per_output)
    print(f"mean NRMSE (std): {nrmse_std:.3f}")

    mse_per_output = np.mean((y_test - y_pred) ** 2, axis=0)
    var_per_output = np.var(y_test, axis=0)
    r2_per_output = 1 - mse_per_output / var_per_output
    print(f"test R2: {np.mean(r2_per_output):.3f}")

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("MEE Curve")
    plt.plot(net.err_curve, label="training")
    plt.plot(net.val_err_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.legend()
    plt.tight_layout()
    plt.show()

    target_plot(y_test, y_pred)
