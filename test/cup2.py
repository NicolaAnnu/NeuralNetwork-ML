import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
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
            "activation": ["leaky_relu"],
            "learning_rate": [0.001, 0.005, 0.01],
            "lam": [0.0001],
            "alpha": [0.9],
            "shuffle": [True],
            "batch_size": [16, 64, 256],
            "convergence": ["train_loss"],
            "patience": [50],
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

    results = [r for r in results if r["loss"] != np.inf]
    best = sorted(results, key=lambda x: x["score"])[0]
    print(f"grid search score: {best['score']:.2f}")
    print(f"grid search std score: {best['std']:.2f}")
    print(f"grid search loss: {best['loss']:.2f}")

    loss_curves = []
    val_loss_curves = []
    score_curves = []
    val_score_curves = []
    losses = []
    val_losses = []

    train_mses = []
    train_mees = []
    train_r2s = []

    test_mses = []
    test_mees = []
    test_r2s = []

    # re-train the model
    params = best["parameters"]
    if params["convergence"] == "early_stopping":
        params["limit"] = best["loss"]

    print(json.dumps(params, indent=2))

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    nets = retrain(
        Regressor,
        params,
        X_train,
        y_train,
        X_test,
        y_test,
        mean_euclidean_error,
        5,
        args.dask,
    )

    for net in nets:
        loss_curves.append(net.loss_curve.copy())
        val_loss_curves.append(net.val_loss_curve.copy())
        score_curves.append(net.score_curve.copy())
        val_score_curves.append(net.val_score_curve.copy())
        losses.append(net.loss)
        val_losses.append(net.val_loss)

        # training
        y_pred = net.predict(X_train)
        y_train_raw = y_scaler.inverse_transform(y_train)
        y_pred = y_scaler.inverse_transform(y_pred)
        train_mses.append(np.mean((y_train_raw - y_pred) ** 2))
        train_mees.append(mean_euclidean_error(y_train_raw, y_pred))
        train_r2s.append(r2(y_train_raw, y_pred))

        # test
        y_pred = net.predict(X_test)
        y_test_raw = y_scaler.inverse_transform(y_test)
        y_pred = y_scaler.inverse_transform(y_pred)

        test_mses.append(np.mean((y_test_raw - y_pred) ** 2))
        test_mees.append(mean_euclidean_error(y_test_raw, y_pred))
        test_r2s.append(r2(y_test_raw, y_pred))

    epochs = [len(lc) for lc in loss_curves]
    print(f"mean convergence in {np.mean(epochs, dtype=int)} epochs")
    print(f"mean loss: {np.mean(losses):.3f}")
    print(f"mean validation loss: {np.mean(val_losses):.3f}")

    print(f"mean train MSE: {np.mean(train_mses):.3f}")
    print(f"mean train MEE: {np.mean(train_mees):.3f}")
    print(f"mean train R2: {np.mean(train_r2s):.3f}")

    print(f"mean test MSE: {np.mean(test_mses):.3f}")
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
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("MEE Curve")
    plot_curve(score_curves, label="training")
    plot_curve(val_score_curves, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    target_plot(y_test_raw, y_pred)
