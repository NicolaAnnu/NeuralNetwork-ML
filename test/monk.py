import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import OneHotEncoder

from neural.network import Classifier
from neural.utils import dump_results, load_results
from neural.validation import grid_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    parser.add_argument("--gs", action="store_true", help="perform a grid search")
    parser.add_argument(
        "--save", action="store_true", help="save grid search results in a file"
    )
    parser.add_argument(
        "--dask", type=str, default=None, help="perform a distributed grid search"
    )
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

    if args.gs:
        hyperparams = {
            "hidden_layer_sizes": [(3,)],
            "activation": ["tanh", "leaky_relu"],
            "learning_rate": [0.05, 0.07, 0.09],
            "lam": [0.0, 0.00001, 0.0001],
            "alpha": [0.0, 0.5, 0.7, 0.9],
            "tol": [1e-6],
            "batch_size": [8, 16, 32],
            "shuffle": [False, True],
            "early_stopping": [False, True],
            "max_iter": [2000],
        }

        results = grid_search(
            model=Classifier,
            hyperparams=hyperparams,
            X=X_train,
            y=y_train,
            k=15,
            metric=accuracy_score,
            scale=False,
            address=args.dask,
            verbose=True,
        )

        # save results to a json file
        if args.save:
            dump_results(f"results/monk{args.id}.json", results)

    else:  # get params from last saved grid search
        results = load_results(f"results/monk{args.id}.json")

    results = [r for r in results if np.isfinite(r["score"])]
    results = [r for r in results if np.isfinite(r["std"])]
    best = sorted(results, key=lambda x: x["score"], reverse=True)[0]
    print(json.dumps(best["parameters"], indent=2))
    print(f"validation score: {best['score']:.2f}")
    print(f"validation std score: {best['std']:.2f}")
    print(f"validation loss: {best['loss']:.2f}")

    # re-train the model
    loss_limit = -np.inf
    params = best["parameters"]
    if params["early_stopping"]:
        params["early_stopping"] = False  # always disable it for retraining
        loss_limit = best["loss"]

    net = Classifier(**params)
    net.fit(X_train, y_train, loss_limit=loss_limit, X_val=X_test, y_val=y_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.3f}")

    # training accuracy
    y_pred = net.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    print(f"train accuracy: {train_accuracy:.2f}")

    # train f1
    train_f1 = f1_score(y_train, y_pred)
    print(f"train f1: {train_f1:.2f}")

    # train confusion matrix
    train_cm = confusion_matrix(y_train, y_pred)
    ConfusionMatrixDisplay(train_cm).plot()
    plt.show()

    # test accuracy
    y_pred = net.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"test accuracy: {test_accuracy:.2f}")

    # test f1
    test_f1 = f1_score(y_test, y_pred)
    print(f"test f1: {test_f1:.2f}")

    # test confusion matrix
    test_cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(test_cm).plot()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test/validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
