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


def accuracy_one_hot(y_true, y_pred):
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    return accuracy_score(y_true, y_pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    parser.add_argument("--gs", action="store_true", help="perform a grid search")
    parser.add_argument(
        "--save", action="store_true", help="save grid search results in a file"
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
            "hidden_layer_sizes": [(3,), (4,)],
            "activation": ["leaky_relu", "tanh"],
            "learning_rate": [0.01, 0.03, 0.05],
            "lam": [0.0],
            "alpha": [0.0],
            "shuffle": [False, True],
            "batch_size": [8, 16, 32],
            "convergence": ["train_loss", "early_stopping"],
            "patience": [10, 30],
            "max_iter": [800],
        }

        results = grid_search(
            model=Classifier,
            hyperparams=hyperparams,
            X=X_train,
            y=y_train,
            k=5,
            metric=accuracy_one_hot,
            scale=False,
            verbose=True,
        )

        if args.save:
            dump_results(f"results/monk{args.id}.json", results)

    else:
        results = load_results(f"results/monk{args.id}.json")

        if isinstance(results, dict):
            results = [results]

        if not isinstance(results, list):
            raise TypeError(f"Expected list of dicts, got {type(results)}")

        print(type(results))
        print(results[:2])

    results = [r for r in results if np.isfinite(r["score"])]
    results = [r for r in results if np.isfinite(r["std"])]

    if not results:
        raise RuntimeError("All grid search runs failed. Check the training errors above.")

    best = sorted(results, key=lambda x: (x["score"], -x["std"]), reverse=True)[0]

    print(f"validation score: {best['score']:.2f}")
    print(f"validation std score: {best['std']:.2f}")
    print(f"validation loss: {best['loss']:.2f}")

    params = best["parameters"]
    print(json.dumps(params, indent=2))

    net = Classifier(**params)
    net.fit(X_train, y_train, accuracy_one_hot, X_val=X_test, y_val=y_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"training loss: {net.loss:.4f}")
    print(f"test loss: {net.val_loss:.4f}")

    y_pred = net.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    print(f"train accuracy: {train_accuracy:.2f}")

    train_f1 = f1_score(y_train, y_pred)
    print(f"train f1: {train_f1:.2f}")

    train_cm = confusion_matrix(y_train, y_pred)
    ConfusionMatrixDisplay(train_cm).plot()
    plt.show()

    y_pred = net.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"test accuracy: {test_accuracy:.2f}")

    test_f1 = f1_score(y_test, y_pred)
    print(f"test f1: {test_f1:.2f}")

    test_cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(test_cm).plot()
    plt.show()

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
    plt.title("Accuracy Curve")
    plt.plot(net.score_curve, label="training")
    plt.plot(net.val_score_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()