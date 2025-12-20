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
from neural.validation import grid_search

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
            "hidden_layer_sizes": [(3,)],
            "activation": ["logistic", "tanh"],
            "learning_rate": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3],
            "lam": [0.0, 0.00005, 0.0001],
            "alpha": [0.0, 0.5, 0.7, 0.9],
            "tol": [1e-6],
            "batch_size": [8, 16, 32, 64],
            "shuffle": [False, True],
            "max_iter": [2000],
        }

        results = grid_search(
            model_type=Classifier,
            hyperparams=hyperparams,
            X=X_train,
            y=y_train,
            k=10,
            metric="accuracy",
            scale=False,
            address="tcp://192.168.1.95:8786",
            verbose=True,
        )

        # save results to a json file
        if args.save:
            with open(f"results/monk{args.id}.json", "w") as fp:
                json.dump(results, fp, indent=2)

    else:  # get params from last saved grid search
        with open(f"results/monk{args.id}.json", "r") as fp:
            results = json.load(fp)

    best = results[0]
    print(json.dumps(best["parameters"], indent=2))
    print(f"best grid search score: {best['score']:.2f}")

    params = best["parameters"]
    net = Classifier(**params)
    net.fit(X_train, y_train, X_test, y_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.3f}")

    # training accuracy
    net_pred = net.predict(X_train)
    train_accuracy = accuracy_score(y_train, net_pred)
    print(f"train accuracy: {train_accuracy:.2f}")

    # train f1
    train_f1 = f1_score(y_train, net_pred)
    print(f"train f1: {train_f1:.2f}")

    # train confusion matrix
    train_cm = confusion_matrix(y_train, net_pred)
    ConfusionMatrixDisplay(train_cm).plot()
    plt.show()

    # test accuracy
    net_pred = net.predict(X_test)
    test_accuracy = accuracy_score(y_test, net_pred)
    print(f"test accuracy: {test_accuracy:.2f}")

    # test f1
    test_f1 = f1_score(y_test, net_pred)
    print(f"test f1: {test_f1:.2f}")

    # test confusion matrix
    test_cm = confusion_matrix(y_test, net_pred)
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
