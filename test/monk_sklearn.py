import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neural.utils import stats  # riuso la tua funzione


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    args = parser.parse_args()
    train = pd.read_csv(f"datasets/monks_train{args.id}.csv")
    X_train_raw = train[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_train = train["class"].to_numpy()

    test = pd.read_csv(f"datasets/monks_test{args.id}.csv")
    X_test_raw = test[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_test = test["class"].to_numpy()
    encoder = OneHotEncoder(sparse_output=False)
    X_train = encoder.fit_transform(X_train_raw)
    X_test = encoder.transform(X_test_raw)

    scaler = StandardScaler(with_mean=False)
    X_train = scaler.fit_transform(X_train)
    X_test = np.asarray(scaler.transform(X_test))

    hyperparams = {
        "hidden_layer_sizes": [(3,)],
        "activation": ["logistic", "tanh"],
        "learning_rate": [0.01, 0.03], 
        "lam": [0.0, 0.0001],         
        "alpha": [0.9],               
        "tol": [1e-5],
        "batch_size": [8, 16],
        "shuffle": [False, True],
        "max_iter": [1000],
    }

    sk_param_grid = {
        "hidden_layer_sizes": hyperparams["hidden_layer_sizes"],
        "activation": hyperparams["activation"],
        "learning_rate_init": hyperparams["learning_rate"],  # lr
        "alpha": hyperparams["lam"],                        # L2 reg
        "momentum": hyperparams["alpha"],                   # momentum
        "tol": hyperparams["tol"],
        "batch_size": hyperparams["batch_size"],
        "shuffle": hyperparams["shuffle"],
        "max_iter": hyperparams["max_iter"],
    }

    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    base_mlp = MLPClassifier(
        solver="sgd",
        learning_rate="constant",  # il passo è learning_rate_init
        n_iter_no_change=50,
        verbose=False,
        random_state=42,
    )

    grid = GridSearchCV(
        estimator=base_mlp,
        param_grid=sk_param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    grid.fit(X_train, y_train)

    net = grid.best_estimator_
    score = grid.best_score_

    print(f"best CV score (sklearn): {score:.4f}")
    print(f"best params (sklearn): {grid.best_params_}")

    net_pred = net.predict(X_train)
    train_score = accuracy_score(y_train, net_pred)
    print(f"train accuracy: {train_score:.2f}")

    # test accuracy
    net_pred = net.predict(X_test)
    test_score = accuracy_score(y_test, net_pred)
    print(f"test accuracy: {test_score:.2f}")
    stats(
        net,
        hyperparams, 
        score,
        train_score,
        test_score,
        f"results/monk{args.id}_sklearn.json",
    )
    if hasattr(net, "loss_curve_"):
        plt.title("Loss Curve (sklearn MLP)")
        plt.plot(net.loss_curve_, label="loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.show()
