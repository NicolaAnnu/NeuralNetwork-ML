import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from neural.network import Classifier
from neural.utils import stats
from neural.validation import grid_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    args = parser.parse_args()

    train = pd.read_csv(f"datasets/monks_train{args.id}.csv")
    X_train = train[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_train = train["class"].to_numpy()

    test = pd.read_csv(f"datasets/monks_test{args.id}.csv")
    X_test = test[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_test = test["class"].to_numpy()

    encoder = OneHotEncoder(sparse_output=False)
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

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
        "batch_size": [8, 16, 32],
        "shuffle": [False, True],
        "max_iter": [1000, 2000],
    }

    net, score = grid_search(
        model_type=Classifier,
        hyperparams=hyperparams,
        X=X_train,
        y=y_train,
        k=5,
        score_metric=accuracy_score,
    )
    # training accuracy
    net_pred = net.predict(X_train)
    train_score = accuracy_score(y_train, net_pred)
    print(f"train accuracy: {train_score:.2f}")

    # test accuracy
    net_pred = net.predict(X_test)
    test_score = accuracy_score(y_test, net_pred)
    print(f"test accuracy: {test_score:.2f}")

    # print stats and save results to json file
    stats(
        net,
        hyperparams,
        score,
        train_score,
        test_score,
        f"results/monk{args.id}.json",
    )

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()