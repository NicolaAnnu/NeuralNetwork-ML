import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from neural.network import Classifier

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

    net = Classifier(
        hidden_layer_sizes=(5,),
        activation="tanh",
        learning_rate=0.1,
        batch_size=100,
        max_iter=500,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(5,),
        activation="tanh",
        solver="sgd",
        learning_rate_init=0.1,
        max_iter=500,
    )

    net.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    plt.plot(net.loss_curve)
    plt.plot(mlp.loss_curve_)
    plt.show()

    net_pred = np.round(net.predict(X_train))
    accuracy = accuracy_score(y_train, net_pred)
    print(f"network train accuracy: {accuracy:.2f}")
    print(f"network loss: {net.loss:.2f}")

    net_pred = np.round(net.predict(X_test))
    accuracy = accuracy_score(y_test, net_pred)
    print(f"network test accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_train)
    accuracy = accuracy_score(y_train, mlp_pred)
    print(f"sklearn train accuracy: {accuracy:.2f}")
    print(f"sklearn loss: {mlp.loss_:.2f}")

    mlp_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, mlp_pred)
    print(f"sklearn test accuracy: {accuracy:.2f}")
