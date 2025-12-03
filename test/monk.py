import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from metrics import Metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

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

    encoder = OneHotEncoder(sparse_output=False)
    X_train = encoder.fit_transform(X_train)
    X_test = np.asarray(encoder.transform(X_test))

    topology = (3,)
    activation = "tanh"
    learning_rate = 0.3
    lam = 0.0001
    alpha = 0.9
    batch_size = 10
    max_iter = 1000

    net = Classifier(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        alpha=alpha,
        batch_size=-1,
        shuffle=True,
        max_iter=max_iter,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=topology,
        activation=activation,
        solver="sgd",
        alpha=lam,
        learning_rate_init=learning_rate,
        momentum=alpha,
        max_iter=max_iter,
    )

    net.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    print(f"network loss: {net.loss:.2f}")
    print(f"sklearn loss: {mlp.loss_:.2f}")

    plt.plot(net.loss_curve, label="network")
    plt.plot(mlp.loss_curve_, label="sklearn")
    plt.legend()
    plt.show()
    
    # Network
    net_pred = net.predict(X_train)
    accuracy = np.mean(net_pred == y_train)
    print(f"network train accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_train)
    accuracy = np.mean(mlp_pred == y_train)
    print(f"sklearn train accuracy: {accuracy:.2f}")

    # Test set
    net_pred = net.predict(X_test)
    accuracy = np.mean(net_pred == y_test)
    print(f"network test accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_test)
    accuracy = np.mean(mlp_pred == y_test)
    print(f"sklearn test accuracy: {accuracy:.2f}")

