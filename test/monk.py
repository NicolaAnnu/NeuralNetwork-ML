import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from neural.network import Network

if __name__ == "__main__":
    train = pd.read_csv("datasets/monks_train1.csv")
    train = train[["class", "a1", "a2", "a3", "a4", "a5", "a6"]]
    X_train = train[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_train = train["class"].to_numpy()

    test = pd.read_csv("datasets/monks_test1.csv")
    test = test[["class", "a1", "a2", "a3", "a4", "a5", "a6"]]
    X_test = test[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_test = test["class"].to_numpy()

    net = Network(
        hidden_layer_sizes=(10, 1),
        activation="logistic",
        learning_rate=0.1,
        batch_size=200,
        max_iter=200,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation="logistic",
        solver="sgd",
        learning_rate_init=0.1,
        max_iter=200,
    )

    net.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    plt.plot(net.loss_curve)
    plt.show()

    net_pred = np.round(net.predict(X_train))
    accuracy = accuracy_score(y_train, net_pred)
    print(f"network train accuracy: {accuracy:.2f}")

    net_pred = np.round(net.predict(X_test))
    accuracy = accuracy_score(y_test, net_pred)
    print(f"network test accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_train)
    accuracy = accuracy_score(y_train, mlp_pred)
    print(f"sklearn train accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_test)
    accuracy = accuracy_score(y_test, mlp_pred)
    print(f"sklearn test accuracy: {accuracy:.2f}")
