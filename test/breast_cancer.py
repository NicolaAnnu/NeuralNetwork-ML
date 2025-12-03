import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from neural.network import Classifier

if __name__ == "__main__":
    X, y = [np.array(i) for i in load_breast_cancer(return_X_y=True)]
    normalizer = StandardScaler()
    X = normalizer.fit_transform(X)
    X_train, X_test, y_train, y_test = [
        np.array(i) for i in train_test_split(X, y, test_size=0.2)
    ]

    topology = (8,)
    activation = "tanh"
    learning_rate = 0.01
    lam = 0.0001
    alpha = 0.9
    batch_size = 16
    max_iter = 1000

    net = Classifier(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        alpha=alpha,
        tol=1e-5,
        batch_size=batch_size,
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

    net_pred = net.predict(X_train)
    accuracy = np.mean(net_pred == y_train)
    print(f"network train accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_train)
    accuracy = np.mean(mlp_pred == y_train)
    print(f"sklearn train accuracy: {accuracy:.2f}")

    net_pred = net.predict(X_test)
    accuracy = np.mean(net_pred == y_test)
    print(f"network test accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_test)
    accuracy = np.mean(mlp_pred == y_test)
    print(f"sklearn test accuracy: {accuracy:.2f}")
