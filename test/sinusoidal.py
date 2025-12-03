import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor

if __name__ == "__main__":
    X = np.linspace(-6, 6, 500)
    y = np.sin(X + 0.1 * np.random.randn(500))
    X = X.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).T[0]

    X_train, X_test, y_train, y_test = [
        np.array(i) for i in train_test_split(X, y, test_size=0.2)
    ]

    topology = (128, 128, 128)
    activation = "relu"
    learning_rate = 0.03
    lam = 0.0001
    alpha = 0.9
    batch_size = 32
    shuffle = False
    max_iter = 5000

    net = Regressor(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        alpha=alpha,
        tol=0,
        batch_size=batch_size,
        shuffle=shuffle,
        max_iter=max_iter,
    )

    mlp = MLPRegressor(
        hidden_layer_sizes=topology,
        activation=activation,
        solver="sgd",
        alpha=lam,
        learning_rate_init=learning_rate,
        momentum=alpha,
        nesterovs_momentum=False,
        shuffle=shuffle,
        max_iter=max_iter,
    )

    net.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    print(f"network loss: {net.loss:.4f}")
    print(f"sklearn loss: {mlp.loss_:.4f}")

    plt.plot(net.loss_curve, label="network")
    plt.plot(mlp.loss_curve_, label="sklearn")
    plt.legend()
    plt.show()

    net_pred = net.predict(X_train)
    mse = mean_squared_error(y_train, net_pred)
    print(f"network train MSE: {mse:.4f}")

    mlp_pred = mlp.predict(X_train)
    mse = mean_squared_error(y_train, mlp_pred)
    print(f"sklearn train MSE: {mse:.4f}")

    x = np.linspace(X.T[0].min() - 0.1, X.T[0].max() + 0.1, 100)
    y_net = net.predict(x.reshape(-1, 1))
    y_mlp = mlp.predict(x.reshape(-1, 1))

    plt.title("Regression on Training Data")
    plt.scatter(X_train.T[0], y_train, c="k", ec="w", label="patterns")
    plt.plot(x, y_net, "r-", label="network")
    plt.plot(x, y_mlp, "b-", label="sklearn")
    plt.legend()
    plt.tight_layout()
    plt.show()

    net_pred = net.predict(X_test)
    mse = mean_squared_error(y_test, net_pred)
    print(f"network test MSE: {mse:.4f}")

    mlp_pred = mlp.predict(X_test)
    mse = mean_squared_error(y_test, mlp_pred)
    print(f"sklearn test MSE: {mse:.4f}")

    plt.title("Regression on Test Data")
    plt.scatter(X_test.T[0], y_test, c="k", ec="w", label="patterns")
    plt.plot(x, y_net, "r-", label="network")
    plt.plot(x, y_mlp, "b-", label="sklearn")
    plt.legend()
    plt.tight_layout()
    plt.show()
