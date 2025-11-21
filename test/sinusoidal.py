import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor

if __name__ == "__main__":
    X = np.linspace(-6, 6, 500)
    y = np.sin(X + 0.3 * np.random.randn(500))
    X = X.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).T[0]

    X_train, X_test, y_train, y_test = [
        np.array(i) for i in train_test_split(X, y, test_size=0.2)
    ]

    topology = (10,)
    activation = "tanh"
    learning_rate = 0.01
    lam = 0.0001
    alpha = 0.9
    batch_size = 10
    max_iter = 1000

    net = Regressor(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        alpha=alpha,
        batch_size=batch_size,
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
    accuracy = mean_squared_error(y_train, net_pred)
    print(f"network train accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_train)
    accuracy = mean_squared_error(y_train, mlp_pred)
    print(f"sklearn train accuracy: {accuracy:.2f}")

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
    accuracy = mean_squared_error(y_test, net_pred)
    print(f"network test accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_test)
    accuracy = mean_squared_error(y_test, mlp_pred)
    print(f"sklearn test accuracy: {accuracy:.2f}")

    plt.title("Regression on Test Data")
    plt.scatter(X_test.T[0], y_test, c="k", ec="w", label="patterns")
    plt.plot(x, y_net, "r-", label="network")
    plt.plot(x, y_mlp, "b-", label="sklearn")
    plt.legend()
    plt.tight_layout()
    plt.show()
