import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor
from neural.validation import grid_search

if __name__ == "__main__":
    n = 500
    X = np.linspace(-6, 6, n)
    y = np.sin(X + 0.1 * np.random.randn(n))
    X = X.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).T[0]

    X_train, X_test, y_train, y_test = [
        np.array(i) for i in train_test_split(X, y, test_size=0.1)
    ]

    # hyperparams = {
    #     "hidden_layer_sizes": [(16,), (32,)],
    #     "activation": ["tanh"],
    #     "learning_rate": [0.01, 0.03],
    #     "lam": [0.0, 0.0001],
    #     "alpha": [0.0, 0.9],
    #     "tol": [1e-6],
    #     "batch_size": [8],
    #     "shuffle": [False],
    #     "max_iter": [1000],
    # }
    #
    # net, score = grid_search(
    #     Regressor,
    #     hyperparams,
    #     X_train,
    #     y_train,
    #     0.1,
    #     mean_squared_error,
    #     retrain=True,
    # )
    # print(f"validation score: {score:.4f}")
    #
    # for k in hyperparams.keys():
    #     print(f"{k}: {net.__dict__[k]}")

    net = Regressor(
        hidden_layer_sizes=(32,),
        activation="tanh",
        learning_rate=0.01,
        lam=0.0001,
        alpha=0.7,
        tol=1e-5,
        batch_size=128,
        max_iter=3000,
    )
    net.fit(X_train, y_train)
    print(f"loss: {net.loss:.4f}")

    plt.plot(net.loss_curve, label="network")
    plt.legend()
    plt.show()

    net_pred = net.predict(X_train)
    mse = mean_squared_error(y_train, net_pred)
    print(f"train MSE: {mse:.4f}")

    x = np.linspace(X.T[0].min() - 0.1, X.T[0].max() + 0.1, 100)
    y_net = net.predict(x.reshape(-1, 1))

    plt.title("Regression on Training Data")
    plt.scatter(X_train.T[0], y_train, c="k", ec="w", label="patterns")
    plt.plot(x, y_net, "r-", label="network")
    plt.legend()
    plt.tight_layout()
    plt.show()

    net_pred = net.predict(X_test)
    mse = mean_squared_error(y_test, net_pred)
    print(f"test MSE: {mse:.4f}")

    plt.title("Regression on Test Data")
    plt.scatter(X_test.T[0], y_test, c="k", ec="w", label="patterns")
    plt.plot(x, y_net, "r-", label="network")
    plt.legend()
    plt.tight_layout()
    plt.show()
