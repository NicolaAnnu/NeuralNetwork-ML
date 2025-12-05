import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor
from neural.validation import grid_search

if __name__ == "__main__":
    n = 500
    X = np.linspace(-10, 10, n)
    y = np.sin(X + 0.1 * np.random.randn(n))
    X = X.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1, 1)).T[0]

    X_train, X_test, y_train, y_test = [
        np.array(i) for i in train_test_split(X, y, test_size=0.1)
    ]

    hyperparams = {
        "hidden_layer_sizes": [(64, 64, 64)],
        "activation": ["relu"],
        "learning_rate": [0.003],
        "lam": [0.0],
        "alpha": [0.0],
        "tol": [1e-6],
        "batch_size": [16],
        "shuffle": [False],
        "max_iter": [1000],
    }

    net, score = grid_search(
        Regressor,
        hyperparams,
        X_train,
        y_train,
        0.1,
        mean_squared_error,
        retrain=True,
    )
    print(f"validation score: {score:.4f}")

    for k in hyperparams.keys():
        print(f"{k}: {net.__dict__[k]}")

    # net = Regressor(
    #     hidden_layer_sizes=(128,),
    #     activation="tanh",
    #     learning_rate=0.001,
    #     lam=0.00000,
    #     alpha=0.0,
    #     tol=1e-8,
    #     batch_size=16,
    #     max_iter=5000,
    # )
    # net.fit(X_train, y_train)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.4f}")

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="network")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    train_pred = net.predict(X_train)
    mse = mean_squared_error(y_train, train_pred)
    print(f"train MSE: {mse:.4f}")

    test_pred = net.predict(X_test)
    mse = mean_squared_error(y_test, test_pred)
    print(f"test MSE: {mse:.4f}")

    x = np.linspace(X.T[0].min() - 0.1, X.T[0].max() + 0.1, 100)
    y_net = net.predict(x.reshape(-1, 1))

    plt.title("Regression")
    plt.scatter(X_train.T[0], y_train, c="b", ec="w", label="train")
    plt.scatter(X_test.T[0], y_test, c="r", ec="w", label="test")
    plt.plot(x, y_net, "k-", label="network")
    plt.legend()
    plt.tight_layout()
    plt.show()
