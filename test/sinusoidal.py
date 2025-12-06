import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor
from neural.validation import grid_search


def neg_mean_squared_error(y_true, y_pred):
    return -mean_squared_error(y_true, y_pred)


def stats(net, score, hyperparams, X_train, X_test, y_train, y_test):
    print("----- grid search results -----")
    print(f"validation score: {score:.4f}")

    for k in hyperparams.keys():
        print(f"{k}: {net.__dict__[k]}")

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.4f}")

    train_pred = net.predict(X_train)
    mse = mean_squared_error(y_train, train_pred)
    print(f"train MSE: {mse:.4f}")

    test_pred = net.predict(X_test)
    mse = mean_squared_error(y_test, test_pred)
    print(f"test MSE: {mse:.4f}")

    print("")


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

    hyperparams = {
        "hidden_layer_sizes": [(32,)],
        "activation": ["tanh"],
        "learning_rate": [0.001, 0.003, 0.3],
        "lam": [0.0, 0.0001],
        "alpha": [0.0],
        "tol": [1e-5],
        "batch_size": [8, 16, 128],
        "shuffle": [False],
        "max_iter": [1000],
    }

    net, score = grid_search(
        model_type=Regressor,
        hyperparams=hyperparams,
        X=X_train,
        y=y_train,
        k=5,
        score_metric=neg_mean_squared_error,
        retrain=True,
    )
    stats(net, score, hyperparams, X_train, X_test, y_train, y_test)
    net1_loss = net.loss_curve.copy()

    x = np.linspace(X.T[0].min() - 0.1, X.T[0].max() + 0.1, 100)
    y1 = net.predict(x.reshape(-1, 1))

    net.fit(X_train, y_train)
    stats(net, score, hyperparams, X_train, X_test, y_train, y_test)
    net2_loss = net.loss_curve.copy()
    y2 = net.predict(x.reshape(-1, 1))

    plt.title("Loss Curve")
    plt.plot(net1_loss, label="retrained network")
    plt.plot(net2_loss, label="not retrained network")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # fit plots
    plt.title("Regression")
    plt.scatter(X_train.T[0], y_train, c="b", ec="w", label="train")
    plt.scatter(X_test.T[0], y_test, c="r", ec="w", label="test")

    plt.plot(x, y1, c="grey", label="retrained")
    plt.plot(x, y2, c="k", label="not retrained")
    plt.legend()
    plt.tight_layout()
    plt.show()
