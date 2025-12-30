import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neural.metrics import mean_euclidean_error, r2
from neural.network import Regressor
from neural.utils import target_plot

if __name__ == "__main__":
    # set headers
    names = ["ID"]
    features = [f"feature{i}" for i in range(12)]
    targets = [f"target{i}" for i in range(4)]
    names.extend(features)
    names.extend(targets)

    train = pd.read_csv(
        "datasets/ml_cup_train.csv",
        header=None,
        names=names,
        skiprows=7,
    )

    # get feature and target columns
    X = train.iloc[:, 1:13].to_numpy()
    y = train.iloc[:, 13:].to_numpy()

    # shuffle the entire dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # keep 20% for test
    test_size = int(np.round(X.shape[0] * 0.2))
    threshold = X.shape[0] - test_size
    X_train = X[:threshold]
    y_train = y[:threshold]
    X_test = X[threshold:]
    y_test = y[threshold:]

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    net = Regressor(
        hidden_layer_sizes=(64, 64, 64),
        activation="leaky_relu",
        learning_rate=0.07,
        lam=0.00001,
        alpha=0.9,
        shuffle=True,
        batch_size=64,
        convergence="early_stopping",
        patience=50,
        limit=-np.inf,
        max_iter=2000,
    )

    net.fit(X_train, y_train, mean_euclidean_error, X_test, y_test)

    # training
    y_pred = net.predict(X_train)
    y_train = y_scaler.inverse_transform(y_train)
    y_pred = y_scaler.inverse_transform(y_pred)
    train_mee = mean_euclidean_error(y_train, y_pred)
    train_r2 = r2(y_train, y_pred)

    # test
    y_pred = net.predict(X_test)
    y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    test_mee = mean_euclidean_error(y_test, y_pred)
    test_r2 = r2(y_test, y_pred)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.3f}")
    print(f"train MEE: {train_mee:.3f}")
    print(f"train R2: {train_r2:.3f}")

    print(f"test MEE: {test_mee:.3f}")
    print(f"test R2: {test_r2:.3f}")

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("MEE Curve")
    plt.plot(net.score_curve, label="training")
    plt.plot(net.val_score_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    target_plot(y_test, y_pred)
