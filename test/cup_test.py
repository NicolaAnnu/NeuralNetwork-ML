import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neural.network import Regressor

if __name__ == "__main__":
    # set headers
    names = ["ID"]
    for i in range(12):
        names.append(f"feature{i}")

    for i in range(4):
        names.append(f"target{i}")

    train = pd.read_csv(f"datasets/ml_cup_train.csv", header=None, names=names)

    # get feature and target columns
    X = train.iloc[:, 1:13].to_numpy()
    y = train.iloc[:, 13:].to_numpy()

    # shuffle the entire dataset
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # keep 10% for test
    test_size = int(np.round(X.shape[0] * 0.2))
    threshold = X.shape[0] - test_size
    X_train = X[:threshold]
    y_train = y[:threshold]
    X_test = X[threshold:]
    y_test = y[threshold:]

    # normalize train and test set
    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    net = Regressor(
        hidden_layer_sizes=(128, 128, 128),
        activation="relu",
        learning_rate=0.003,
        lam=0.0001,
        alpha=0.7,
        tol=1e-5,
        batch_size=64,
        shuffle=False,
        max_iter=5000,
    )
    net.fit(X_train, y_train, X_test, y_test)
    print(f"loss: {net.loss:.4f}")

    # training accuracy
    y_pred = net.predict(X_train)
    y_train = y_scaler.inverse_transform(y_train)
    y_pred = y_scaler.inverse_transform(y_pred)
    train_score = np.mean(np.linalg.norm(y_train - y_pred, axis=1))
    print(f"train mee: {train_score:.4f}")

    # test accuracy
    y_pred = net.predict(X_test)
    y_test = y_scaler.inverse_transform(y_test)
    y_pred = y_scaler.inverse_transform(y_pred)
    test_score = np.mean(np.linalg.norm(y_test - y_pred, axis=1))
    print(f"test mee: {test_score:.4f}")

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
