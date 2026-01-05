import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.metrics import mean_euclidean_error
from neural.network import Regressor

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

    # stratified split
    y_scalar = np.linalg.norm(y, axis=1)
    bins = np.percentile(y_scalar, np.linspace(0, 100, 11))
    y_bins = np.digitize(y_scalar, bins[1:-1])
    X_train, X_test, y_train, y_test = [
        np.asarray(i)
        for i in train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=0,
            stratify=y_bins,
        )
    ]

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_test = np.asarray(X_scaler.transform(X_test))

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    y_test = np.asarray(y_scaler.transform(y_test))

    mlp = Regressor(
        (64, 64, 64),
        "leaky_relu",
        0.01,
        0.0001,
        0.9,
        True,
        16,
        max_iter=2000,
    )

    mlp.fit(X_train, y_train, mean_euclidean_error, X_test, y_test)

    y_pred = mlp.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred)
    y_test = y_scaler.inverse_transform(y_test)
    mee = mean_euclidean_error(y_test, y_pred)
    print(f"MEE: {mee:.4f}")

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plt.plot(mlp.loss_curve, label="training")
    plt.plot(mlp.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("MEE Curve")
    plt.plot(mlp.score_curve, label="training")
    plt.plot(mlp.val_score_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("MEE")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
