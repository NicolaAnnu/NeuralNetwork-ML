import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LeakyReLU
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def mean_euclidean_error(y_true, y_pred):
    return tf.reduce_mean(tf.norm(y_true - y_pred, axis=1))


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


train_mses = []
test_mses = []

train_mees = []
test_mees = []

train_mse_curves = []
test_mse_curves = []
train_mee_curves = []
test_mee_curves = []

for i in range(10):
    l2 = regularizers.l2(1e-5)

    model = Sequential(
        [
            Dense(64, input_shape=(X_train.shape[1],), kernel_regularizer=l2),
            LeakyReLU(alpha=0.01),
            Dense(64, kernel_regularizer=l2),
            LeakyReLU(alpha=0.01),
            Dense(64, kernel_regularizer=l2),
            LeakyReLU(alpha=0.01),
            Dense(4, kernel_regularizer=l2),  # output regressione 4D
        ]
    )

    optimizer = SGD(learning_rate=0.01, momentum=0.9, decay=0.0)

    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[mean_euclidean_error],
    )

    early_stopping = EarlyStopping(
        monitor="val_mean_euclidean_error",
        mode="min",
        patience=100,
        restore_best_weights=True,
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=1500,
        batch_size=16,
        shuffle=True,
        callbacks=[early_stopping],
    )

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    y_train_pred = y_scaler.inverse_transform(y_train_pred)
    y_test_pred = y_scaler.inverse_transform(y_test_pred)

    y_train_raw = y_scaler.inverse_transform(y_train)
    y_test_raw = y_scaler.inverse_transform(y_test)

    train_mse = mean_squared_error(y_train_raw, y_train_pred)
    test_mse = mean_squared_error(y_test_raw, y_test_pred)
    train_mses.append(train_mse)
    test_mses.append(test_mse)

    train_mee = mean_euclidean_error(y_train_raw, y_train_pred).numpy().item()
    test_mee = mean_euclidean_error(y_test_raw, y_test_pred).numpy().item()
    train_mees.append(train_mee)
    test_mees.append(test_mee)

    train_mse_curves.append(list(history.history["loss"]))
    test_mse_curves.append(list(history.history["val_loss"]))

    train_mee_curves.append(list(history.history["mean_euclidean_error"]))
    test_mee_curves.append(list(history.history["val_mean_euclidean_error"]))

data = {
    "loss": np.mean(train_mses),
    "val_loss": np.mean(test_mses),
    "loss_curve": train_mse_curves,
    "val_loss_curve": test_mse_curves,
    "score": np.mean(train_mees),
    "val_score": np.mean(test_mees),
    "score_curve": train_mee_curves,
    "val_score_curve": test_mee_curves,
}

with open("results/curves/keras.json", "w") as fp:
    json.dump(data, fp, indent=2)


# # Loss
# plt.figure(figsize=(6, 4), dpi=150)
# plt.title("Loss Curve")
# plt.plot(history.history["loss"], label="training")
# plt.plot(history.history["val_loss"], label="test")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()

# # MEE
# plt.figure(figsize=(6, 4), dpi=150)
# plt.title("MEE Curve")
# plt.plot(history.history["mean_euclidean_error"], label="training")
# plt.plot(history.history["val_mean_euclidean_error"], label="test")
# plt.xlabel("Epochs")
# plt.ylabel("MEE")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()
