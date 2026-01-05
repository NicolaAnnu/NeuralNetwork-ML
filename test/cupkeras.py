import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential, regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Dense, LeakyReLU
from keras.optimizers import SGD
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

l2 = regularizers.l2(1e-4)

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

optimizer = SGD(learning_rate=0.003, momentum=0.9, decay=0.0)

model.compile(
    optimizer=optimizer,
    loss=mean_euclidean_error,
    metrics=[mean_euclidean_error],
)


early_stopping = EarlyStopping(
    monitor="val_mean_euclidean_error",
    mode="min",
    patience=100,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=2000,
    batch_size=32,
    shuffle=True,
    callbacks=[early_stopping],
)


y_pred = model.predict(X_test)

y_pred = y_scaler.inverse_transform(y_pred)
y_test = y_scaler.inverse_transform(y_test)
mee = mean_euclidean_error(y_test, y_pred)
print(f"MEE: {mee:.4f}")


# Loss
plt.figure(figsize=(6, 5), dpi=150)
plt.title("Loss Curve")
plt.plot(history.history["loss"], label="training")
plt.plot(history.history["val_loss"], label="test")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# MEE
plt.figure(figsize=(6, 5), dpi=150)
plt.title("MEE Curve")
plt.plot(history.history["mean_euclidean_error"], label="training")
plt.plot(history.history["val_mean_euclidean_error"], label="test")
plt.xlabel("Epochs")
plt.ylabel("MEE")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
