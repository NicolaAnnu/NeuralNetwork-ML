import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Dataset ---
X = np.linspace(-6, 6, 500)
y = np.sin(X + 0.3 * np.random.randn(500))
X = X.reshape(-1, 1)

# Normalizzazione
scaler_X = StandardScaler()
X = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y.reshape(-1, 1)).T[0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
assert isinstance(X_train, np.ndarray)

# --- Parametri rete ---
topology = (10,)
activation = "tanh"
learning_rate = 0.001
momentum = 0.7
lam = 0.0001
batch_size = 10
max_iter = 1000  # epoche

# --- Modello Keras ---
model = Sequential()
model.add(
    Dense(
        topology[0],
        input_shape=(X_train.shape[1],),
        activation=activation,
        kernel_regularizer=l2(lam),
    )
)

# output layer lineare
model.add(Dense(1, activation="linear"))

optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
model.compile(optimizer=optimizer, loss="mse")

# Fit
history = model.fit(X_train, y_train, epochs=max_iter, batch_size=batch_size)

# Loss curve
plt.plot(history.history["loss"], label="Keras")
plt.legend()
plt.show()

# Predizioni
y_pred_train = model.predict(X_train).flatten()
y_pred_test = model.predict(X_test).flatten()

print(f"Train MSE: {mean_squared_error(y_train, y_pred_train):.4f}")
print(f"Test MSE: {mean_squared_error(y_test, y_pred_test):.4f}")

# Plot
x_plot = np.linspace(X.T[0].min() - 0.1, X.T[0].max() + 0.1, 100).reshape(-1, 1)
y_plot = model.predict(x_plot).flatten()

plt.scatter(X_train.T[0], y_train, c="k", label="Train patterns")
plt.plot(x_plot, y_plot, "r-", label="Keras prediction")
plt.legend()
plt.show()
