
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import keras as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam # Import Adam optimizer

import pandas as pd

# Data loading and scaling
df = pd.read_csv("ml_cup_train.csv", header=None)
X = df.iloc[:, 1:9].to_numpy()
y = df.iloc[:, 9:13].to_numpy()

from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
y = scaler_Y.fit_transform(y)
X = scaler_X.fit_transform(X)

# Model definition
inputs = Input(shape=(X.shape[1],))
hidden = Dense(64,activation="relu")(inputs)
outputs = Dense(y.shape[1], activation='linear')(hidden)
model = Model(inputs=inputs, outputs=outputs)

# Model compilation
adam_optimizer = Adam(learning_rate=0.001) # Instantiate Adam optimizer
model.compile(
    optimizer=adam_optimizer, # Pass the optimizer instance
    loss="mse",
    metrics=["mae"] 
)

# Model training
history = model.fit(
    X,
    y,
    epochs=50,
    batch_size=16,
    validation_split=0.2
)

# Plotting training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Load test data and preprocess
df = pd.read_csv("ml_cup_test.csv", header=None)
X_test = df.iloc[:, 1:9].to_numpy()
y_test = df.iloc[:, 9:13].to_numpy()
X_test = scaler_X.transform(X_test)
y_test = scaler_Y.transform(y_test) 

# Evaluate model on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

# Calculate MEE
mee = np.mean(np.linalg.norm(y_test - y_pred, axis=1))
print("MEE:", mee)

# Print test loss and MAE
print(f"loss:{test_loss}, mae:{test_mae}")

# Calculate and print range of y_test
max_test = np.max(y_test)
min_test = np.min(y_test)  
delta_train = max_test - min_test
print(max_test, min_test)
print(f"delta train :{delta_train}")

plt.show()