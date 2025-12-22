
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import keras as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam # Import Adam optimizer
from sklearn.utils import shuffle
import pandas as pd
from keras.callbacks import EarlyStopping



# Data loading and scaling
df = pd.read_csv("ml_cup_train.csv", header=None)
X = df.iloc[:, 1:13].to_numpy()
y = df.iloc[:, 13:].to_numpy()

# 1. Genera una sequenza casuale di indici lunga quanto i tuoi dati
indices = np.random.permutation(X.shape[0])

# 2. Usa questi indici per riordinare entrambi gli array
X = X[indices]
y = y[indices]



from sklearn.preprocessing import StandardScaler


split_index = int(0.8 * X.shape[0])
X_test = X[split_index:,:]
y_test = y[split_index:,:]

X = X[:split_index,:]
y = y[:split_index,:]

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
y = scaler_Y.fit_transform(y)
X = scaler_X.fit_transform(X)



# Model definition
inputs = Input(shape=(X.shape[1],))
hidden = Dense(128,activation="relu")(inputs)
hidden = Dense(64,activation="relu")(hidden)
hidden = Dense(32,activation="relu")(hidden)
outputs = Dense(y.shape[1], activation='linear')(hidden)
model = Model(inputs=inputs, outputs=outputs)

# Model compilation
adam_optimizer = Adam(learning_rate = 0.0001) # Instantiate Adam optimizer
model.compile(
    optimizer=adam_optimizer, # Pass the optimizer instance
    loss="mse",
    metrics=["mae"] 
)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',       # Controlla l'errore sui dati di test/validazione
    patience=100,              # Aspetta 20 epoche senza miglioramenti prima di stoppare
    restore_best_weights=True
 ) # Alla fine, ripristina i pesi migliori ottenuti
# Model training
history = model.fit(
    X,
    y,
    epochs=2000,
    batch_size=16,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# Plotting training history
plt.figure(0)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

# Load test data and preprocess

X_test = scaler_X.transform(X_test)
y_test = scaler_Y.transform(y_test) 

# Evaluate model on test data
test_loss, test_mae = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)

y_pred_real = scaler_Y.inverse_transform(y_pred)
y_test = scaler_Y.inverse_transform(y_test)

# Calculate MEE
mee = np.mean(np.linalg.norm(y_test - y_pred_real, axis=1))
print("MEE:", mee)

# Print test loss and MAE
print(f"loss:{test_loss}, mae:{test_mae}")

# Calculate and print range of y_test
max_test = np.max(y_test)
min_test = np.min(y_test)  
delta_train = max_test - min_test
print(max_test, min_test)
print(f"delta train :{delta_train}")

for i in [0, 1, 2, 3]:
    score = np.mean(((y_test[:, i] - y_pred[:, i]))/(y_test[:,i]))
    print(f"RMSE output {i+1}:", score)
plt.figure(1)
plt.scatter(y_test[:, 0], y_pred_real[:, 0], label="Output 1", alpha=0.5)
plt.figure(2)
plt.scatter(y_test[:, 1], y_pred_real[:, 1], label="Output 2", alpha=0.5)
plt.figure(3)
plt.scatter(y_test[:, 2], y_pred_real[:, 2], label="Output 3", alpha=0.5)
plt.figure(4)
plt.scatter(y_test[:, 3], y_pred_real[:, 3], label="Output 4", alpha=0.5)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.show()

plt.show()