
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
outputs = Dense(y.shape[1], activation='linear')(hidden)
model = Model(inputs=inputs, outputs=outputs)

# Model compilation
adam_optimizer = Adam(learning_rate = 0.0005) # Instantiate Adam optimizer
model.compile(
    optimizer=adam_optimizer, # Pass the optimizer instance
    loss="mse",
    metrics=["mae"] 
)
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(
    monitor='val_loss',       # Controlla l'errore sui dati di test/validazione
    patience=40,              # Aspetta 10 epoche senza miglioramenti prima di stoppare
    restore_best_weights=True
 ) # Alla fine, ripristina i pesi migliori ottenuti
# Model training
history = model.fit(
    X,
    y,
    epochs=2000,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Plotting training history
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

plt.show()