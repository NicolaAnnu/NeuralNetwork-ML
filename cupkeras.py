import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler

# --- 1. CARICAMENTO E PREPARAZIONE DATI ---
# Assicurati che il file sia nella stessa cartella
try:
    df = pd.read_csv("ml_cup_train.csv", header=None)
except FileNotFoundError:
    print("Errore: File 'ml_cup_train.csv' non trovato. Carico dati dummy per test.")
    # Genero dati casuali solo se non trova il file, per farti vedere che il codice gira
    df = pd.DataFrame(np.random.rand(1000, 15))

# Separazione Feature e Target (adatta gli indici se necessario)
X = df.iloc[:, 1:13].to_numpy() # Colonne 1 a 12
y = df.iloc[:, 13:].to_numpy()  # Colonne 13 in poi

# Shuffling
indices = np.random.permutation(X.shape[0])
X = X[indices]
y = y[indices]

# Splitting (80% Train, 20% Test)
split_index = int(0.8 * X.shape[0])
X_train = X[:split_index, :]
y_train = y[:split_index, :]
X_test = X[split_index:, :]
y_test = y[split_index:, :]

# Scaling
# IMPORTANTE: Il fit si fa SOLO sul train per evitare data leakage
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

scaler_Y = StandardScaler()
y_train = scaler_Y.fit_transform(y_train)
y_test_scaled = scaler_Y.transform(y_test) # Scaliamo anche y_test per calcolare la loss durante il test

# --- 2. DEFINIZIONE MODELLO ---
inputs = Input(shape=(X_train.shape[1],))

# L2 va inserito qui, nei layer (kernel_regularizer)
hidden = Dense(64, activation="leaky_relu", kernel_regularizer=l2(1e-6))(inputs)
hidden = Dense(64, activation="leaky_relu", kernel_regularizer=l2(1e-6))(hidden)
hidden = Dense(64, activation="leaky_relu", kernel_regularizer=l2(1e-6))(hidden)

outputs = Dense(y_train.shape[1], activation='linear')(hidden)

model = Model(inputs=inputs, outputs=outputs)

# --- 3. COMPILAZIONE ---
# Qui definisci il momento
opt = SGD(learning_rate=0.001, momentum=0.9)

model.compile(
    optimizer=opt,
    loss="mse",
    metrics=["mae"] # Aggiunto MAE per averlo nella valutazione finale
)

# --- 4. TRAINING ---

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=0
)

print("Inizio training...")
history = model.fit(
    X_train,
    y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.1, # Prende il 10% di X_train per la validazione durante le epoche
   # callbacks=[early_stopping],
    verbose=1
)

# --- 5. GRAFICI ---
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title('Curve di apprendimento')
plt.xlabel('Epoche')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.show()

# --- 6. VALUTAZIONE E MEE ---
# Valutazione sui dati di test scalati
results = model.evaluate(X_test, y_test_scaled, verbose=0)
test_loss = results[0] # MSE
test_mae = results[1]  # MAE

# Predizione
y_pred_scaled = model.predict(X_test)

# Inversione dello scaling per tornare ai valori reali
y_pred_real = scaler_Y.inverse_transform(y_pred_scaled)
# Nota: y_test era l'originale non scalato, quindi usiamo quello per il confronto reale

# Calcolo MEE (Mean Euclidean Error)
# Calcola la distanza euclidea per ogni riga, poi fa la media
mee = np.mean(np.linalg.norm(y_test - y_pred_real, axis=1))

print(f"Test Loss (MSE Scaled): {test_loss:.4f}")
print(f"Test MAE (Scaled):      {test_mae:.4f}")
print(f"MEE (Errore Reale):     {mee:.4f}")


# Statistiche sui dati
max_test = np.max(y_test)
min_test = np.min(y_test)
delta_data = max_test - min_test
print(f"Range dati reali: {min_test:.2f} - {max_test:.2f} (Delta: {delta_data:.2f})")