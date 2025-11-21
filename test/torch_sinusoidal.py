import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


# -----------------------------
# 1. Create a synthetic regression dataset
# -----------------------------
def create_dataset(n=500):
    X = np.linspace(-6, 6, n).reshape(-1, 1)
    y = np.sin(X) + 0.3 * np.random.randn(n, 1)
    return X, y


X, y = create_dataset()

# -----------------------------
# 2. Normalize using StandardScaler
# -----------------------------
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Convert to torch tensors
X_torch = torch.tensor(X_scaled, dtype=torch.float32)
y_torch = torch.tensor(y_scaled, dtype=torch.float32)


# -----------------------------
# 3. Define a simple MLP model
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


model = MLP()

# -----------------------------
# 4. Loss + optimizer
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -----------------------------
# 5. Training loop
# -----------------------------
epochs = 500
loss_curve = []

for epoch in range(epochs):
    optimizer.zero_grad()
    y_pred = model(X_torch)
    loss = criterion(y_pred, y_torch)
    loss.backward()
    optimizer.step()
    loss_curve.append(loss.item())

# -----------------------------
# 6. Plot training loss
# -----------------------------
plt.plot(loss_curve)
plt.title("Training Loss")
plt.show()

# -----------------------------
# 7. Predictions (denormalized)
# -----------------------------
with torch.no_grad():
    pred_scaled = model(X_torch).numpy()
    pred = scaler_y.inverse_transform(pred_scaled)

plt.scatter(X, y, s=10, label="True")
plt.plot(X, pred, color="red", label="PyTorch model")
plt.legend()
plt.show()
