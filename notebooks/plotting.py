import matplotlib.pyplot as plt
import numpy as np


def loss_curve(loss):
    plt.figure(figsize=(5, 3), dpi=150)
    plt.title("Loss Curve")

    plt.plot(loss, label="loss")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # griglia 2D
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # predizioni sulla griglia
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(5, 4), dpi=150)
    plt.title("Decision Boundary")

    # decision boundary (regioni colorate)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu")

    cls0 = X[y == 0].T
    cls1 = X[y == 1].T

    plt.scatter(cls0[0], cls0[1], c="r", ec="w", label="class 0")
    plt.scatter(cls1[0], cls1[1], c="b", ec="w", label="class 1")

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.show()


def loss_decision_boundary(loss1, loss2, X, y, model1, model2):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    # ---------------------
    # Loss curve
    # ---------------------
    m = min(len(loss1), len(loss2))
    ax1.set_title("Loss Curve")
    ax1.plot(loss1[:m], label="neuron")
    ax1.plot(loss2[:m], label="sklearn")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # ---------------------
    # Decision boundary
    # ---------------------
    ax2.set_title("Decision Boundary")

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Grid
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predictions
    Z1 = model1.predict(grid).reshape(xx.shape)
    Z2 = model2.predict(grid).reshape(xx.shape)

    # Plot ONLY the boundary (level=0.5)
    c1 = ax2.contour(xx, yy, Z1, levels=[0.5], colors="cyan", linewidths=1)
    c2 = ax2.contour(xx, yy, Z2, levels=[0.5], colors="orange", linewidths=1)

    # Scatter points
    ax2.scatter(X[y == 0, 0], X[y == 0, 1], c="red", label="class 0", ec="k")
    ax2.scatter(X[y == 1, 0], X[y == 1, 1], c="blue", label="class 1", ec="k")

    # Custom legend for boundaries
    boundary_handles = [
        c1.legend_elements()[0][0],
        c2.legend_elements()[0][0],
    ]
    ax2.legend(
        boundary_handles + ax2.collections[-2:],  # boundaries + points
        ["sklearn", "neuron", "class 0", "class 1"],
    )

    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")

    plt.tight_layout()
    plt.show()


def loss_fit(loss1, loss2, X, y, model1, model2):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

    m = min(len(loss1), len(loss2))
    ax1.set_title("Loss Curve")
    ax1.plot(loss1[:m], label="neuron")
    ax1.plot(loss2[:m], label="sklearn")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.set_title("Regression")
    ax2.scatter(X.T[0], y, ec="k", label="examples")

    x = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    pred = model1.predict(x.reshape(-1, 1))
    ax2.plot(x, pred, "g-", label="neuron")

    x = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)
    pred = model2.predict(x.reshape(-1, 1))
    ax2.plot(x, pred, "r-", label="sklearn")

    ax2.set_xlabel("Feature")
    ax2.set_ylabel("Target")
    ax2.legend()

    plt.tight_layout()
    plt.show()
