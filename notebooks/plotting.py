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
