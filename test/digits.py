import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from neural.network import Classifier


def accuracy_one_hot(y_true, y_pred):
    # y_true è one-hot, y_pred sono classi
    y_true_cls = np.argmax(y_true, axis=1)
    return accuracy_score(y_true_cls, y_pred)


if __name__ == "__main__":
    X, y = load_digits(return_X_y=True)

    X_train, X_test, y_train, y_test = [
        np.asarray(i) for i in train_test_split(X, y, test_size=0.1)
    ]

    print(X.shape)

    # re-train the model
    net = Classifier(
        hidden_layer_sizes=(64, 32),
        activation="leaky_relu",
        learning_rate=0.01,
        lam=0.0001,
        alpha=0.7,
        shuffle=True,
        batch_size=64,
        convergence="train_loss",
        tol=0.0,
        patience=50,
        max_iter=1000,
    )
    net.fit(X_train, y_train, accuracy_one_hot, X_val=X_test, y_val=y_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"training loss: {net.loss:.4f}")
    print(f"test loss: {net.val_loss:.4f}")

    # training accuracy
    y_pred = net.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred)
    train_f1 = f1_score(y_train, y_pred, average="macro")
    print(f"train accuracy: {train_accuracy:.2f}")
    print(f"train f1: {train_f1:.2f}")

    # test accuracy
    y_pred = net.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"test accuracy: {test_accuracy:.2f}")
    print(f"test f1: {test_f1:.2f}")

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Accuracy Curve")
    plt.plot(net.score_curve, label="training")
    plt.plot(net.val_score_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()
