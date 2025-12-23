import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from neural.network import Classifier

if __name__ == "__main__":
    X, y = [np.array(i) for i in load_breast_cancer(return_X_y=True)]

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = [
        np.asarray(i) for i in train_test_split(X, y, test_size=0.2)
    ]
    X_train = scaler.fit_transform(X_train)
    X_test = np.asarray(scaler.transform(X_test))

    net = Classifier(
        hidden_layer_sizes=(16,),
        activation="elu",
        learning_rate=0.01,
        lam=0.0,
        alpha=0.7,
        tol=1e-6,
        batch_size=16,
        shuffle=True,
        early_stopping=True,
        max_iter=2000,
    )

    net.fit(X_train, y_train, X_test, y_test)
    y_pred = net.predict(X_test)

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.3f}")

    # training accuracy
    net_pred = net.predict(X_train)
    train_accuracy = accuracy_score(y_train, net_pred)
    print(f"train accuracy: {train_accuracy:.2f}")

    # train f1
    train_f1 = f1_score(y_train, net_pred)
    print(f"train f1: {train_f1:.2f}")

    # train confusion matrix
    train_cm = confusion_matrix(y_train, net_pred)
    ConfusionMatrixDisplay(train_cm).plot()
    plt.show()

    # test accuracy
    net_pred = net.predict(X_test)
    test_accuracy = accuracy_score(y_test, net_pred)
    print(f"test accuracy: {test_accuracy:.2f}")

    # test f1
    test_f1 = f1_score(y_test, net_pred)
    print(f"test f1: {test_f1:.2f}")

    # test confusion matrix
    test_cm = confusion_matrix(y_test, net_pred)
    ConfusionMatrixDisplay(test_cm).plot()
    plt.show()

    plt.figure(figsize=(6, 5), dpi=150)
    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="training")
    plt.plot(net.val_loss_curve, label="test")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()
