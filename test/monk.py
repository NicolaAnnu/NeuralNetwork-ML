import argparse
import pathlib, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

  # raggiungi i moduli locali
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from neural.network import Classifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    args = parser.parse_args()

    # carica dati
    train = pd.read_csv(f"datasets/monks_train{args.id}.csv")
    test = pd.read_csv(f"datasets/monks_test{args.id}.csv")

    feat_cols = ["a1", "a2", "a3", "a4", "a5", "a6"]
    X_train_raw = train[feat_cols].to_numpy()
    y_train = train["class"].to_numpy()
    X_test_raw = test[feat_cols].to_numpy()
    y_test = test["class"].to_numpy()

    # one-hot sulle feature categoriche
    encoder = OneHotEncoder(sparse_output=False)
    X_train_enc = encoder.fit_transform(X_train_raw)
    X_test_enc = encoder.transform(X_test_raw)

    # hold-out validation
    X_train, X_val, y_train, y_val = train_test_split(
          X_train_enc, y_train, test_size=0.2, random_state=42, stratify=y_train
      )

    topology = (5,)
    activation = "tanh"
    learning_rate = 0.01
    lam = 0.0001
    alpha = 0.9
    batch_size = 10
    max_iter = 500

    net = Classifier(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
      )

    mlp = MLPClassifier(
        hidden_layer_sizes=topology,
        activation=activation,
        solver="sgd",
        alpha=lam,
        learning_rate_init=learning_rate,
        momentum=alpha,
        max_iter=max_iter,
        random_state=42,
      )

    net.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    print(f"network loss: {net.loss:.2f}")
    print(f"sklearn loss: {mlp.loss_:.2f}")

    plt.plot(net.loss_curve, label="network")
    plt.plot(mlp.loss_curve_, label="sklearn")
    plt.legend()
    plt.show()

    # validazione
    preds_val = net.predict(X_val)
    acc_val = accuracy_score(y_val, preds_val)
    print(f"Validation accuracy: {acc_val:.2f}")

    # train accuracy
    train_acc_net = accuracy_score(y_train, net.predict(X_train))
    train_acc_mlp = accuracy_score(y_train, mlp.predict(X_train))
    print(f"network train accuracy: {train_acc_net:.2f}")
    print(f"sklearn train accuracy: {train_acc_mlp:.2f}")

    # test accuracy
    test_acc_net = accuracy_score(y_test, net.predict(X_test_enc))
    test_acc_mlp = accuracy_score(y_test, mlp.predict(X_test_enc))
    print(f"network test accuracy: {test_acc_net:.2f}")
    print(f"sklearn test accuracy: {test_acc_mlp:.2f}")