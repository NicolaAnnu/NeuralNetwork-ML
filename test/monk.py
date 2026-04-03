import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

from neural.network import Classifier
from neural.utils import stats
from neural.validation import grid_search

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("id", type=int, help="monk dataset ID")
    args = parser.parse_args()

    train = pd.read_csv(f"datasets/monks_train{args.id}.csv")
    X_train = train[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_train = train["class"].to_numpy()

    test = pd.read_csv(f"datasets/monks_test{args.id}.csv")
    X_test = test[["a1", "a2", "a3", "a4", "a5", "a6"]].to_numpy()
    y_test = test["class"].to_numpy()

    encoder = OneHotEncoder(sparse_output=False)
    X_train = encoder.fit_transform(X_train)
    X_test = encoder.transform(X_test)

<<<<<<< HEAD
    hyperparams = {
        "hidden_layer_sizes": [(3,)],
        "activation": ["logistic", "tanh"],
        "learning_rate": [0.01, 0.03],
        "lam": [0.0, 0.0001],
        "alpha": [0.9],
        "tol": [1e-5],
        "batch_size": [8, 16, 32, 64],
        "shuffle": [False, True],
        "max_iter": [1000],
    }

    net, score = grid_search(
        model_type=Classifier,
        hyperparams=hyperparams,
        X=X_train,
        y=y_train,
        k=10,
        score_metric=accuracy_score,
        verbose = False,
    )
    # training accuracy
    net_pred = net.predict(X_train)
    train_score = accuracy_score(y_train, net_pred)
    print(f"train accuracy: {train_score:.2f}")

    # test accuracy
    net_pred = net.predict(X_test)
    test_score = accuracy_score(y_test, net_pred)
    print(f"test accuracy: {test_score:.2f}")

    # print stats and save results to json file
    stats(
        net,
        hyperparams,
        score,
        train_score,
        test_score,
        f"results/monk{args.id}.json",
=======
    topology = (3,)
    activation = "tanh"
    learning_rate = 0.3
    lam = 0.0001
    alpha = 0.9
    batch_size = 10
    max_iter = 1000

    net = Classifier(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        lam=lam,
        alpha=alpha,
        batch_size=-1,
        shuffle=True,
        max_iter=max_iter,
>>>>>>> main
    )

    plt.title("Loss Curve")
    plt.plot(net.loss_curve, label="loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
<<<<<<< HEAD
    plt.tight_layout()
    plt.show()
=======
    plt.show()
    
    # Network
    net_pred = net.predict(X_train)
    accuracy = np.mean(net_pred == y_train)
    print(f"network train accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_train)
    accuracy = np.mean(mlp_pred == y_train)
    print(f"sklearn train accuracy: {accuracy:.2f}")

    # Test set
    net_pred = net.predict(X_test)
    accuracy = np.mean(net_pred == y_test)
    print(f"network test accuracy: {accuracy:.2f}")

    mlp_pred = mlp.predict(X_test)
    accuracy = np.mean(mlp_pred == y_test)
    print(f"sklearn test accuracy: {accuracy:.2f}")

>>>>>>> main
