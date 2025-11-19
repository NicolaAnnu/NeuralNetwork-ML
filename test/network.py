import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from neural.network import Classifier

if __name__ == "__main__":
    n_features = 2
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=100,
            n_features=n_features,
            n_informative=n_features,
            n_redundant=0,
            n_repeated=0,
            n_classes=2,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
    ]

    topology = (5, 5)
    activation = "logistic"
    learning_rate = 0.1
    max_iter = 200

    net = Classifier(
        hidden_layer_sizes=topology,
        activation=activation,
        learning_rate=learning_rate,
        max_iter=max_iter,
    )

    mlp = MLPClassifier(
        hidden_layer_sizes=topology,
        activation=activation,
        solver="sgd",
        learning_rate_init=learning_rate,
        max_iter=max_iter,
    )

    net.fit(X, y)
    mlp.fit(X, y)

    net_pred = np.asarray([np.round(net.predict(x)) for x in X])
    mlp_pred = mlp.predict(X)

    accuracy = accuracy_score(y, net_pred)
    print(f"network accuracy: {accuracy:.2f}")

    accuracy = accuracy_score(y, mlp_pred)
    print(f"sklearn accuracy: {accuracy:.2f}")
