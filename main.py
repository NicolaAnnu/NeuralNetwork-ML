import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


def logistic(x):
    return 1 / (1 + np.exp(-x))


class Neuron:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def train(self, X, y):
        self.W = np.random.random(X.shape[1])
        self.b = np.random.random()

        loss = []
        for _ in range(500):
            for i in range(len(y)):
                net = self.b + (self.W @ X[i])
                out = logistic(net)
                error = out - y[i]
                loss.append(error**2)

                gradient = 2 * error * out * (1 - out) * X[i]
                gradient_bias = 2 * error * out * (1 - out)

                self.W = self.W - self.learning_rate * gradient
                self.b = self.b - self.learning_rate * gradient_bias

        return loss

    def predict(self, X):
        return np.round(logistic(self.b + (X @ self.W)))


if __name__ == "__main__":
    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=50,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            class_sep=2,
            random_state=42,
        )
    ]

    neuron = Neuron(0.3)
    loss = neuron.train(X, y)
    pred = neuron.predict(X)
    accuracy = accuracy_score(y, pred)
    print(f"neuron accuracy: {accuracy}")

    x = [i for i in range(len(loss))]
    plt.plot(x, loss)
    plt.show()

    mlp = MLPClassifier()
    mlp.fit(X, y)
    pred = mlp.predict(X)

    accuracy = accuracy_score(y, pred)
    print(f"neural network accuracy: {accuracy}")

    cls0 = X[y == 0].T
    cls1 = X[y == 1].T

    plt.scatter(cls0[0], cls0[1], c="r", label="class 0")
    plt.scatter(cls1[0], cls1[1], c="b", label="class 1")
    plt.legend()
    plt.show()
