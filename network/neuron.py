import numpy as np

from network.activations import activations


class Neuron:
    def __init__(
        self,
        activation: str = "logistic",
        learning_rate: float = 0.01,
    ) -> None:
        self.activation = activations[activation]
        self.learning_rate = learning_rate

    def init_weights(self, n: int) -> None:
        # randomly initialize weights and bias
        self.W = np.random.random(n)
        self.b = np.random.random()

    def output(self, X: np.ndarray) -> np.ndarray:
        # save for backpropagation
        self.net = self.b + X @ self.W
        self.out = self.activation[0](self.net)

        return self.out

    def update_weights(self, X: np.ndarray, error: np.ndarray) -> None:
        delta = 2 * error * self.activation[1](self.net)

        # compute gradients
        weights_gradient = X.T @ delta
        bias_gradient = np.sum(delta)

        # update weights and bias through learning rule
        self.W -= self.learning_rate * weights_gradient
        self.b -= self.learning_rate * bias_gradient


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.metrics import accuracy_score

    X, y = [
        np.array(i)
        for i in make_classification(
            n_samples=100,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_repeated=0,
            n_clusters_per_class=1,
            class_sep=1.5,
            random_state=0,
        )
    ]

    neuron = Neuron(activation="logistic", learning_rate=0.1)
    neuron.init_weights(X.shape[1])

    batch_size = 5
    for epoch in range(500):
        for i in range(0, len(y), batch_size):
            out = neuron.output(X[i : i + batch_size, :])
            error = out - y[i : i + batch_size]
            neuron.update_weights(X[i : i + batch_size, :], error)

    out = np.round(neuron.output(X))
    accuracy = accuracy_score(y, out)
    print(f"accuracy: {accuracy:.2f}")
