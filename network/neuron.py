import numpy as np


def logistic(x):
    return 1 / (1 + np.exp(-x))


def logistic_derivative(x):
    logit = logistic(x)
    return logit * (1 - logit)


activation_functions = {"logistic": (logistic, logistic_derivative)}


class Neuron:
    def __init__(
        self,
        activation: str = "logistic",
        learning_rate: float = 0.01,
        max_iter: int = 200,
    ) -> None:
        self.activation = activation
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        # randomly initialize weights and bias
        self.W = np.random.random(X.shape[1])
        self.b = np.random.random()

        # get the activation function from the dictionary
        activation = activation_functions[self.activation]

        self.loss = np.zeros(self.max_iter)  # to track the loss

        for epoch in range(self.max_iter):
            epoch_loss = 0.0  # loss accumulator
            for i in range(len(y)):
                # compute the scalar product (b + w^T x)
                net = self.b + (self.W @ X[i])
                out = activation[0](net)

                # compute the error on the i-th pattern
                error = out - y[i]

                # compute gradients
                weights_gradient = 2 * error * activation[1](net) * X[i]
                bias_gradient = 2 * error * activation[1](net)

                # update weigths and bias through learning rule
                self.W -= self.learning_rate * weights_gradient
                self.b -= self.learning_rate * bias_gradient

                epoch_loss += np.pow(error, 2)

            self.loss[epoch] += epoch_loss / len(y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.round(logistic(self.b + (X @ self.W)))
