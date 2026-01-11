import numpy as np


class TrainLoss:
    def __init__(
        self,
        tol: float = 1e-5,
        patience: int = 10,
        limit: float = -np.inf,
    ) -> None:
        self.tol = tol
        self.patience = patience
        self.limit = limit

        self.counter = 0
        self.best_loss = np.inf

    def should_stop(self, loss: float, val_loss: None | float) -> bool:
        if self.limit != -np.inf:
            return loss < self.limit

        # if the loss does not change enough increase counter
        if self.best_loss - loss > self.tol:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def restore_weights(self) -> bool:
        return False


class EarlyStopping:
    def __init__(
        self,
        tol: float = 0.0,
        patience: int = 10,
        limit: float = -np.inf,
    ) -> None:
        self.tol = tol
        self.patience = patience
        self.limit = limit

        self.counter = 0
        self.best_loss = np.inf

    def should_stop(self, loss: float, val_loss: float) -> bool:
        if self.limit != -np.inf:
            return loss < self.limit

        # if the val loss is worst than best loss increase counter
        if self.best_loss - val_loss > self.tol:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def restore_weights(self) -> bool:
        return True


def get_criteria(method: str, tol: float, patience: int, limit: float):
    criterias = {
        "train_loss": TrainLoss,
        "early_stopping": EarlyStopping,
    }

    return criterias[method](tol, patience, limit)
