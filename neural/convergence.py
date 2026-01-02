import numpy as np


class LossConvergence:
    def __init__(
        self,
        patience: int = 10,
        limit: float = -np.inf,
    ) -> None:
        self.patience = patience
        self.limit = limit

        self.counter = 0
        self.best_loss = np.inf

    def should_stop(self, loss: float, val_loss: None | float) -> bool:
        if self.limit != -np.inf:
            return loss < self.limit

        if abs(self.best_loss - loss) < 1e-6:
            self.counter += 1
        else:
            if loss < self.best_loss:
                self.best_loss = loss
                self.counter = 0

        return self.counter >= self.patience

    @property
    def restore_weights(self) -> bool:
        return False


class EarlyStopping:
    def __init__(
        self,
        patience: int = 10,
        limit: float = -np.inf,
    ) -> None:
        self.patience = patience
        self.limit = limit

        self.counter = 0
        self.best_loss = np.inf

    def should_stop(self, loss: float, val_loss: float) -> bool:
        if self.limit != -np.inf:
            return loss < self.limit

        if self.best_loss > val_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

    @property
    def restore_weights(self) -> bool:
        return True


methods = {
    "loss_convergence": LossConvergence,
    "early_stopping": EarlyStopping,
}
