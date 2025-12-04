from itertools import product

import numpy as np
from sklearn.model_selection import train_test_split

from neural.network import Classifier, Regressor


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    score_metric,
    retrain: bool = False,
) -> Classifier | Regressor:
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = product(*values)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_fraction
    )

    best_score = -np.inf
    best_model = None
    for comb in combinations:
        params = {k: v for k, v in zip(keys, comb)}

        model = model_type(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        score = score_metric(y_val, predictions)

        if score > best_score:
            best_score = score
            best_model = model

    assert best_model is not None
    if retrain:
        best_model.fit(X, y)

    return best_model
