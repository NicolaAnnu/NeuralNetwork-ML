from itertools import product

import numpy as np
from sklearn.model_selection import train_test_split


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    score_metric,
):
    combinations = product(*hyperparams.values())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_fraction
    )

    best_score = 0.0
    best_model = model_type()
    for comb in combinations:
        for i, k in enumerate(hyperparams.keys()):
            hyperparams[k] = comb[i]

        model = model_type(**hyperparams)
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = score_metric(y_val, predictions)

        if score > best_score:
            best_score = score
            best_model = model

    best_model.fit(X, y)

    return best_model
