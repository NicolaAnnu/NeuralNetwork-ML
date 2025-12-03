from itertools import product

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from neural.network import Classifier, Regressor


def train_and_score(model_type, params, X_train, y_train, X_val, y_val, score_metric):
    model = model_type(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    score = score_metric(y_val, pred)

    return score, model


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
        X, y, test_size=validation_fraction, random_state=42
    )

    results = Parallel(n_jobs=-1)(
        delayed(train_and_score)(
            model_type,
            {k: v for k, v in zip(keys, comb)},  # params
            X_train,
            y_train,
            X_val,
            y_val,
            score_metric,
        )
        for comb in combinations
    )

    # Scegli il migliore
    best_score, best_model = max(results, key=lambda x: x[0])

    if retrain:
        best_model.fit(X, y)

    return best_model
