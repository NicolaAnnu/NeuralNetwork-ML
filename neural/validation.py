from itertools import product

import numpy as np
import psutil
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

from neural.network import Classifier, Regressor


def train_and_score(model_type, params, X_train, X_val, y_train, y_val, score_metric):
    model = model_type(**params)

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        score = score_metric(y_val, predictions)
    except Exception:
        score = -np.inf

    return model, score


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    validation_fraction: float,
    score_metric,
    retrain: bool = False,
) -> tuple[Classifier | Regressor, float]:
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    print(f"start validation of {len(combinations)} models")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=validation_fraction
    )

    # use only physical cores
    n_cpus = psutil.cpu_count(logical=False)

    results = Parallel(n_jobs=n_cpus)(
        delayed(train_and_score)(
            model_type,
            {k: v for k, v in zip(keys, comb)},
            X_train,
            X_val,
            y_train,
            y_val,
            score_metric,
        )
        for comb in combinations
    )

    best_model, best_score = max(results, key=lambda x: x[1])

    if retrain:
        best_model.fit(X, y)

    return best_model, best_score
