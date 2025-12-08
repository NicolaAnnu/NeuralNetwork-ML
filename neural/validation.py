from itertools import product

import numpy as np
import psutil
from joblib import Parallel, delayed

from neural.network import Classifier, Regressor


def kfold(n, k):
    q = n // k
    r = n % k

    ranges = []
    start = 0
    for _ in range(k):
        end = start + q + (1 if r > 0 else 0)
        ranges.append([start, end])
        start = end
        r -= 1

    return ranges


def train_and_score(model_type, params, X, y, k, score_metric):
    mask = y != y
    folds = kfold(X.shape[0], k)

    scores = []
    for start, end in folds:
        mask[start:end] = True

        X_val = X[mask]
        y_val = y[mask]

        X_train = X[~mask]
        y_train = y[~mask]

        model = model_type(**params)

        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = score_metric(y_val, predictions)
            scores.append(score)
        except Exception:
            score = -np.inf

        mask[start:end] = False

    return model, np.mean(scores)


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
    retrain: bool = True,
) -> tuple[Classifier | Regressor, float]:
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    # use only physical cores
    n_cpus = psutil.cpu_count(logical=False)

    results = Parallel(n_jobs=n_cpus)(
        delayed(train_and_score)(
            model_type,
            {k: v for k, v in zip(keys, comb)},
            X,
            y,
            k,
            score_metric,
        )
        for comb in combinations
    )

    best_model, best_score = max(results, key=lambda x: x[1])

    if retrain:
        best_model.fit(X, y)

    return best_model, best_score
