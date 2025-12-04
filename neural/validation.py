from itertools import product
import numpy as np
from sklearn.model_selection import KFold
from neural.network import Classifier, Regressor


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int,
    score_metric,
    retrain: bool = False,
):
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = product(*values)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_score = -np.inf
    best_params = None
    best_model = None

    for comb in combinations:
        params = {k: v for k, v in zip(keys, comb)}
        fold_scores = []

        # K fold loop
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            model = model_type(**params)
            model.fit(X_train, y_train)

            predictions = model.predict(X_val)
            score = score_metric(y_val, predictions)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores)
        print(f"Params: {params} | mean {n_splits}-fold score: {mean_score:.3f}")

        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_model = model_type(**params)

    # retrain sul dataset completo (se richiesto)
    if retrain and best_model is not None:
        best_model.fit(X, y)

    return best_model, best_params, best_score