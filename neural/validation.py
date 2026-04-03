import time
from itertools import product
from typing import Type
import traceback

import numpy as np
from sklearn.preprocessing import StandardScaler

from neural.network import Network


# creates k folds of same size (some fold could differ for 1 element)
def kfold(n, k):
    min_size = n // k
    carry = n % k

    ranges = []
    start = 0
    for _ in range(k):
        end = start + min_size + (1 if carry > 0 else 0)
        ranges.append([start, end])
        start = end
        carry -= 1

    return ranges


def train_and_score(
    model: Type[Network],
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    metric,
    scale: bool,
    params: dict,
) -> dict:
    folds = kfold(X.shape[0], k)

    mask = np.array([False for _ in range(len(y))])
    scores = []
    losses = []

    for start, end in folds:
        mask[start:end] = True

        X_val = X[mask]
        y_val = y[mask]

        X_train = X[~mask]
        y_train = y[~mask]

        if scale:
            X_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train)
            X_val = np.asarray(X_scaler.transform(X_val))

            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train)
            y_val = np.asarray(y_scaler.transform(y_val))

        net = model(**params)

        try:
            net.fit(X_train, y_train, metric, X_val=X_val, y_val=y_val)
            predictions = net.predict(X_val)

            if scale:
                y_val = y_scaler.inverse_transform(y_val)
                predictions = y_scaler.inverse_transform(predictions)

            loss = np.mean((y_val - predictions) ** 2)
            score = metric(y_val, predictions)

            if net.stopping.restore_weights:
                params["limit"] = net.loss

        except Exception as e:
            print(f"\nError with params {params}: {e}")
            traceback.print_exc()
            return {
                "score": np.nan,
                "std": np.nan,
                "loss": np.inf,
                "parameters": params,
            }

        scores.append(score)
        losses.append(loss)
        mask[start:end] = False

    return {
        "score": np.mean(scores),
        "std": np.std(scores),
        "loss": np.mean(losses),
        "parameters": params,
    }


def grid_search(
    model: Type[Network],
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    metric,
    scale: bool = False,
    verbose: bool = False,
) -> list[dict]:

    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    if verbose:
        print(f"total combinations: {len(combinations)}")
        print(f"total training: {len(combinations) * k}")

    params_list = [{k: v for k, v in zip(keys, comb)} for comb in combinations]

    start = time.perf_counter()

    results = []
    for i, params in enumerate(params_list):
        if verbose:
            print(f"Running {i+1}/{len(params_list)}")

        result = train_and_score(model, X, y, k, metric, scale, params)
        results.append(result)

    end = time.perf_counter()

    if verbose:
        print(f"duration: {end - start:.2f} seconds")

    return results