import time
from itertools import product

import numpy as np
from dask.delayed import delayed
from dask.distributed import Client, progress
from sklearn.preprocessing import StandardScaler


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
    model,
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
    for start, end in folds:
        mask[start:end] = True

        X_val = X[mask]
        y_val = y[mask]

        X_train = X[~mask]
        y_train = y[~mask]

        if scale:
            X_scaler = StandardScaler()
            X_train = X_scaler.fit_transform(X_train)
            X_val = X_scaler.transform(X_val)

            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train)
            y_val = y_scaler.transform(y_val)

        net = model(**params)

        try:
            net.fit(X_train, y_train)
            predictions = net.predict(X_val)

            if scale:
                y_val = y_scaler.inverse_transform(y_val)
                predictions = y_scaler.inverse_transform(predictions)

            score = metric(y_val, predictions)
        except Exception:
            return {
                "score": -np.inf,
                "std": np.inf,
                "parameters": params,
            }

        scores.append(score)
        mask[start:end] = False

    return {
        "score": np.mean(scores),
        "std": np.std(scores),
        "parameters": params,
    }


def grid_search(
    model,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    metric,
    scale: bool = False,
    address: None | str = None,
    verbose: bool = False,
) -> list[dict]:
    # dask init
    if address:
        client = Client(address)
    else:
        client = Client()

    if verbose:
        print(f"dask dashboard: {client.dashboard_link}")

    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    if verbose:
        print(f"total combinations: {len(combinations)}")
        print(f"total training: {len(combinations) * k}")

    params_list = [{k: v for k, v in zip(keys, comb)} for comb in combinations]
    tasks = [
        delayed(train_and_score)(model, X, y, k, metric, scale, params)
        for params in params_list
    ]

    # perform parallel k-folds
    start = time.perf_counter()
    futures = client.compute(tasks)
    if verbose:
        progress(futures)
    results = client.gather(futures)
    end = time.perf_counter()

    # log some statistics
    assert isinstance(results, list)
    if verbose:
        failed = sum(r["score"] == -np.inf for r in results)
        print(f"failed {failed} times")
        print(f"duration: {end - start:.2f} seconds")

    return results
