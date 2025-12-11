import json
import multiprocessing as mp
from functools import partial
from itertools import product

import numpy as np
import psutil
from tqdm import tqdm

from neural.network import Classifier, Regressor


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


def train_and_score(model_type, X, y, k, score_metric, params):
    folds = kfold(X.shape[0], k)

    mask = np.array([False for _ in range(len(y))])
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
        except Exception:
            score = -np.inf

        scores.append(score)
        mask[start:end] = False

    return np.mean(scores)


def order(x: tuple):
    score, params = x
    hls = params["hidden_layer_sizes"]
    lam = params["lam"]

    return (score, -len(hls), -sum(hls), lam)


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
    log: bool = True,
) -> tuple[Classifier | Regressor, float]:
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    fn = partial(train_and_score, model_type, X, y, k, score_metric)
    params = [{k: v for k, v in zip(keys, comb)} for comb in combinations]

    # use only physical cores
    n_cpus = psutil.cpu_count(logical=False)
    with mp.Pool(processes=n_cpus) as pool:
        scores = []
        for res in tqdm(
            pool.imap_unordered(fn, params),
            total=len(params),
            desc="grid search",
            ncols=80,
        ):
            scores.append(res)

    scores, params = zip(*sorted(zip(scores, params), key=order, reverse=True))

    # log some statistics
    if log:
        print(f"failed {scores.count(-np.inf)} times")
        for s, p in zip(scores, params):
            print(f"score: {s}")
            print(f"{json.dumps(p, indent=4)}")

    best_score = scores[0]
    best_params = params[0]

    # get the best model and retrain on full training set
    model = model_type(**best_params)
    model.fit(X, y)

    return model, best_score


def nested_grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
    log: bool = True,
) -> tuple[Classifier | Regressor, float]:
    hyperparams2 = {}
    for key in hyperparams.keys():
        v = hyperparams[key]
        if isinstance(v[0], float) or isinstance(v[0], int):
            if len(v) == 1:
                hyperparams2[key] = v
            else:
                hyperparams2[key] = [v[0], v[len(v) // 2]]
        else:
            hyperparams2[key] = v

    keys = list(hyperparams2.keys())
    values = list(hyperparams2.values())
    combinations = list(product(*values))

    fn = partial(train_and_score, model_type, X, y, k, score_metric)
    params = [{k: v for k, v in zip(keys, comb)} for comb in combinations]

    # use only physical cores
    n_cpus = psutil.cpu_count(logical=False)
    with mp.Pool(processes=n_cpus) as pool:
        scores = []
        for res in tqdm(
            pool.imap_unordered(fn, params),
            total=len(params),
            desc="nested grid search",
            ncols=80,
        ):
            scores.append(res)

    scores, params = zip(*sorted(zip(scores, params), key=order, reverse=True))

    hyperparams2 = {}
    for key in hyperparams.keys():
        v = hyperparams[key]
        if isinstance(v[0], float) or isinstance(v[0], int):
            if len(v) == 1:
                hyperparams2[key] = v
            else:
                if params[0][key] == v[0]:
                    hyperparams2[key] = v[: len(v) // 2]
                else:
                    hyperparams2[key] = v[len(v) // 2 :]
        else:
            hyperparams2[key] = v

    return grid_search(model_type, hyperparams2, X, y, k, score_metric, log)
