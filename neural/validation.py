import json
import multiprocessing as mp
from functools import partial
from itertools import product

import numpy as np
import psutil
from sklearn.preprocessing import StandardScaler
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


def train_and_score(
    model_type,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
    scale: bool,
    params: dict,
):
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
    scale: bool = False,
    verbose: bool = False,
) -> tuple[Classifier | Regressor, float]:
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    fn = partial(train_and_score, model_type, X, y, k, score_metric, scale)
    params = [{k: v for k, v in zip(keys, comb)} for comb in combinations]

    # use only physical cores
    n_cpus = psutil.cpu_count(logical=False)

    # run the search in parallel
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
    if verbose:
        # print the number of crashed folds
        print(f"failed {scores.count(-np.inf)} times")

        # print top 3 scores and hyperparameters
        for s, p in zip(scores[:3], params[:3]):
            print(f"score: {s}")
            print(f"{json.dumps(p, indent=4)}")

    best_score = scores[0]
    best_params = params[0]

    # get the best model and retrain on full training set
    model = model_type(**best_params)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y)

    model.fit(X, y)

    return model, best_score


def nested_grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
    scale: bool = False,
    verbose: bool = False,
) -> tuple[Classifier | Regressor, float]:
    model, _ = grid_search(
        model_type,
        hyperparams,
        X,
        y,
        k,
        score_metric,
        scale,
        verbose,
    )

    hyperparams2 = {}
    for key in hyperparams.keys():
        best_value = model.__dict__[key]

        if len(hyperparams[key]) == 1:
            hyperparams2[key] = [best_value]
        elif isinstance(best_value, float):
            center = best_value
            width = best_value / 3
            hyperparams2[key] = np.linspace(
                center - width, center + width, 4, dtype=float
            ).tolist()
        elif isinstance(best_value, int):
            # center = best_value
            # width = best_value // 2
            # hyperparams2[key] = np.linspace(
            #     center - width, center + width, 3, dtype=int
            # ).tolist()
            hyperparams2[key] = [best_value]
        else:
            hyperparams2[key] = [best_value]

    return grid_search(model_type, hyperparams2, X, y, k, score_metric, scale, verbose)
