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


def train_and_score(params, model_type, X, y, k, score_metric):
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


def grid_search(
    model_type,
    hyperparams: dict,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
) -> tuple[Classifier | Regressor, float]:
    keys = list(hyperparams.keys())
    values = list(hyperparams.values())
    combinations = list(product(*values))

    fn = partial(
        train_and_score, model_type=model_type, X=X, y=y, k=k, score_metric=score_metric
    )
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

    scores, params = zip(*sorted(zip(scores, params), key=lambda x: x[0], reverse=True))

    best_score = scores[0]
    best_params = params[0]

    model = model_type(**best_params)
    model.fit(X, y)

    return model, best_score
