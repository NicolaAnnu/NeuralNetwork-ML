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
    model_type,
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    score_metric,
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

        model = model_type(**params)

        try:
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            score = score_metric(y_val, predictions)
        except Exception:
            return {"score": float(-np.inf), "parameters": params}

        scores.append(score)
        mask[start:end] = False

    return {"score": float(np.mean(scores) - np.std(scores)), "parameters": params}


def order(x: dict):
    score, params = x["score"], x["parameters"]
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
        print(f"{len(combinations) * k} total training")

    params_list = [{k: v for k, v in zip(keys, comb)} for comb in combinations]
    tasks = [
        delayed(train_and_score)(model_type, X, y, k, score_metric, scale, params)
        for params in params_list
    ]

    try:
        start = time.perf_counter()
        futures = client.compute(tasks)
        if verbose:
            progress(futures)

        results = client.gather(futures)
        end = time.perf_counter()
    except KeyboardInterrupt:
        print("keybaord interrupt")
        client.cancel(futures, force=True)
    finally:
        client.close()

    results = sorted(results, key=order, reverse=True)
    # log some statistics
    if verbose:
        failed = sum(r["score"] == -np.inf for r in results)
        print(f"failed {failed} times")
        print(f"duration: {end - start:.2f} seconds")

    return results


# def nested_grid_search(
#     model_type,
#     hyperparams: dict,
#     X: np.ndarray,
#     y: np.ndarray,
#     k: int,
#     score_metric,
#     scale: bool = False,
#     verbose: bool = False,
# ) -> tuple[Classifier | Regressor, float]:
#     model, _ = grid_search(
#         model_type,
#         hyperparams,
#         X,
#         y,
#         k,
#         score_metric,
#         scale,
#         verbose,
#     )
#
#     hyperparams2 = {}
#     for key in hyperparams.keys():
#         best_value = model.__dict__[key]
#
#         if len(hyperparams[key]) == 1:
#             hyperparams2[key] = [best_value]
#         elif isinstance(best_value, float):
#             center = best_value
#             width = best_value / 3
#             hyperparams2[key] = np.linspace(
#                 center - width, center + width, 4, dtype=float
#             ).tolist()
#         elif isinstance(best_value, int):
#             # center = best_value
#             # width = best_value // 2
#             # hyperparams2[key] = np.linspace(
#             #     center - width, center + width, 3, dtype=int
#             # ).tolist()
#             hyperparams2[key] = [best_value]
#         else:
#             hyperparams2[key] = [best_value]
#
#     return grid_search(model_type, hyperparams2, X, y, k, score_metric, scale, verbose)
