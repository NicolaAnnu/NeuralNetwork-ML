from itertools import product
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
from neural.network import Classifier, Regressor
    
def kfold(n, k):
      if k <= 0:
          raise ValueError("error")
      if k > n:
          raise ValueError("error")
      base, extra = divmod(n, k)
      ranges, start = [], 0
      for _ in range(k):
          end = start + base + (1 if 0 < extra else 0)
          ranges.append((start, end))
          start = end
      return ranges

def train_and_score(params, model_type, X, y, k, score_metric):
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    current = 0
    scores = []

    for fold_size in fold_sizes:
        start, end = current, current + fold_size
        val_idx = indices[start:end]
        train_idx = np.concatenate([indices[:start], indices[end:]])

        model = model_type(**params)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[val_idx])
        scores.append(score_metric(y[val_idx], preds))

        current = end

    return float(np.mean(scores))

def _score_params(args):
      params, model_type, X, y, k, score_metric = args
      return train_and_score(params, model_type, X, y, k, score_metric), params

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
      params = [{k: v for k, v in zip(keys, comb)} for comb in combinations]

      n_cpus = mp.cpu_count()
      tasks = [(p, model_type, X, y, k, score_metric) for p in params]

      scores_params = []
      with mp.Pool(processes=n_cpus) as pool:
          for score, param in tqdm(
              pool.imap_unordered(_score_params, tasks),
              total=len(tasks),
              desc="grid search",
              ncols=80,
          ):
              scores_params.append((score, param))

      best_score, best_params = max(scores_params, key=lambda x: x[0])
      model = model_type(**best_params)
      model.fit(X, y)
      return model, best_score