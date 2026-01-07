import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neural.metrics import mean_euclidean_error
from neural.network import Regressor
from neural.utils import load_results

if __name__ == "__main__":
    # set headers
    names = ["ID"]
    features = [f"feature{i}" for i in range(12)]
    targets = [f"target{i}" for i in range(4)]
    names.extend(features)
    names.extend(targets)

    train = pd.read_csv(
        "datasets/ml_cup_train.csv",
        header=None,
        names=names,
        skiprows=7,
    )

    # get feature and target columns
    X = train.iloc[:, 1:13].to_numpy()
    y = train.iloc[:, 13:].to_numpy()

    results = load_results("results/cup.json")
    results = [r for r in results if r["loss"] != np.inf]

    best = sorted(results, key=lambda x: x["score"])[0]

    # re-train the model
    params = best["parameters"]
    print(json.dumps(params, indent=2))

    X_scaler = StandardScaler()
    X = X_scaler.fit_transform(X)

    y_scaler = StandardScaler()
    y = y_scaler.fit_transform(y)

    net = Regressor(**params)
    net.fit(X, y, mean_euclidean_error)

    train = pd.read_csv(
        "datasets/ml_cup_blind_test.csv",
        header=None,
        names=names,
        skiprows=7,
    )

    # get feature and target columns
    X_blind = train.iloc[:, 1:13].to_numpy()

    X_blind = X_scaler.transform(X_blind)
    y_pred = net.predict(X_blind)
    y_pred = y_scaler.inverse_transform(y_pred)

    df = pd.DataFrame(y_pred)
    df.to_csv("results/blind_cup.csv", index=True, header=False)
