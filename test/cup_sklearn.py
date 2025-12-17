import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RESULTS_PATH = Path("results/sklearn_cup.json")
PREDICTIONS_PATH = Path("results/sklearn_cup_predictions.csv")

# Hyperparameter grid for optional grid search on the CUP regression task.
PARAM_GRID = {
    "hidden_layer_sizes": [(32,), (64,), (64, 32)],
    "activation": ["tanh"],
    "output_activation": ["linear"],
    "learning_rate": [0.001, 0.01],
    "lam": [0.0, 0.0001],
    "alpha": [0.9],
    "tol": [1e-5],
    "batch_size": [32, 64],
    "max_iter": [1000, 2000, 3000],  # più epoche possibili
    "n_iter_no_change": [200],  # evita early stop precoce
}


PARAM_KEY_MAP = {
    "hidden_layer_sizes": "model__hidden_layer_sizes",
    "activation": "model__activation",
    "learning_rate": "model__learning_rate_init",
    "lam": "model__alpha",  # L2 regularization
    "alpha": "model__momentum",  # momentum
    "tol": "model__tol",
    "batch_size": "model__batch_size",
    "max_iter": "model__max_iter",
    "n_iter_no_change": "model__n_iter_no_change",
    "shuffle": "model__shuffle",
    "output_activation": None,
}


def normalize_params(params: dict) -> dict:
    """Map custom keys to pipeline namespaced keys (model__*, scaler__*)."""
    normalized = {}
    for key, value in params.items():
        if key.startswith("model__") or key.startswith("scaler__"):
            normalized[key] = value
            continue

        mapped = PARAM_KEY_MAP.get(key)
        if mapped is None:
            continue
        if mapped:
            normalized[mapped] = value
    return normalized


def load_cup_data():
    """Load CUP data: 12 features + 4 targets for train, only features for test."""
    train = pd.read_csv("datasets/ml_cup_train.csv", header=None)
    test = pd.read_csv("datasets/ml_cup_test.csv", header=None)

    X_train = train.iloc[:, 1:13].to_numpy()
    y_train = train.iloc[:, 13:].to_numpy()

    X_test = test.iloc[:, 1:].to_numpy()
    test_ids = test.iloc[:, 0].to_numpy(dtype=int)

    return X_train, y_train, X_test, test_ids


def build_pipeline(params: Optional[dict] = None) -> Pipeline:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    solver="sgd",
                    learning_rate="constant",
                    random_state=42,
                    shuffle=False,  # per coerenza con la rete custom (default False)
                ),
            ),
        ]
    )
    if params is not None:
        pipeline.set_params(**normalize_params(params))
    return pipeline


def load_params_from_file() -> dict:
    if RESULTS_PATH.exists() and RESULTS_PATH.stat().st_size:
        try:
            with RESULTS_PATH.open() as fp:
                data = json.load(fp)
            params = data.get("parameters")
            if isinstance(params, dict):
                return normalize_params(params)
        except json.JSONDecodeError:
            pass
    raise FileNotFoundError(
        "results/sklearn_cup.json not found or invalid. Run with --gs and --save first."
    )


def save_results(params: dict, cv_rmse: Optional[float], train_rmse: float) -> None:
    with RESULTS_PATH.open("w") as fp:
        json.dump(
            {
                "cv_rmse": cv_rmse,
                "train_rmse": train_rmse,
                "parameters": normalize_params(params),
            },
            fp,
            indent=4,
        )


def plot_loss_curve(model: Pipeline) -> None:
    mlp = model.named_steps["model"]
    if not hasattr(mlp, "loss_curve_"):
        return
    plt.title("CUP Loss Curve (sklearn MLPRegressor)")
    plt.plot(mlp.loss_curve_, label="training loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gs",
        action="store_true",
        help="run a grid search on CUP regression hyperparameters",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save params/metrics to JSON and predictions to CSV",
    )
    args = parser.parse_args()

    X_train, y_train, X_test, test_ids = load_cup_data()

    if args.gs:
        cv = KFold(n_splits=10, shuffle=True, random_state=42)
        grid = GridSearchCV(
            estimator=build_pipeline(),
            param_grid=normalize_params(PARAM_GRID),
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
            verbose=2,
            return_train_score=False,
            refit=True,
        )
        grid.fit(X_train, y_train)

        model = grid.best_estimator_
        params = grid.best_params_
        cv_rmse = (-grid.best_score_) ** 0.5

        print(f"best CV RMSE: {cv_rmse:.4f}")
        print(f"best params: {params}")
    else:
        params = load_params_from_file()
        model = build_pipeline(params)
        model.fit(X_train, y_train)
        cv_rmse = None

    train_pred = model.predict(X_train)
    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    print(f"train RMSE: {train_rmse:.4f}")

    test_pred = model.predict(X_test)
    test_df = pd.DataFrame(
        test_pred, columns=[f"y{i+1}" for i in range(test_pred.shape[1])]
    )
    test_df.insert(0, "id", test_ids)

    if args.save:
        save_results(params, cv_rmse, train_rmse)
        test_df.to_csv(PREDICTIONS_PATH, index=False)
        print(f"saved results to {RESULTS_PATH} and predictions to {PREDICTIONS_PATH}")

    plot_loss_curve(model)
