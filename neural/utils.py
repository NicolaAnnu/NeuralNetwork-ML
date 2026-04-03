import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from dask.delayed import delayed
from dask.distributed import Client, progress


def dump_results(filepath: str, results: list[dict]):
    path = Path(filepath)
    if not path.exists():
        with open(path, "w+") as fp:
            json.dump(results, fp, indent=2)
    else:
        with open(path, "r+") as fp:
            data = json.load(fp)
            data.extend(results)
            fp.seek(0)
            json.dump(data, fp, indent=2)
            fp.truncate()


def load_results(filepath: str) -> list[dict]:
    with open(filepath, "r") as fp:
        return json.load(fp)


def retrain(model, params, X_train, y_train, X_test, y_test, metric, n, address):
    def fit(net, X_train, y_train, metric, X_test, y_test):
        net.fit(X_train, y_train, metric, X_test, y_test)
        return net

    # dask init
    if address:
        client = Client(address)
    else:
        client = Client()

    print(f"dask dashboard: {client.dashboard_link}")

    nets = [model(**params) for _ in range(n)]
    tasks = [
        delayed(fit)(net, X_train, y_train, metric, X_test, y_test) for net in nets
    ]

    # perform parallel k-folds
    futures = client.compute(tasks)
    progress(futures)
    results = client.gather(futures)
    client.close()

    return results


def plot_curve(loss_curves, label):
    max_len = max(len(curve) for curve in loss_curves)
    loss_matrix = np.full((len(loss_curves), max_len), np.nan)
    for i, curve in enumerate(loss_curves):
        loss_matrix[i, : len(curve)] = curve

    mean_loss = np.nanmean(loss_matrix, axis=0)
    std_loss = np.nanstd(loss_matrix, axis=0)
    epochs = np.arange(len(mean_loss))

    plt.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss, alpha=0.25)
    plt.plot(epochs, mean_loss, label=label)


def target_plot(y_true, y_pred):
    _, axes = plt.subplots(2, 2, figsize=(8, 5), dpi=150, constrained_layout=True)

    for i, ax in enumerate(axes.flat):
        ax.scatter(y_pred[:, i], y_true[:, i], alpha=0.6)
        x = np.linspace(y_pred[:, i].min() - 0.5, y_pred[:, i].max() + 0.5, 5)
        y = x
        ax.plot(x, y, "g--", label="Ideal")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Target", fontsize=10)
        ax.set_title(f"Output {i + 1}", fontsize=10)
        ax.legend()

    plt.show()