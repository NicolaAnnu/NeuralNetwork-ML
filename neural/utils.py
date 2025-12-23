import json
from pathlib import Path

import matplotlib.pyplot as plt


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


def target_plot(y_true, y_pred):
    _, axes = plt.subplots(2, 2, figsize=(8, 5), dpi=150, constrained_layout=True)

    for i, ax in enumerate(axes.flat):
        ax.scatter(y_pred[:, i], y_true[:, i], alpha=0.6)
        ax.set_xlabel("predicted", fontsize=10)
        ax.set_ylabel("target", fontsize=10)
        ax.set_title(f"output {i + 1}", fontsize=10)

    plt.show()
