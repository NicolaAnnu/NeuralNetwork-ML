import json

from neural.network import Classifier, Regressor


def stats(
    net: Classifier | Regressor,
    hyperparams: dict,
    train_score: float,
    test_score: float,
):
    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.4f}")

    print(f"train score: {train_score:.4f}")
    print(f"test score: {test_score:.4f}")

    net_params = {k: net.__dict__[k] for k in hyperparams.keys()}
    print(f"{json.dumps(net_params, indent=4)}")


def save_stats(
    net: Classifier | Regressor,
    hyperparams: dict,
    validation_score: float,
    filepath: None | str = None,
):
    net_params = {k: net.__dict__[k] for k in hyperparams.keys()}

    # save results to a json file
    if filepath is not None:
        with open(filepath, "w") as fp:
            out = {
                "validation_score": validation_score,
                "parameters": net_params,
            }
            json.dump(out, fp, indent=4)
