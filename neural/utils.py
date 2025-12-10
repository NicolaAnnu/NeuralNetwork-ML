import json

from neural.network import Classifier, Regressor


def stats(
    net: Classifier | Regressor,
    hyperparams: dict,
    validation_score: float,
    train_score: float,
    test_score: float,
    filepath: None | str = None,
):
    print(f"validation score: {validation_score:.4f}")
    net_params = {k: net.__dict__[k] for k in hyperparams.keys()}
    for k in net_params.keys():
        print(f"{k}: {net_params[k]}")

    print(f"converged in {len(net.loss_curve)} epochs")
    print(f"loss: {net.loss:.4f}")

    # save results to a json file
    if filepath is not None:
        with open(filepath, "w") as fp:
            out = {
                "validation_score": validation_score,
                "training_score": train_score,
                "test_score": test_score,
                "parameters": net_params,
            }
            json.dump(out, fp, indent=4)