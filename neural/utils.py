import json
from pathlib import Path


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
