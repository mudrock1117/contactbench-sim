"""Compute official quantile statistics from ContactBench training episodes."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .config import ASSET_ID, DEFAULT_DATA, DEFAULT_MODEL_ROOT
from .dataset import ContactBenchInputs, ContactBenchPi05Dataset
from .official import activate_openpi


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--output", default=str(DEFAULT_MODEL_ROOT / "assets"))
    p.add_argument("--prompt", default="pick up the thin object on the table")
    a = p.parse_args(argv)
    activate_openpi()
    from openpi.models import model
    from openpi.shared import normalize
    transform = ContactBenchInputs(model.ModelType.PI05)
    dataset = ContactBenchPi05Dataset(a.data, prompt=a.prompt)
    stats = {"state": normalize.RunningStats(),
             "actions": normalize.RunningStats()}
    for index in range(len(dataset)):
        item = transform(dataset[index])
        stats["state"].update(np.asarray(item["state"], np.float32))
        stats["actions"].update(np.asarray(item["actions"], np.float32))
    result = {key: value.get_statistics() for key, value in stats.items()}
    output = Path(a.output) / ASSET_ID
    normalize.save(output, result)
    print({
        "output": str(output / "norm_stats.json"),
        "windows": len(dataset), "state_dim": 10,
        "action_shape": [32, 10], "fit_split": "train_episodes_only"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
