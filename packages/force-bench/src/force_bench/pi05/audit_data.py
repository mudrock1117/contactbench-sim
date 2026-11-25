"""Audit ContactBench pi0.5 windows without constructing the model."""
from __future__ import annotations

import argparse
import json
import numpy as np

from .config import DEFAULT_DATA
from .dataset import ContactBenchPi05Dataset, PROMPT


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args(argv)
    dataset = ContactBenchPi05Dataset(a.data)
    count = len(dataset) if a.limit <= 0 else min(len(dataset), a.limit)
    action_min = np.full(10, np.inf)
    action_max = np.full(10, -np.inf)
    max_rgb_age = 0.0
    for index in range(count):
        item = dataset[index]
        if item["wrist_rgb"].shape != (240, 320, 3):
            raise ValueError("invalid wrist image")
        if item["state"].shape != (10,) or item["actions"].shape != (32, 10):
            raise ValueError("invalid state/action contract")
        if float(item["rgb_timestamp"]) > float(item["timestamp"]) + 1e-12:
            raise ValueError("future image")
        if not np.isfinite(item["actions"]).all():
            raise ValueError("non-finite action")
        action_min = np.minimum(action_min, item["actions"].min(axis=0))
        action_max = np.maximum(action_max, item["actions"].max(axis=0))
        max_rgb_age = max(max_rgb_age, float(item["rgb_age_s"]))
    print(json.dumps({
        "windows": len(dataset), "audited": count,
        "wrist_rgb": [240, 320, 3], "state": [10],
        "actions": [32, 10], "prompt": PROMPT,
        "physics_hz": 60, "policy_hz": 24, "rgb_hz": 12,
        "max_rgb_age_s": max_rgb_age,
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "future_rgb": False,
        "privileged_input": False,
        "delta_actions_transform": False,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
