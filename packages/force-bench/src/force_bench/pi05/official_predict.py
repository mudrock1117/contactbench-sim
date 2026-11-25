"""Offline prediction with the locked official OpenPI pi0.5 policy."""
from __future__ import annotations

import argparse
import json
import numpy as np

from .config import DEFAULT_DATA, DEFAULT_MODEL_ROOT
from .dataset import ContactBenchPi05Dataset
from .inference import load_policy


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("smoke", "smoke_no_ema", "smoke_lora", "paper"), default="smoke")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    _, policy = load_policy(
        a.profile, a.checkpoint, device=a.device,
        data_path=a.data, model_root=a.model_root)
    item = ContactBenchPi05Dataset(a.data)[0]
    result = policy.infer({
        "wrist_rgb": item["wrist_rgb"],
        "state": item["state"],
        "prompt": item["prompt"],
    })
    actions = np.asarray(result["actions"], np.float32)
    if actions.shape != (32, 10) or not np.isfinite(actions).all():
        raise RuntimeError(f"invalid pi0.5 action output {actions.shape}")
    expert = np.asarray(item["actions"], np.float32)
    print(json.dumps({
        "implementation": "Physical-Intelligence/openpi@15a9616",
        "model": "pi05_base_finetune",
        "shape": list(actions.shape),
        "finite": True,
        "min": float(actions.min()), "max": float(actions.max()),
        "mae": float(np.abs(actions - expert).mean()),
        "policy_hz": 24, "physics_hz": 60,
        "rgb_timestamp": float(item["rgb_timestamp"]),
        "rgb_age_s": float(item["rgb_age_s"]),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
