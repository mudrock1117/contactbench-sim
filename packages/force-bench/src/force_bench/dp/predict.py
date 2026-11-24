from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from .checkpoint import load_policy
from .data import build_dataset, episode_splits


def sample_tensors(sample, config, device):
    proprio = torch.as_tensor(sample["proprio"], device=device).float().unsqueeze(0)
    if len(config.camera_names) == 1:
        images = {config.camera_names[0]: torch.as_tensor(
            sample["rgb"], device=device).unsqueeze(0)}
    else:
        images = {name: torch.as_tensor(sample["cameras"][name]["rgb"],
                                        device=device).unsqueeze(0)
                  for name in config.camera_names}
    return proprio, images


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline Diffusion Policy prediction check")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    model, config, _ = load_policy(args.checkpoint, args.device)
    if args.data:
        config.data_path = args.data
    paths = episode_splits(config)["train"]
    dataset = build_dataset(paths, config)
    sample = dataset[args.index]
    proprio, images = sample_tensors(sample, config, args.device)
    with torch.inference_mode():
        prediction = model.predict_actions(proprio, images)[0].cpu().numpy()
    expert = np.asarray(sample["actions"])
    report = {"shape": list(prediction.shape),
              "finite": bool(np.isfinite(prediction).all()),
              "predicted_min": float(prediction.min()),
              "predicted_max": float(prediction.max()),
              "expert_min": float(expert.min()), "expert_max": float(expert.max()),
              "mae": float(np.abs(prediction - expert).mean()),
              "rmse": float(np.sqrt(np.square(prediction - expert).mean())),
              "action_space": "9D joint position target (7 rad + 2 m), not paper 10D EEF"}
    print(json.dumps(report, indent=2))
    return 0 if report["finite"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
