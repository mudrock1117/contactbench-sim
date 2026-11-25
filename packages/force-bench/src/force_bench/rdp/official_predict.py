"""Offline prediction through the locked official RDP latent policy."""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from hydra.utils import instantiate
from .official import workspace_from_yaml


def _load(config, checkpoint, device):
    ws = workspace_from_yaml(config, expected_stage="ldp")
    ws.load_checkpoint(path=checkpoint, map_location=device)
    policy = (getattr(ws, "ema_model", None) or ws.model).to(device).eval()
    return ws, policy


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)
    ws, policy = _load(a.config, a.checkpoint, a.device)
    dataset = instantiate(ws.cfg.task.dataset)
    item = dataset[0]
    obs = {k: v.unsqueeze(0).to(a.device) for k, v in item["obs"].items()}
    extended = {
        k: v.unsqueeze(0).to(a.device) for k, v in item["extended_obs"].items()}
    ratio = int(ws.cfg.dataset_obs_temporal_downsample_ratio)
    with torch.inference_mode():
        result = policy.predict_action(
            obs, extended_obs_dict=extended,
            dataset_obs_temporal_downsample_ratio=ratio)
    action = result["action"][0].cpu().numpy()
    start = int(ws.cfg.n_obs_steps) * ratio - 1
    expert = item["action"][start:start + len(action)].numpy()
    print(json.dumps({
        "implementation": "xiaoxiaoxh/reactive_diffusion_policy@8ed82a3",
        "shape": list(action.shape),
        "finite": bool(np.isfinite(action).all()),
        "min": float(action.min()), "max": float(action.max()),
        "mae": float(np.abs(action - expert).mean()),
        "action_start_index": start,
        "fast_action_hz": 24, "slow_policy_hz": 12,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
