"""Offline prediction through the locked official RDP latent policy."""
from __future__ import annotations
import argparse
import json
import numpy as np
import torch
from hydra.utils import instantiate
from .checkpoint import restore_workspace_normalizers
from .official import workspace_from_yaml


def _load(config, checkpoint, device):
    ws = workspace_from_yaml(config, expected_stage="ldp")
    payload = ws.load_checkpoint(path=checkpoint, map_location=device)
    saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
    saved_fingerprint = (saved_cfg.get("rdp_pca_fingerprint")
                         if saved_cfg is not None else None)
    if saved_fingerprint != ws.cfg.rdp_pca_fingerprint:
        raise ValueError(
            "checkpoint PCA fingerprint differs from the configured artifact")
    # Upstream VAE.state_dict() omits its normalizer.  Restore the nested AT
    # from each serialized parent LDP normalizer before selecting model/EMA.
    restore_workspace_normalizers(ws)
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
    start = int(ws.cfg.n_obs_steps) * ratio - 1
    n_action = int(ws.cfg.n_action_steps)
    # Plan the slow latent once, then reproduce online causal AT decoding:
    # action k sees marker samples only through its own 24 Hz tick.
    with torch.inference_mode():
        planned = policy.predict_action(
            obs, dataset_obs_temporal_downsample_ratio=ratio,
            return_latent_action=True)
        latent = planned["action_pred"][:, 0]
        decoded = []
        for offset in range(n_action):
            available = start + 1 + offset
            causal = {key: value[:, :available] for key, value in extended.items()}
            result = policy.predict_from_latent_action(
                latent, extended_obs_dict=causal,
                extended_obs_last_step=available,
                dataset_obs_temporal_downsample_ratio=ratio,
                extend_obs_pad_after=True)
            decoded.append(result["action"][:, offset])
        action = torch.stack(decoded, dim=1)[0].cpu().numpy()
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
