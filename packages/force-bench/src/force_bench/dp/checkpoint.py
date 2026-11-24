from __future__ import annotations

from pathlib import Path

import torch

from .config import DPConfig
from .model import DiffusionPolicy


def save_checkpoint(path, model, ema, optimizer, config, step, epoch):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": 1, "model": model.state_dict(),
                "ema": ema.shadow.state_dict(), "optimizer": optimizer.state_dict(),
                "normalization": model.normalization_state(),
                "config": config.to_dict(), "step": int(step), "epoch": int(epoch)},
               temporary)
    temporary.replace(path)


def load_policy(path, device="cpu", *, use_ema=True):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = DPConfig(**checkpoint["config"])
    model = DiffusionPolicy(config).to(device)
    model.load_state_dict(checkpoint["ema" if use_ema else "model"])
    model.eval()
    return model, config, checkpoint
