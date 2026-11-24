from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import load_config
from .data import (TorchWindowDataset, auxiliary_action_audit, build_dataset,
                   compute_statistics, episode_splits)
from .model import DiffusionPolicy, EMA


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train ContactBench B1 Diffusion Policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--device")
    parser.add_argument("--resume")
    args = parser.parse_args(argv)
    overrides = {key: value for key, value in {
        "data_path": args.data, "output_dir": args.output,
        "device": args.device, "resume": args.resume}.items() if value is not None}
    config = load_config(args.config, overrides)
    random.seed(config.seed); np.random.seed(config.seed); torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    splits = episode_splits(config)
    train_numpy = build_dataset(splits["train"], config, as_torch=False)
    statistics = compute_statistics(train_numpy)
    audit = auxiliary_action_audit(splits["train"])
    (output / "run_config.json").write_text(json.dumps({
        "config": config.to_dict(), "windows": len(train_numpy),
        "splits": {key: [str(path) for path in value] for key, value in splits.items()},
        "auxiliary_action_audit": audit}, indent=2))
    dataset = TorchWindowDataset(build_dataset(splits["train"], config, as_torch=True),
                                 config.camera_names)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers, pin_memory=True,
                        generator=generator, drop_last=False)
    device = torch.device(config.device)
    model = DiffusionPolicy(config).to(device); model.set_normalization(statistics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    ema = EMA(model, config.ema_decay)
    step = start_epoch = 0
    if config.resume:
        state = torch.load(config.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); ema.shadow.load_state_dict(state["ema"])
        optimizer.load_state_dict(state["optimizer"])
        step, start_epoch = int(state["step"]), int(state["epoch"])
    log_path = output / "train.jsonl"
    model.train()
    for epoch in range(start_epoch, config.epochs):
        for batch in loader:
            proprio = batch["proprio"].to(device, non_blocking=True).float()
            actions = batch["actions"].to(device, non_blocking=True).float()
            images = {name: value.to(device, non_blocking=True)
                      for name, value in batch["images"].items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(proprio, images, actions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); ema.update(model); step += 1
            if step % config.log_every == 0:
                record = {"step": step, "epoch": epoch, "loss": float(loss.detach())}
                print(json.dumps(record), flush=True)
                with log_path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            if step % config.checkpoint_every == 0:
                save_checkpoint(output / f"checkpoint_{step:08d}.pt", model, ema,
                                optimizer, config, step, epoch)
                save_checkpoint(output / "latest.pt", model, ema, optimizer,
                                config, step, epoch)
            if step >= config.max_steps:
                break
        if step >= config.max_steps:
            break
    save_checkpoint(output / "latest.pt", model, ema, optimizer, config, step, epoch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
