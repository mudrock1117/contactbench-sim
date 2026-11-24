"""Thin loader for real-stanford/diffusion_policy@5ba07ac.

No encoder, U-Net, scheduler, normalizer, or workspace is reimplemented here.
The previous force_bench.dp.model implementation remains available as legacy.
"""
from __future__ import annotations
from pathlib import Path
import subprocess
import sys
import yaml

def repository_root():
    return Path(__file__).resolve().parents[5]

def activate_upstream(*, verify=True):
    root = repository_root()
    lock = yaml.safe_load((root / "upstreams.lock.yaml").read_text())
    spec = lock["upstreams"]["diffusion_policy"]
    path = root / spec["path"]
    if not path.is_dir():
        raise FileNotFoundError(f"missing {path}; run scripts/fetch_upstreams.py --name diffusion_policy")
    if verify:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        wanted = subprocess.check_output(
            ["git", "rev-parse", f"{spec['commit']}^{{commit}}"], cwd=path, text=True).strip()
        if head != wanted:
            raise RuntimeError(f"diffusion_policy revision mismatch: {head} != {wanted}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path

def workspace_from_yaml(path):
    """Instantiate the pinned upstream TrainDiffusionUnetHybridWorkspace."""
    activate_upstream()
    from omegaconf import OmegaConf
    from hydra.utils import get_class
    cfg = OmegaConf.load(path)
    output_dir = Path(OmegaConf.to_container(cfg.multi_run, resolve=True)["run_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    expected = "diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace"
    if not workspace.__class__.__module__.startswith(expected):
        raise TypeError(f"configuration did not create the official workspace: {workspace!r}")
    return workspace

def run_workspace(path):
    workspace = workspace_from_yaml(path)
    workspace.run()
    return workspace


def validate_checkpoint_contract(cfg, payload):
    """Reject a checkpoint whose model/time contract differs from runtime YAML."""
    saved = payload.get("cfg")
    if saved is None:
        raise ValueError("official checkpoint has no saved cfg")
    fields = ("horizon", "n_obs_steps", "n_action_steps")
    for field in fields:
        if int(saved[field]) != int(cfg[field]):
            raise ValueError(f"checkpoint {field}={saved[field]} but config requests {cfg[field]}")
    if list(saved.shape_meta.action.shape) != list(cfg.shape_meta.action.shape):
        raise ValueError("checkpoint action shape differs from config")
    from omegaconf import OmegaConf
    saved_obs = OmegaConf.to_container(saved.shape_meta.obs, resolve=True)
    runtime_obs = OmegaConf.to_container(cfg.shape_meta.obs, resolve=True)
    if saved_obs != runtime_obs:
        raise ValueError("checkpoint observation shapes differ from config")
