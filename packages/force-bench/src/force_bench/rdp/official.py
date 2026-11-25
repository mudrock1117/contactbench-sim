"""Loader for the locked official RDP source tree."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import yaml


UPSTREAM_NAME = "reactive_diffusion_policy"
UPSTREAM_COMMIT = "8ed82a3"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def activate_upstream(*, verify: bool = True) -> Path:
    root = repository_root()
    lock = yaml.safe_load((root / "upstreams.lock.yaml").read_text())
    spec = lock["upstreams"][UPSTREAM_NAME]
    if not str(spec["commit"]).startswith(UPSTREAM_COMMIT):
        raise RuntimeError("upstreams.lock.yaml does not pin RDP 8ed82a3")
    path = root / spec["path"]
    if not path.is_dir():
        raise FileNotFoundError(
            f"missing {path}; run scripts/fetch_upstreams.py --name {UPSTREAM_NAME}")
    if verify:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        wanted = subprocess.check_output(
            ["git", "rev-parse", f"{spec['commit']}^{{commit}}"],
            cwd=path, text=True).strip()
        if head != wanted:
            raise RuntimeError(f"RDP revision mismatch: {head} != {wanted}")
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path


def workspace_from_yaml(path: str | Path, *, expected_stage: str | None = None):
    """Instantiate an unmodified upstream AT or LDP workspace."""
    activate_upstream()
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    output_dir = Path(str(cfg.multi_run.run_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = get_class(cfg._target_)(cfg, output_dir=str(output_dir))
    allowed = {
        "at": "reactive_diffusion_policy.workspace.train_at_workspace",
        "ldp": "reactive_diffusion_policy.workspace.train_diffusion_unet_image_workspace",
    }
    module = workspace.__class__.__module__
    if expected_stage is not None and not module.startswith(allowed[expected_stage]):
        raise TypeError(f"configuration created {module}, expected official {expected_stage}")
    if not any(module.startswith(value) for value in allowed.values()):
        raise TypeError(f"configuration did not create an official RDP workspace: {module}")
    return workspace


def run_workspace(path: str | Path, *, stage: str):
    workspace = workspace_from_yaml(path, expected_stage=stage)
    workspace.run()
    return workspace
