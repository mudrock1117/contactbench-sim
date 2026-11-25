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


def _detached(value, OmegaConf):
    """Copy a config subtree without inherited Hydra struct flags."""
    return OmegaConf.create(OmegaConf.to_container(value, resolve=False))


def _validate_contactbench_profile(cfg, upstream_config):
    from omegaconf import OmegaConf
    shape = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    expected_obs = (
        {"agent_pos": {"shape": [10], "type": "low_dim"},
         "tactile_marker_pca": {"shape": [30], "type": "low_dim"}}
        if upstream_config == "train_at_workspace" else
        {"wrist_rgb": {"shape": [3, 240, 320], "type": "rgb"},
         "agent_pos": {"shape": [10], "type": "low_dim"},
         "tactile_marker_pca": {"shape": [30], "type": "low_dim"}})
    if shape["obs"] != expected_obs:
        raise ValueError(f"unexpected ContactBench RDP observation contract: {shape['obs']}")
    if shape.get("extended_obs") != {
            "tactile_marker_pca": {"shape": [30], "type": "low_dim"}}:
        raise ValueError("RDP extended observation must be the 30D dual-pad PCA")
    if shape.get("action") != {"shape": [10]}:
        raise ValueError("RDP action contract must be 10D")
    if OmegaConf.to_container(cfg.task.shape_meta, resolve=True) != shape:
        raise ValueError("task and top-level shape_meta differ")
    if OmegaConf.to_container(cfg.policy.shape_meta, resolve=True) != shape:
        raise ValueError("policy and top-level shape_meta differ")
    dataset = cfg.task.dataset
    if dataset._target_ != (
            "force_bench.rdp.official_dataset.ContactBenchOfficialRDPDataset"):
        raise ValueError("profile did not replace the upstream dataset")
    allowed = {
        "_target_", "data_path", "pca_transform_path", "pca_mean_path",
        "pca_mode", "pca_artifact_path", "observation_horizon", "horizon",
        "obs_temporal_downsample_ratio", "success_only", "seed", "val_ratio",
        "at", "use_latent_action_before_vq"}
    leftovers = set(dataset.keys()) - allowed
    if leftovers:
        raise ValueError(f"upstream dataset fields survived replacement: {sorted(leftovers)}")
    if int(dataset.observation_horizon) != 2 or int(dataset.horizon) != 32:
        raise ValueError("RDP dataset horizons must be obs=2, action=32")
    if int(dataset.obs_temporal_downsample_ratio) != 2:
        raise ValueError("RDP dataset must downsample 24 Hz observations to 12 Hz")
    if int(cfg.horizon) != 32 or int(cfg.n_obs_steps) != 2:
        raise ValueError("official RDP model horizons differ from ContactBench")
    if "env_runner" in cfg.task:
        raise ValueError("real-robot env_runner survived ContactBench task replacement")


def compose_profile_config(path: str | Path):
    """Compose upstream defaults, then replace ContactBench contract subtrees."""
    upstream_root = activate_upstream()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # Both pinned workspaces register this resolver only when their Python
    # modules are imported. Composition happens earlier, so register it here.
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    profile = OmegaConf.load(path)
    if "upstream_config" not in profile:
        cfg = _detached(profile, OmegaConf)
        OmegaConf.resolve(cfg)
        return cfg
    upstream_config = str(profile.upstream_config)
    with initialize_config_dir(
            version_base=None,
            config_dir=str(upstream_root / "reactive_diffusion_policy" / "config")):
        composed = compose(config_name=upstream_config)
    # Recreate both trees to remove Hydra's inherited struct restrictions.
    cfg = _detached(composed, OmegaConf)
    overlay = _detached(profile.get("config", {}), OmegaConf)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(overlay, False)
    cfg = OmegaConf.merge(cfg, overlay)

    # Recursive merge is correct for official model/training overrides, but
    # these schemas describe a different task and must never retain old keys.
    for key in ("shape_meta", "task"):
        if key in overlay:
            cfg[key] = _detached(overlay[key], OmegaConf)
    if "policy" in overlay and "shape_meta" in overlay.policy:
        cfg.policy.shape_meta = _detached(overlay.policy.shape_meta, OmegaConf)

    OmegaConf.resolve(cfg)
    _validate_contactbench_profile(cfg, upstream_config)
    return cfg


def workspace_from_yaml(path: str | Path, *, expected_stage: str | None = None):
    """Instantiate an unmodified upstream AT or LDP workspace."""
    activate_upstream()
    from hydra.utils import get_class
    cfg = compose_profile_config(path)
    dataset_cfg = cfg.task.dataset
    if "pca_mode" in dataset_cfg:
        from .tactile import pca_basis_fingerprint
        cfg.rdp_pca_fingerprint = pca_basis_fingerprint(
            mode=str(dataset_cfg.pca_mode),
            transform_path=dataset_cfg.get("pca_transform_path"),
            mean_path=dataset_cfg.get("pca_mean_path"),
            artifact_path=dataset_cfg.get("pca_artifact_path"))
        at_checkpoint = cfg.get("at_load_dir")
        if at_checkpoint:
            import torch
            payload = torch.load(str(at_checkpoint), map_location="cpu")
            saved_cfg = payload.get("cfg") if isinstance(payload, dict) else None
            saved_fingerprint = (saved_cfg.get("rdp_pca_fingerprint")
                                 if saved_cfg is not None else None)
            if saved_fingerprint != cfg.rdp_pca_fingerprint:
                raise ValueError(
                    "AT checkpoint PCA fingerprint differs from LDP configuration")
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
