from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class DPConfig:
    data_path: str = "outputs/datasets/card_multimodal"
    output_dir: str = "outputs/dp/card_wrist"
    camera: str = "wrist"
    physics_hz: int = 60
    policy_hz: int = 12
    observation_horizon: int = 2
    action_horizon: int = 16
    execution_horizon: int = 8
    action_dim: int = 9
    proprio_dim: int = 18
    image_height: int = 240
    image_width: int = 320
    image_resize: int = 224
    image_crop: int = 0
    image_imagenet_normalize: bool = True
    vision_feature_dim: int = 512
    vision_pretrained: bool = False
    unet_dims: tuple[int, ...] = (256, 512, 1024)
    diffusion_embed_dim: int = 256
    diffusion_train_timesteps: int = 100
    inference_steps: int = 100
    batch_size: int = 16
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-6
    epochs: int = 1000
    max_steps: int = 5000
    num_workers: int = 4
    seed: int = 0
    device: str = "cuda:0"
    ema_decay: float = 0.999
    log_every: int = 20
    checkpoint_every: int = 500
    resume: str = ""
    overfit_single_episode: bool = True
    split_level: str = "episode"
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1

    def __post_init__(self):
        if self.camera not in ("fixed", "wrist", "both"):
            raise ValueError("camera must be fixed, wrist, or both")
        if self.physics_hz != 60 or self.policy_hz != 12:
            raise ValueError("B1 requires explicit 60 Hz physics and 12 Hz policy time")
        if self.observation_horizon < 1 or self.action_horizon < 1:
            raise ValueError("horizons must be positive")
        if not 1 <= self.execution_horizon <= self.action_horizon:
            raise ValueError("execution_horizon must be in [1, action_horizon]")
        if self.action_dim != 9 or self.proprio_dim != 18:
            raise ValueError("B1 baseline is explicitly 9D action / 18D proprio")
        if self.vision_feature_dim != 512:
            raise ValueError("ResNet-18 encoder output is fixed at 512 features")
        if self.diffusion_train_timesteps < 2 or self.inference_steps < 1:
            raise ValueError("diffusion step counts are invalid")
        if self.epochs < 1 or self.max_steps < 1 or self.batch_size < 1:
            raise ValueError("training counts must be positive")
        self.unet_dims = tuple(int(value) for value in self.unet_dims)

    @property
    def camera_names(self):
        return ("fixed", "wrist") if self.camera == "both" else (self.camera,)

    def to_dict(self):
        result = asdict(self)
        result["unet_dims"] = list(self.unet_dims)
        return result


def load_config(path, overrides=None):
    values = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(values, dict):
        raise ValueError("DP config must be a YAML mapping")
    if overrides:
        values.update(overrides)
    allowed = {field.name for field in fields(DPConfig)}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unknown DP config fields: {sorted(unknown)}")
    return DPConfig(**values)
