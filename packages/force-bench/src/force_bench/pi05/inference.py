"""Shared official pi0.5 checkpoint loading for offline and socket inference."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .config import ASSET_ID, DEFAULT_DATA, DEFAULT_MODEL_ROOT, make_config
from .official import activate_openpi


def _validate_stats(checkpoint):
    activate_openpi()
    from openpi.shared import normalize
    stats = normalize.load(Path(checkpoint) / "assets" / ASSET_ID)
    expected = {"state": 10, "actions": 10}
    for key, width in expected.items():
        if key not in stats:
            raise ValueError(f"checkpoint normalization is missing {key}")
        value = stats[key]
        for name in ("mean", "std", "q01", "q99"):
            array = np.asarray(getattr(value, name))
            if array.shape != (width,) or not np.isfinite(array).all():
                raise ValueError(
                    f"invalid {key} {name}: shape={array.shape}, finite={np.isfinite(array).all()}")
    return stats


def load_policy(config_profile, checkpoint, *, device=None,
                data_path=DEFAULT_DATA, model_root=DEFAULT_MODEL_ROOT):
    root = activate_openpi()
    from openpi.policies import policy_config
    profile = str(config_profile)
    config = make_config(
        profile, exp_name="inference", data_path=data_path,
        model_root=model_root)
    stats = _validate_stats(checkpoint)
    policy = policy_config.create_trained_policy(
        config, checkpoint,
        default_prompt=config.data.prompt,
        norm_stats=stats,
        pytorch_device=device)
    return config, policy
