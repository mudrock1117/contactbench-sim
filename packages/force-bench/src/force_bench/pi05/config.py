"""Official OpenPI pi0.5 ContactBench train configurations."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from .dataset import (
    ContactBenchInputs, ContactBenchOutputs, ContactBenchPi05Dataset,
    PROMPT, ACTION_HORIZON, MODEL_ACTION_DIM)
from .official import activate_openpi

DEFAULT_MODEL_ROOT = Path("/home/zdh/contactbench_models/pi05")
DEFAULT_DATA = Path("outputs/datasets/card_multimodal_a4")
ASSET_ID = "contactbench_card"


activate_openpi()
from openpi import transforms
from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import config as openpi_config
from openpi.training import optimizer
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class ContactBenchDataFactory:
    data_path: str
    assets_dir: str
    prompt: str = PROMPT

    def create(self, assets_dirs, model_config):
        stats = None
        path = Path(self.assets_dir) / ASSET_ID
        try:
            stats = normalize.load(path)
        except FileNotFoundError:
            pass
        return openpi_config.DataConfig(
            repo_id="contactbench/card_multimodal_a4",
            asset_id=ASSET_ID,
            norm_stats=stats,
            data_transforms=transforms.Group(
                inputs=[ContactBenchInputs(model_config.model_type)],
                outputs=[ContactBenchOutputs()]),
            model_transforms=openpi_config.ModelTransformFactory(
                default_prompt=self.prompt)(model_config),
            use_quantile_norm=True,
            action_sequence_keys=("actions",),
            prompt_from_task=False)


def make_config(profile, *, exp_name, data_path=DEFAULT_DATA,
                model_root=DEFAULT_MODEL_ROOT, resume=False, overwrite=False):
    if profile not in ("smoke", "smoke_no_ema", "smoke_lora", "paper"):
        raise ValueError(
            "profile must be smoke, smoke_no_ema, smoke_lora, or paper")
    root = Path(model_root)
    smoke_like = profile != "paper"
    lora = profile == "smoke_lora"
    no_ema = profile in ("smoke_no_ema", "smoke_lora")
    model_config = pi0_config.Pi0Config(
        pi05=True, action_dim=MODEL_ACTION_DIM,
        action_horizon=ACTION_HORIZON,
        paligemma_variant="gemma_2b_lora" if lora else "gemma_2b",
        action_expert_variant="gemma_300m_lora" if lora else "gemma_300m")
    return openpi_config.TrainConfig(
        name=f"contactbench_pi05_{profile}",
        project_name="contactbench-pi05",
        exp_name=str(exp_name),
        model=model_config,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(root / "pretrained" / "pi05_base" / "params")),
        data=ContactBenchDataFactory(
            data_path=str(data_path), assets_dir=str(root / "assets")),
        assets_base_dir=str(root / "assets"),
        checkpoint_base_dir=str(root / "checkpoints"),
        seed=0,
        batch_size=1 if smoke_like else 64,
        num_workers=0 if smoke_like else 4,
        num_train_steps=10 if smoke_like else 20_000,
        log_interval=1 if smoke_like else 100,
        save_interval=10 if smoke_like else 1_000,
        keep_period=None if smoke_like else 5_000,
        overwrite=bool(overwrite),
        resume=bool(resume),
        wandb_enabled=not smoke_like,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1 if smoke_like else 500,
            peak_lr=5e-5,
            decay_steps=10 if smoke_like else 20_000,
            decay_lr=5e-6),
        optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=model_config.get_freeze_filter(),
        ema_decay=None if no_ema else 0.999,
        fsdp_devices=1,
        policy_metadata={
            "task": "retrieve_card_from_deck",
            "prompt": PROMPT,
            "policy_hz": 24,
            "action_semantics": "fixed_chunk_anchor_relative_eef_10d",
            "physical_action_dim": 10,
            "model_action_dim": 32,
        })


def install_dataset_factory(config):
    """Inject only the raw dataset; retain official transforms and loaders."""
    from openpi.training import data_loader
    data_path = config.data.data_path
    prompt = config.data.prompt

    def create_torch_dataset(data_config, action_horizon, model_config):
        if int(action_horizon) != ACTION_HORIZON:
            raise ValueError("ContactBench pi0.5 requires action horizon 32")
        return ContactBenchPi05Dataset(data_path, prompt=prompt)

    data_loader.create_torch_dataset = create_torch_dataset
    return config
