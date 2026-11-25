"""CPU/static contracts for the locked official OpenPI pi0.5 adapter."""
from pathlib import Path
import json
import numpy as np
import pytest


def _upstream_available():
    from force_bench.pi05.official import repository_root
    return (repository_root() / "third_party" / "openpi").is_dir()


def test_pi05_lock_is_exact():
    import yaml
    from force_bench.pi05 import OPENPI_COMMIT
    from force_bench.pi05.official import repository_root
    lock = yaml.safe_load((repository_root() / "upstreams.lock.yaml").read_text())
    assert lock["upstreams"]["openpi"]["commit"] == OPENPI_COMMIT


@pytest.mark.skipif(not _upstream_available(), reason="locked OpenPI checkout required")
def test_pi05_wrist_input_masks_and_official_padding():
    from force_bench.pi05.dataset import (
        ContactBenchInputs, ContactBenchOutputs, PROMPT)
    from force_bench.pi05.official import activate_openpi
    activate_openpi()
    from openpi.models import model
    from openpi.transforms import PadStatesAndActions

    raw = {
        "wrist_rgb": np.zeros((240, 320, 3), np.uint8),
        "state": np.array([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.04], np.float32),
        "actions": np.zeros((32, 10), np.float32),
        "prompt": PROMPT,
    }
    value = ContactBenchInputs(model.ModelType.PI05)(raw)
    assert value["image"]["left_wrist_0_rgb"].shape == (240, 320, 3)
    assert value["image_mask"] == {
        "base_0_rgb": np.False_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    padded = PadStatesAndActions(32)(value)
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (32, 32)
    output = ContactBenchOutputs()({"actions": padded["actions"]})
    assert output["actions"].shape == (32, 10)


def test_pi05_24hz_schedule_and_fixed_anchor_contract():
    from force_bench.data.timebase import PolicyFrameScheduler, PolicyTimeConfig
    from force_bench.robot.eef_action import encode_relative_eef
    schedule = PolicyFrameScheduler(PolicyTimeConfig(60, 24, 1, 32))
    assert [schedule.pop() for _ in range(4)] == [2, 3, 2, 3]
    anchor = np.eye(4)
    target_a = np.eye(4); target_a[0, 3] = 0.01
    target_b = np.eye(4); target_b[0, 3] = 0.02
    a = encode_relative_eef(anchor, target_a, 0.03)
    b = encode_relative_eef(anchor, target_b, 0.03)
    assert a.shape == b.shape == (10,)
    assert a[0] == pytest.approx(0.01)
    assert b[0] == pytest.approx(0.02)


@pytest.mark.skipif(not _upstream_available(), reason="locked OpenPI checkout required")
def test_pi05_config_uses_official_32d_pi05_without_delta_transform(tmp_path):
    from force_bench.pi05.config import make_config
    cfg = make_config(
        "smoke", exp_name="test", data_path="dataset", model_root=tmp_path)
    assert cfg.model.pi05 is True
    assert cfg.model.action_dim == 32
    assert cfg.model.action_horizon == 32
    data = cfg.data.create(cfg.assets_dirs, cfg.model)
    names = [type(value).__name__ for value in data.data_transforms.inputs]
    assert names == ["ContactBenchInputs"]
    assert "DeltaActions" not in names
    assert cfg.batch_size == 1 and cfg.num_train_steps == 10


@pytest.mark.skipif(not _upstream_available(), reason="locked OpenPI checkout required")
def test_pi05_smoke_memory_profiles_preserve_official_contract(tmp_path):
    from force_bench.pi05.config import make_config

    smoke = make_config("smoke", exp_name="full", model_root=tmp_path)
    no_ema = make_config("smoke_no_ema", exp_name="no_ema", model_root=tmp_path)
    lora = make_config("smoke_lora", exp_name="lora", model_root=tmp_path)
    paper = make_config("paper", exp_name="paper", model_root=tmp_path)

    assert smoke.ema_decay == paper.ema_decay == 0.999
    assert no_ema.ema_decay is None
    assert lora.ema_decay is None
    for cfg in (smoke, no_ema, lora):
        assert cfg.batch_size == 1
        assert cfg.num_train_steps == 10
        assert cfg.model.pi05 is True
        assert cfg.model.action_dim == 32
        assert cfg.model.action_horizon == 32
        assert cfg.weight_loader.params_path.endswith("pi05_base/params")
    assert no_ema.model.paligemma_variant == "gemma_2b"
    assert no_ema.model.action_expert_variant == "gemma_300m"
    assert lora.model.paligemma_variant == "gemma_2b_lora"
    assert lora.model.action_expert_variant == "gemma_300m_lora"
    assert repr(lora.freeze_filter) == repr(lora.model.get_freeze_filter())
    assert repr(no_ema.freeze_filter) == repr(no_ema.model.get_freeze_filter())


@pytest.mark.skipif(not _upstream_available(), reason="locked OpenPI checkout required")
def test_pi05_normalization_validation_requires_exact_10d(tmp_path):
    from force_bench.pi05.config import ASSET_ID
    from force_bench.pi05.inference import _validate_stats
    from force_bench.pi05.official import activate_openpi
    activate_openpi()
    from openpi.shared import normalize
    good = normalize.NormStats(
        mean=np.zeros(10), std=np.ones(10),
        q01=np.zeros(10), q99=np.ones(10))
    normalize.save(tmp_path / "assets" / ASSET_ID, {
        "state": good, "actions": good})
    loaded = _validate_stats(tmp_path)
    assert set(loaded) == {"state", "actions"}

    bad = normalize.NormStats(
        mean=np.zeros(9), std=np.ones(9),
        q01=np.zeros(9), q99=np.ones(9))
    normalize.save(tmp_path / "assets" / ASSET_ID, {
        "state": good, "actions": bad})
    with pytest.raises(ValueError, match="invalid actions"):
        _validate_stats(tmp_path)


@pytest.mark.skipif(not _upstream_available(), reason="locked OpenPI checkout required")
def test_pi05_offline_and_online_share_policy_loader():
    from force_bench.pi05.inference import load_policy
    from force_bench.pi05 import official_predict, official_server
    assert official_predict.load_policy is load_policy
    assert official_server.load_policy is load_policy
