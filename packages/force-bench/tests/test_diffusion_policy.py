"""CPU/synthetic contracts for the optional B1 Diffusion Policy module."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("diffusers")

from force_bench.data.timebase import ActionChunkExecutor
from force_bench.dp.checkpoint import load_policy, save_checkpoint
from force_bench.dp.config import DPConfig
from force_bench.dp.data import TorchWindowDataset, window_spec
from force_bench.dp.model import (ConditionalUnet1D, DiffusionPolicy, EMA,
                                  ImageEncoder)


def tiny_config():
    return DPConfig(camera="wrist", image_resize=32, unet_dims=(16, 32, 64),
                    diffusion_embed_dim=32, diffusion_train_timesteps=8,
                    inference_steps=4, batch_size=2, max_steps=1, epochs=1,
                    num_workers=0, device="cpu")


def batch(config, size=2):
    return (torch.randn(size, 2, 18),
            {"wrist": torch.randint(0, 256, (size, 2, 3, 32, 32),
                                     dtype=torch.uint8)},
            torch.randn(size, 16, 9).clamp(-1, 1))


def test_model_shape_loss_gradient_and_iterative_ddpm():
    config = tiny_config(); model = DiffusionPolicy(config)
    proprio, images, actions = batch(config)
    loss = model.compute_loss(proprio, images, actions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())
    model.eval()
    predicted = model.predict_actions(proprio, images, inference_steps=4)
    assert predicted.shape == (2, 16, 9)
    assert torch.isfinite(predicted).all()


@pytest.mark.parametrize("down_dims", [(16, 32, 64), (256, 512, 1024)])
def test_conditional_unet_skip_channels_forward_and_backward(down_dims):
    # kernel_size=1 keeps the production channel topology while making the
    # 256/512/1024 CPU regression reasonably small.
    model = ConditionalUnet1D(9, global_condition_dim=24,
                              diffusion_embed_dim=16, down_dims=down_dims,
                              kernel_size=1)
    sample = torch.randn(1, 16, 9, requires_grad=True)
    condition = torch.randn(1, 24, requires_grad=True)
    output = model(sample, torch.tensor([3]), condition)
    assert output.shape == (1, 16, 9)
    assert len(model.up_specs) == len(down_dims)
    for spec, modules in zip(model.up_specs, model.up):
        current, skip, output_channels, next_channels = spec
        first, _, upsample = modules
        assert first.blocks[0].net[0].in_channels == current + skip
        assert first.blocks[0].net[0].out_channels == output_channels
        if isinstance(upsample, torch.nn.ConvTranspose1d):
            assert upsample.in_channels == output_channels
            assert upsample.out_channels == next_channels
        else:
            assert output_channels == next_channels == down_dims[0]
    output.square().mean().backward()
    assert sample.grad is not None and torch.isfinite(sample.grad).all()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters() if parameter.requires_grad)


def test_resnet_and_ddpm_shapes_preserve_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = ImageEncoder(resize=32, pretrained=False).to(device)
    images = torch.randint(0, 256, (2, 3, 40, 48), dtype=torch.uint8,
                           device=device)
    features = encoder(images)
    assert features.shape == (2, 512) and features.device == device
    config = tiny_config(); model = DiffusionPolicy(config).to(device)
    clean = torch.randn(2, 16, 9, device=device)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([0, 7], device=device)
    noisy = model.scheduler.add_noise(clean, noise, timesteps)
    assert noisy.shape == clean.shape and noisy.device == device


def test_fixed_wrist_and_both_visual_modes_build_real_conditions():
    for camera in ("fixed", "wrist", "both"):
        config = tiny_config(); config.camera = camera
        model = DiffusionPolicy(config)
        proprio = torch.zeros(1, 2, 18)
        images = {name: torch.zeros(1, 2, 3, 32, 32, dtype=torch.uint8)
                  for name in config.camera_names}
        condition = model.encode_observation(proprio, images)
        expected = 2 * (18 + 512 * len(config.camera_names))
        assert condition.shape == (1, expected)


def test_action_normalization_round_trip():
    model = DiffusionPolicy(tiny_config())
    statistics = {"proprio_mean": np.zeros(18), "proprio_std": np.ones(18),
                  "action_min": -np.arange(1, 10), "action_max": np.arange(1, 10)}
    model.set_normalization(statistics)
    actions = torch.linspace(-1, 1, 18).reshape(2, 9)
    torch.testing.assert_close(model.unnormalize_actions(model.normalize_actions(actions)),
                               actions)


def test_checkpoint_round_trip(tmp_path):
    config = tiny_config(); model = DiffusionPolicy(config)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = EMA(model)
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, ema, optimizer, config, step=7, epoch=2)
    restored, restored_config, state = load_policy(path)
    assert restored_config.policy_hz == 12 and state["step"] == 7
    assert set(restored.state_dict()) == set(model.state_dict())


def test_explicit_12hz_window_and_replanning_semantics():
    spec = window_spec(tiny_config())
    assert spec.policy_time_config.physics_hz == 60
    assert spec.policy_time_config.policy_hz == 12
    assert spec.observation_horizon == 2 and spec.action_horizon == 16
    executor = ActionChunkExecutor(); first = np.arange(16 * 9).reshape(16, 9)
    executor.replan(first)
    np.testing.assert_array_equal(executor.pop(), first[0])
    replacement = np.full((8, 9), 3.0); executor.replan(replacement)
    np.testing.assert_array_equal(executor.pop(), replacement[0])
    assert executor.remaining == 7


def test_training_adapter_exposes_no_privileged_or_future_fields():
    class Synthetic:
        def __len__(self): return 1
        def __getitem__(self, index):
            return {"proprio": np.zeros((2, 18), np.float32),
                    "rgb": np.zeros((2, 3, 8, 8), np.uint8),
                    "actions": np.zeros((16, 9), np.float32),
                    "object_pose": np.ones(7), "language_instruction": "hidden",
                    "tactile": np.ones((2, 2, 16, 16))}
    sample = TorchWindowDataset(Synthetic(), ("wrist",))[0]
    assert set(sample) == {"proprio", "images", "actions"}
    assert set(sample["images"]) == {"wrist"}
