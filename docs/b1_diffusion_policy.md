# B1 Diffusion Policy baseline

This module is a real conditional denoising policy, not an MLP action
regressor.  It follows Stanford's official hybrid image policy structure:
per-observation visual/low-dimensional features form a global condition; a
Conditional 1D U-Net receives sinusoidal diffusion-step embeddings and FiLM
scale/bias modulation; training predicts random DDPM noise and inference runs
the scheduler's complete iterative reverse process.  An EMA copy is used for
deployment.

The ContactBench adapter deliberately differs from the official repository in
four ways.  It uses a torchvision ResNet-18 with GroupNorm instead of the
Robomimic observation encoder; a small dataclass/YAML trainer replaces Hydra;
it consumes `EpisodeWindowDataset` directly; and action index zero is the
action effective at the current policy timestamp (the dataset already defines
that causal convention), rather than applying an additional observation-step
offset.  The scheduler is Hugging Face `DDPMScheduler` with the official
hybrid configuration's squared-cosine beta schedule and epsilon prediction.

## Semantics and leakage boundary

The explicit time configuration is 60 Hz physics, 12 Hz policy, two causal
observations and sixteen future 12 Hz joint targets.  The model receives only
CHW uint8 RGB and the first 18 robot proprioceptive scalars selected by the
episode's non-privileged field metadata.  Tactile, language, object poses,
expert phases, future RGB and future proprioception are not inputs.  Actions
are `[7 joint radians, 2 finger metres]`; this 9D engineering baseline is not
the paper's 10D end-effector action representation.

Normalization statistics are computed only from the complete training
episodes and stored in every checkpoint.  `overfit_single_episode: true` uses
all windows of the sole successful episode and creates no misleading random
window test split.  Multi-episode runs use deterministic whole task, episode,
or seed groups.

`run_config.json` records maximum `action_target_velocity` and
`action_feedforward`.  Joint-position-only deployment is equivalent to the
recorded robot command only when both are zero.  The policy runner never
injects expert controls or calls the expert callback.

## Isolated training environment

The simulator remains in `contactbench`.  Training uses Python 3.11 and does
not install Newton/Warp:

```bash
conda create -n contactbench-dp python=3.11 -y
conda activate contactbench-dp
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install diffusers>=0.31 pyyaml imageio pillow numpy pytest
python -m pip install -e packages/force-bench --no-deps
```

The pinned CUDA 12.4 pair is an official PyTorch release and works with a
sufficiently recent NVIDIA driver; it does not require the simulator's Python
packages.

## Train and inspect

```bash
conda activate contactbench-dp
CUDA_VISIBLE_DEVICES=0 python -m force_bench.dp.train \
  --config packages/force-bench/configs/dp_card_wrist.yaml \
  --data outputs/datasets/card_multimodal

CUDA_VISIBLE_DEVICES=0 python -m force_bench.dp.predict \
  --checkpoint outputs/dp/card_wrist_overfit/latest.pt \
  --data outputs/datasets/card_multimodal --index 0 --device cuda:0
```

## Cross-environment closed loop

PyTorch inference and Newton simulation are separate processes connected only
over localhost by length-prefixed compressed NumPy arrays.  Start the server:

```bash
conda activate contactbench-dp
CUDA_VISIBLE_DEVICES=0 python -m force_bench.dp.server \
  --checkpoint outputs/dp/card_wrist_overfit/latest.pt \
  --host 127.0.0.1 --port 8765 --device cuda:0
```

Then run the unchanged simulator in another shell:

```bash
conda activate contactbench
TB_GELSIGHT=0 CUDA_VISIBLE_DEVICES=0 python -m force_bench.dp.run_closed_loop \
  --scene retrieve_card_from_deck --camera wrist \
  --host 127.0.0.1 --port 8765 --execution-horizon 8 --max-steps 600
```

The Card evaluator checks the original state thresholds for the retrieved card
and remaining deck plus release and 5 cm retreat.  Original rollout contact
history checks are unavailable in the policy loop, so `strict_success` remains
null and `physical_success` is never reported as the appendix/paper success
rate.
