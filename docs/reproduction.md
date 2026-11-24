# ContactBench reproduction lock and official B1

## Versions

The existing force_bench.dp.model implementation remains the legacy engineering baseline.
Official B1 directly imports the encoder, ConditionalUnet1D, DDPM scheduler,
LinearNormalizer, action slicing, EMA and workspace from
real-stanford/diffusion_policy at 5ba07ac.

Locked future routes are reactive_diffusion_policy at 8ed82a3, openpi at 15a9616,
and ftp1-policy at 89fa681. upstreams.lock.yaml is authoritative; third_party is ignored.

## A4 action and timing contract

The paper-facing action is 10D: translation in the fixed chunk-anchor TCP frame
(m), the first two columns of R_anchor_tcp_target (6D), and absolute gripper
width (m). A replan captures one measured TCP anchor. Every target in that chunk
uses the same anchor and is executed through existing FK/IK, limits and failure handling.

New episodes record measured command-start TCP pose, FK of the exact recorded 9D
expert target, measured width and commanded width. Old episodes missing these
fields are rejected for 10D; state differences are never substituted for commands.
Recorded action_target_velocity and action_feedforward remain part of the equivalence audit.

At 12 Hz an action spans five 60 Hz frames. At 24 Hz the causal online schedule
alternates 3 and 2 frames. Offline selection uses timestamps. Official horizon is
16, observation history 2 and execution horizon 8. Upstream predict_action starts
at n_obs_steps-1 (index 1), so the dataset places the causal previous command at
index 0 and current command at index 1. Images are causal only.

## Install

    python scripts/fetch_upstreams.py --verify-only
    conda create -n contactbench-dp python=3.10 -y
    conda activate contactbench-dp
    python -m pip install --upgrade pip
    python -m pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -e third_party/diffusion_policy
    python -m pip install -e packages/force-bench --no-deps

Torch/CUDA are deployment pins for RTX 4090, not paper-reported hyperparameters.
Undisclosed network, EMA and scheduler values are inherited from the pinned upstream.

## Data and training

outputs/datasets/card_multimodal remains usable by legacy 9D. Official 10D can
reuse it only if it contains the four A4 FK/anchor arrays. Otherwise collect into
a new directory (never overwrite old data):

    conda activate contactbench
    TB_COLLECT_DIR=outputs/datasets/card_multimodal_a4 TB_COLLECT_TASK=retrieve_card_from_deck TB_COLLECT_RGB=1 TB_COLLECT_RGB_MODE=wrist TB_COLLECT_RGB_FPS=12 python -m force_bench.appendix_tasks.retrieve_card_from_deck

Smoke and paper runs:

    conda activate contactbench-dp
    TB_DP_DATA=outputs/datasets/card_multimodal_a4 python -m force_bench.dp.official_train --config packages/force-bench/config/dp/official_smoke.yaml
    TB_DP_DATA=outputs/datasets/card_multimodal_a4 python -m force_bench.dp.official_train --config packages/force-bench/config/dp/official_paper.yaml

Paper config is wrist RGB 3x240x320, 12 Hz, obs 2, horizon 16, action 10D,
batch 64, LR 1e-4, cosine, warmup 500 and 501 epochs from scratch.
Single-episode overfit uses every window; its validation is diagnostic, not an
independent test split. The upstream-compatible offline runner deliberately returns
metrics only and never starts Newton from the training workspace. Images are divided
by 255 and receive an explicit identity normalizer; action and agent_pos use the
official LinearNormalizer.

Prediction:

    TB_DP_DATA=outputs/datasets/card_multimodal_a4 python -m force_bench.dp.official_predict --config packages/force-bench/config/dp/official_smoke.yaml --checkpoint outputs/dp/official_card_smoke/checkpoints/latest.ckpt

Training-side inference server:

    conda activate contactbench-dp
    python -m force_bench.dp.official_server --config packages/force-bench/config/dp/official_smoke.yaml --checkpoint outputs/dp/official_card_smoke/checkpoints/latest.ckpt --device cuda:0

Simulation-side closed loop:

    conda activate contactbench
    python -m force_bench.dp.run_official_closed_loop --policy-hz 12 --execution-horizon 8 --max-steps 600

The existing Card evaluator and TCP/NPZ transport are reused. No expert controller
or expert auxiliary command is injected. Upstream predicts a 16-step action_pred,
then returns exactly action_pred[1:9], so each closed-loop replan executes exactly
eight 10D actions. Dataset index 1 is the command at the current observation and
every target in the sequence is expressed against that current fixed TCP anchor.
Episodes with missing/non-finite fields or nonzero recorded velocity/feedforward
controls are rejected rather than silently converted.

## Checks

    python -m compileall -q packages/force-bench/src/force_bench scripts
    pytest -q packages/force-bench/tests/test_a4_official_dp.py
    pytest -q packages/force-bench/tests/test_diffusion_policy.py
    pytest -q packages/force-bench/tests/test_policy_interfaces.py

## Limits and next routes

Old Card data usually needs recollection for exact 10D but remains valid legacy 9D.
The 10D representation does not prove position-only control is equivalent when
recorded auxiliary commands are nonzero. Formal results need whole episode/seed
splits and multiple episodes. RDP, pi0.5 and FTP-1 are version-locked only and
still need model-specific adapters. GelSight pressure is physical pressure, not
tactile RGB, and no conversion is implied.
