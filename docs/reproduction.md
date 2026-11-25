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

At 12 Hz an action spans five 60 Hz frames. At 24 Hz the causal online schedule uses cumulative floor edges and
alternates 2 and 3 frames. Offline selection uses timestamps. Official horizon is
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


## Offline dependency boundary

Episode discovery, RGB checksum/timestamp validation, window construction, pure
10D EEF conversion, official training, offline prediction and the PyTorch inference
server do not import Warp, Newton, MuJoCo or TeleopCamera. Simulation dependencies
are loaded only when a capture/live camera or online EEF controller is constructed.
This same boundary is intended for the later RDP, pi0.5 and FTP-1 dataset adapters.

    python -c "from force_bench.data.dataset import EpisodeWindowDataset; from force_bench.robot.eef_action import encode_relative_eef; print('offline imports ok')"
    pytest -q packages/force-bench/tests/test_offline_dependency_isolation.py


## B2 — official Reactive Diffusion Policy

B2 is locked to `xiaoxiaoxh/reactive_diffusion_policy@8ed82a3`.  The
`force_bench.rdp` package is adapter code only: AT is the upstream
`VAE`, LDP is the upstream `LatentDiffusionUnetImagePolicy`, and training
uses the two upstream workspaces.  Third-party sources are never patched.

### Time and tensor contract

- physics: 60 Hz;
- fast action/tactile axis: 24 Hz (physics frames alternate 2 and 3);
- slow visual/latent planning: 12 Hz;
- two causal slow observations are selected from four 24 Hz slots with the
  upstream `[::-2][::-1]` rule, ending at the current anchor;
- AT horizon: 32 fast samples; the official action slice starts at index
  `2 * 2 - 1 = 3`;
- 10-D action: fixed-chunk-anchor relative EEF translation (m), rotation 6D,
  and absolute gripper width (m);
- observation: wrist RGB `[2,3,240,320]`, relative EEF state `[2,10]`,
  and two independent 15-D GelSight PCA embeddings `[2,30]`;
- AT's RNN decoder receives causal 24 Hz marker embeddings.  Online inference
  plans one latent at 12 Hz, then decodes one action at each of two 24 Hz
  ticks using only marker samples available at that tick.

### Marker/PCA compatibility gate

Hydrogel exposes physical normal and shear displacement, from which collection
can record the two 9x7 marker grids as
`tactile_marker_displacement_px [action,2,63,2]`. Components are image
`(x right, y down)`, in pixels; rows are ordered by the simulator's fixed
9x7 row-major identities. Set `TB_COLLECT_TACTILE_MARKERS=1` to enable it.

The official PCA files are present and have the expected affine contract:
normalized `[63*2]` offsets, `(x/width,y/height)`, then
`(x - mean) @ W -> 15D`. At 8ed82a3 the real detector declares a 7-row by
9-column grid and flattens its tracker output row-major. ContactBench renders a
9-row by 7-column grid because its tactile image axes are rotated. The public
code contains no calibration establishing rotation direction and marker-ID
correspondence between those two grids. A transpose inferred only from the
shared count of 63 is therefore not accepted as certification.

Two modes are supported:

- `official`: the original upstream mean/basis and strict
  `official_pca_compatible=true` gate; unchanged.
- `simulation_fitted`: independently fits centered 15-D PCA per physical pad
  from normalized ContactBench marker displacement in training episodes.
  The immutable artifact records episode IDs, source array hashes, split,
  fit scope, marker contract, normalization, NumPy version, code revision,
  reconstruction metrics and checksums of both means/bases.

A one-episode artifact must be explicitly marked
`engineering_smoke_single_episode`. Formal artifacts accept training episodes
only and must be fitted after the episode/seed split. Test episodes are never an
input. Pressure is never accepted as marker motion and no random feature is
generated. Training checkpoints store the PCA fingerprint; prediction and the
inference server reject a changed artifact, and the simulator client sends the
same fingerprint at every replan.

### Isolated environment

```bash
conda env create -f environments/contactbench-rdp.yaml
conda activate contactbench-rdp
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
  torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -e packages/force-bench --no-deps
python scripts/fetch_upstreams.py --name reactive_diffusion_policy
```

The simulator remains in the separate `contactbench` environment.  The
socket server is the boundary between PyTorch/cu117 and Warp/Newton.

### Commands (run only after the marker contract is certified)

```bash
export TB_RDP_DATASET=outputs/datasets/card_multimodal_rdp
export TB_RDP_UPSTREAM=third_party/reactive_diffusion_policy
export TB_RDP_PCA_MODE=simulation_fitted
export TB_RDP_PCA_ARTIFACT=outputs/rdp/pca/card_sim15

# Current single successful trajectory: engineering smoke only.
python -m force_bench.rdp.fit_pca "$TB_RDP_DATASET" \
  --output "$TB_RDP_PCA_ARTIFACT" --split-name train \
  --smoke-single-episode

python -m force_bench.rdp.audit_data "$TB_RDP_DATASET" \
  --pca-mode simulation_fitted --pca-artifact "$TB_RDP_PCA_ARTIFACT"

# Formal runs use a train-only directory/list and omit --smoke-single-episode.
# python -m force_bench.rdp.fit_pca outputs/datasets/card_rdp_train \
#   --output outputs/rdp/pca/card_train_sim15 --split-name train

# AT smoke, then paper-oriented training
python -m force_bench.rdp.official_train --stage at \
  --config packages/force-bench/config/rdp/at_smoke.yaml
python -m force_bench.rdp.official_train --stage at \
  --config packages/force-bench/config/rdp/at_paper.yaml

export TB_RDP_AT_CHECKPOINT=outputs/rdp/at_smoke/checkpoints/latest.ckpt
python -m force_bench.rdp.official_train --stage ldp \
  --config packages/force-bench/config/rdp/ldp_smoke.yaml
python -m force_bench.rdp.official_train --stage ldp \
  --config packages/force-bench/config/rdp/ldp_paper.yaml

python -m force_bench.rdp.official_predict \
  --config packages/force-bench/config/rdp/ldp_smoke.yaml \
  --checkpoint outputs/rdp/ldp_smoke/checkpoints/latest.ckpt

python -m force_bench.rdp.official_server \
  --config packages/force-bench/config/rdp/ldp_smoke.yaml \
  --checkpoint outputs/rdp/ldp_smoke/checkpoints/latest.ckpt --port 8766
```

In the simulation environment:

```bash
TB_GELSIGHT=1 python -m force_bench.rdp.run_closed_loop \
  --scene retrieve_card_from_deck --port 8766 \
  --pca-mode simulation_fitted \
  --pca-artifact outputs/rdp/pca/card_sim15
```

Collection for a new marker-bearing multimodal episode uses the normal task
command plus `TB_COLLECT_TACTILE_MARKERS=1`; it does not alter expert
control, cameras, physics, or QA.  The already validated DP data and B1
commands remain unchanged.

Static/regression checks recommended:

```bash
python -m compileall -q packages/force-bench/src/force_bench/rdp
pytest -q packages/force-bench/tests/test_rdp_official.py
pytest -q packages/force-bench/tests/test_a4_official_dp.py \
  packages/force-bench/tests/test_policy_interfaces.py
```
