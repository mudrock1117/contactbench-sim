#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ftp1_root="${repo_root}/third_party/ftp1-policy"
cache_root="${CONTACTBENCH_FTP1_CACHE:-/home/${USER}/contactbench_models/ftp1}"
repo_id="contactbench_retrieve_card"
config_path="${repo_root}/packages/force-bench/config/ftp1/dataset_contactbench.json"

cd "${ftp1_root}"
export OPENPI_DATA_HOME="${cache_root}/openpi"
uv run python scripts/zarr_compute_norm_stats.py \
  ftp1 \
  --repo_id="${repo_id}" \
  --data.repo-id="${repo_id}" \
  --exp_name=contactbench_retrieve_card_norm_smoke \
  --no-use_val_dataset \
  --no-create_train_val_split \
  --no-wandb_enabled \
  --checkpoint_base_dir="${cache_root}/checkpoints" \
  --assets_base_dir="${cache_root}/assets" \
  --dataset_config_path="${config_path}" \
  --batch_size=1 \
  --action_down_sample_steps=1 \
  --proprioception_pose_rep=relative \
  --action_pose_rep=relative \
  --proprioception_joint_rep=abs \
  --action_joint_rep=mix \
  --norm_type=zscore \
  --norm_sample_ratio=1.0 \
  --norm_batch_size=1 \
  --norm_num_workers=0
