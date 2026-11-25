"""Fit a versioned per-pad 15-D PCA artifact from training episodes only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

from force_bench.data.episode import discover_episodes, load_episode
from .tactile import MARKER_COUNT, marker_metadata


def _array_sha256(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _fit(value):
    mean = value.mean(axis=0, dtype=np.float64)
    centered = value.astype(np.float64) - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    W = vt[:15].T
    # Canonicalize component signs so repeated fits are deterministic.
    pivots = np.argmax(np.abs(W), axis=0)
    signs = np.sign(W[pivots, np.arange(W.shape[1])])
    signs[signs == 0] = 1
    W *= signs
    projected = centered @ W
    reconstructed = projected @ W.T + mean
    total = float(np.square(singular).sum())
    captured = float(np.square(singular[:15]).sum() / total) if total > 0 else 1.0
    rmse = float(np.sqrt(np.mean(np.square(value - reconstructed))))
    return mean.astype(np.float32), W.astype(np.float32), captured, rmse


def _git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--output", required=True)
    p.add_argument("--split-name", choices=("train",), default="train")
    p.add_argument("--episode-id", action="append", default=[])
    p.add_argument("--train-episode-list",
                   help="newline-delimited train episode IDs; required for formal fits")
    p.add_argument("--smoke-single-episode", action="store_true")
    a = p.parse_args(argv)

    output = Path(a.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PCA artifact {output}")
    selected_ids = set(a.episode_id)
    list_path = Path(a.train_episode_list) if a.train_episode_list else None
    if list_path is not None:
        if selected_ids:
            raise ValueError("use --episode-id or --train-episode-list, not both")
        selected_ids = {
            line.strip() for line in list_path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")}
        if not selected_ids:
            raise ValueError("train episode list is empty")
    paths = discover_episodes(a.dataset, success_only=True)
    episodes = []
    for path in paths:
        ep = load_episode(path)
        if selected_ids and str(ep.metadata["episode_id"]) not in selected_ids:
            continue
        episodes.append((path, ep))
    if not episodes:
        raise ValueError("no selected successful training episodes")
    if len(episodes) == 1 and not a.smoke_single_episode:
        raise ValueError(
            "one-episode PCA is smoke-only; pass --smoke-single-episode explicitly")
    if len(episodes) > 1 and a.smoke_single_episode:
        raise ValueError("--smoke-single-episode requires exactly one selected episode")
    if not a.smoke_single_episode and list_path is None:
        raise ValueError(
            "formal PCA fitting requires --train-episode-list to prevent test leakage")
    if list_path is not None:
        found = {str(ep.metadata["episode_id"]) for _, ep in episodes}
        missing = selected_ids - found
        if missing:
            raise ValueError(f"train episode IDs not found: {sorted(missing)}")

    per_side = [[], []]
    sources = []
    for path, ep in episodes:
        if "tactile_marker_displacement_px" not in ep.arrays:
            raise ValueError(f"{path}: missing marker displacement")
        marker = np.asarray(ep["tactile_marker_displacement_px"], np.float32)
        valid = np.asarray(ep["tactile_marker_valid"], bool)
        if marker.shape[1:] != (2, MARKER_COUNT, 2) or valid.shape != marker.shape[:2]:
            raise ValueError(f"{path}: invalid marker/valid shapes")
        normalized = marker.copy()
        normalized[..., 0] /= 240.0
        normalized[..., 1] /= 320.0
        for side in range(2):
            per_side[side].append(normalized[valid[:, side], side].reshape(-1, 126))
        sources.append({
            "episode_id": str(ep.metadata["episode_id"]),
            "path": str(path),
            "frames": int(len(marker)),
            "marker_sha256": _array_sha256(marker),
            "valid_sha256": _array_sha256(valid),
        })

    output.mkdir(parents=True)
    metrics = {}
    hashes = {}
    for side_name, chunks in zip(("left", "right"), per_side):
        value = np.concatenate(chunks)
        if len(value) < 15 or not np.isfinite(value).all():
            raise ValueError(f"{side_name}: insufficient or non-finite fit samples")
        mean, W, captured, rmse = _fit(value)
        mean_path = output / f"{side_name}_mean.npy"
        transform_path = output / f"{side_name}_transform.npy"
        np.save(mean_path, mean)
        np.save(transform_path, W)
        hashes[mean_path.name] = hashlib.sha256(mean_path.read_bytes()).hexdigest()
        hashes[transform_path.name] = hashlib.sha256(transform_path.read_bytes()).hexdigest()
        metrics[side_name] = {
            "samples": int(len(value)),
            "explained_variance_ratio_15d": captured,
            "normalized_reconstruction_rmse": rmse,
        }

    manifest = {
        "format_version": 1,
        "mode": "simulation_fitted",
        "component_count_per_side": 15,
        "pad_order": ["gel_left", "gel_right"],
        "marker_contract": marker_metadata(official_pca_compatible=False),
        "normalization": {
            "x": "divide_by_image_width_px",
            "y": "divide_by_image_height_px",
        },
        "fit_split": a.split_name,
        "fit_scope": "engineering_smoke_single_episode"
                     if a.smoke_single_episode else "training_episodes_only",
        "source_dataset": str(Path(a.dataset)),
        "train_episode_list": (None if list_path is None else {
            "path": str(list_path),
            "sha256": hashlib.sha256(list_path.read_bytes()).hexdigest(),
        }),
        "source_episodes": sources,
        "fit_algorithm": "numpy.linalg.svd; centered; per-pad; deterministic sign",
        "numpy_version": np.__version__,
        "force_bench_commit": _git_commit(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "sha256": hashes,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "artifact": str(output), "mode": manifest["mode"],
        "fit_scope": manifest["fit_scope"], "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
