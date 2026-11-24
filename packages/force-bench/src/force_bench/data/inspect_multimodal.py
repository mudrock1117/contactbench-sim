"""Report tactile pressure integrity and causal RGB/tactile alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .episode import load_episode


def inspect_multimodal_episode(path):
    episode = load_episode(path)
    if "tactile" not in episode.arrays:
        raise ValueError("episode has no tactile pressure arrays")
    tactile = episode["tactile"]
    valid = episode["tactile_valid"]
    timestamps = episode["timestamps"]
    per_pad = {}
    for index, name in enumerate(("left", "right")):
        frames = tactile[:, index]
        valid_frames = frames[valid[:, index]]
        per_pad[name] = {
            "shape": list(frames.shape), "units": "Pa", "dtype": str(frames.dtype),
            "valid_fraction": float(valid[:, index].mean()),
            "pressure_min": (float(valid_frames.min()) if valid_frames.size else None),
            "pressure_max": (float(valid_frames.max()) if valid_frames.size else None),
            "nonzero_frame_fraction": float(np.any(frames > 0, axis=(1, 2)).mean())}
    alignments = {}
    rgb_root = Path(path) / "rgb"
    for camera in episode.metadata.get("rgb", {}).get("camera_names", []):
        rgb_t = np.load(rgb_root / camera / "timestamps.npy", allow_pickle=False)
        state_index = np.searchsorted(timestamps, rgb_t, side="right") - 1
        valid_index = state_index >= 0
        delta = rgb_t[valid_index] - timestamps[state_index[valid_index]]
        alignments[camera] = {
            "rgb_frames": int(len(rgb_t)),
            "causal_state_match_fraction": float(valid_index.mean()) if len(rgb_t) else None,
            "max_rgb_minus_tactile_timestamp_s": (float(np.max(np.abs(delta)))
                                                   if delta.size else None),
            "note": "tactile is sampled on the state timebase; pressure is not tactile RGB"}
    return {"episode": str(Path(path).resolve()), "tactile": per_pad,
            "visual_tactile_alignment": alignments}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    args = parser.parse_args(argv)
    try:
        report = inspect_multimodal_episode(args.episode)
    except (ValueError, OSError, KeyError) as error:
        print(json.dumps({"pass": False, "error": str(error)}, indent=2))
        return 1
    print(json.dumps({"pass": True, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
