"""Convert successful raw ContactBench episodes to one official FTP-1 Zarr v2 dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from force_bench.data.episode import discover_episodes
from .data import episode_to_ftp1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="raw ContactBench episode root")
    parser.add_argument("--output", required=True, help="new .zarr path; existing output is rejected")
    parser.add_argument("--success-only", action="store_true", default=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite FTP-1 output: {output}")
    episodes = discover_episodes(args.data, success_only=True)
    if not episodes:
        raise RuntimeError("no successful complete episodes found")
    from openpi.replay_buffer import ReplayBuffer
    replay = ReplayBuffer.create_from_path(str(output), mode="a")
    for path in episodes:
        replay.add_episode(episode_to_ftp1(path), compressors="disk")
    print(f"wrote {len(episodes)} successful episodes to {output}")


if __name__ == "__main__":
    main()
