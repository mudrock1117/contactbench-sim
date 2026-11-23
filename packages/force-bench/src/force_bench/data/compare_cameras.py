"""Create a fixed/wrist contact sheet from one completed episode.

When no reliable phase log is supplied, columns are explicitly labelled as
time samples; they are not presented as inferred contact/rotation phases.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from .episode import load_episode


def _nearest(directory, timestamp):
    times = np.load(directory / "timestamps.npy", allow_pickle=False)
    index = int(np.argmin(np.abs(times - timestamp)))
    return iio.imread(directory / "images" / f"{index:06d}.png"), float(times[index])


def make_contact_sheet(episode_path, output, *, times=None):
    episode_path, output = Path(episode_path), Path(output)
    episode = load_episode(episode_path)
    for name in ("fixed", "wrist"):
        if not (episode_path / "rgb" / name).is_dir():
            raise ValueError(f"episode has no {name} camera")
    if times is None:
        end = float(episode["timestamps"][-1])
        times = np.linspace(0.0, end, 4)
        labels = [f"time sample {value:.2f}s" for value in times]
    else:
        times = np.asarray(times, dtype=float)
        if times.shape != (4,):
            raise ValueError("exactly four explicit sample times are required")
        labels = [f"explicit time {value:.2f}s" for value in times]
    cells = []
    for name in ("fixed", "wrist"):
        row = []
        for value, label in zip(times, labels):
            frame, actual = _nearest(episode_path / "rgb" / name, value)
            image = Image.fromarray(frame)
            canvas = Image.new("RGB", (image.width, image.height + 24), "white")
            canvas.paste(image, (0, 24))
            ImageDraw.Draw(canvas).text((4, 4), f"{name}: {label} (frame {actual:.2f}s)", fill="black")
            row.append(canvas)
        cells.append(row)
    width = sum(cell.width for cell in cells[0])
    height = sum(row[0].height for row in cells)
    sheet = Image.new("RGB", (width, height), "white")
    y = 0
    for row in cells:
        x = 0
        for cell in row:
            sheet.paste(cell, (x, y)); x += cell.width
        y += row[0].height
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("output")
    parser.add_argument("--times", nargs=4, type=float,
                        help="four simulator times; without these, output is labelled time sampling")
    args = parser.parse_args(argv)
    make_contact_sheet(args.episode, args.output, times=args.times)


if __name__ == "__main__":
    main()
