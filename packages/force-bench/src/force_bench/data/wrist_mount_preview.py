"""Plot rigid wrist-camera candidates in the FR3 TCP frame (no simulation).

This checks mount direction and gross self-occlusion geometry.  It is not a
replacement for rendering a short trajectory, because robot meshes and the
task scene are intentionally not loaded here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


CANDIDATES = {
    "tool": {"position": [0.055, 0.0, -0.035], "target": [0.0, 0.0, 0.110], "fov_deg": 72.0},
    "wide": {"position": [0.065, 0.0, -0.045], "target": [0.0, 0.0, 0.130], "fov_deg": 80.0},
    "close": {"position": [0.050, 0.0, -0.025], "target": [0.0, 0.0, 0.085], "fov_deg": 65.0},
}


def plot_candidates(output, names=("tool", "wide", "close")):
    import matplotlib.pyplot as plt

    unknown = set(names) - set(CANDIDATES)
    if unknown:
        raise ValueError(f"unknown candidates: {sorted(unknown)}")
    figure = plt.figure(figsize=(11, 5))
    ax = figure.add_subplot(111, projection="3d")
    colors = ("tab:blue", "tab:orange", "tab:green")
    ax.plot([0, 0], [0, 0], [-0.06, 0.16], "k--", label="TCP +Z / tool approach")
    ax.scatter([0], [0], [0], color="black", marker="x", s=60, label="TCP origin")
    for color, name in zip(colors, names):
        candidate = CANDIDATES[name]
        position, target = np.asarray(candidate["position"]), np.asarray(candidate["target"])
        direction = target - position
        ax.scatter(*position, color=color, s=45)
        ax.quiver(*position, *direction, color=color, arrow_length_ratio=0.12,
                  label=f"{name}: {candidate['fov_deg']:.0f} deg")
        ax.scatter(*target, color=color, marker=".", s=25)
    ax.set(xlabel="TCP +X [m]", ylabel="TCP +Y [m]", zlabel="TCP +Z [m]",
           title="Rigid wrist-camera candidates; arrows are optical forward (-Z_camera)")
    ax.set_box_aspect((1.0, 0.7, 1.5))
    ax.legend(loc="upper left")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--candidates", nargs="+", choices=sorted(CANDIDATES),
                        default=list(CANDIDATES))
    args = parser.parse_args(argv)
    plot_candidates(args.output, args.candidates)


if __name__ == "__main__":
    main()
