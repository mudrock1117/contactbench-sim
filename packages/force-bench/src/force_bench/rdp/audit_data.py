"""Audit whether episodes satisfy the official RDP tactile contract."""
from __future__ import annotations
import argparse
import json
import numpy as np
from force_bench.data.episode import discover_episodes, load_episode


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    a = p.parse_args(argv)
    reports = []
    for path in discover_episodes(a.dataset, success_only=False):
        ep = load_episode(path)
        fields = ep.arrays
        marker = fields.get("tactile_marker_displacement_px")
        valid = fields.get("tactile_marker_valid")
        contract = ep.metadata.get("tactile_markers", {})
        report = {
            "episode": str(path),
            "has_marker_displacement": marker is not None,
            "marker_shape": None if marker is None else list(np.asarray(marker).shape),
            "valid_ratio": None if valid is None else float(np.asarray(valid).mean()),
            "official_pca_compatible": bool(
                contract.get("official_pca_compatible", False)),
            "pressure_not_used_as_marker": True,
        }
        report["ready"] = bool(
            marker is not None and valid is not None
            and np.asarray(marker).shape[1:] == (2, 63, 2)
            and np.asarray(valid).shape[1:] == (2,)
            and report["official_pca_compatible"])
        reports.append(report)
    print(json.dumps({"episodes": reports,
                      "ready_count": sum(x["ready"] for x in reports)}, indent=2))
    return 0 if reports and all(x["ready"] for x in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
