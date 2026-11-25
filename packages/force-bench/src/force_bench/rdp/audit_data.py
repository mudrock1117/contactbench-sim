"""Audit episodes against the selected official or simulation-fitted PCA."""
from __future__ import annotations
import argparse
import json
import numpy as np
from force_bench.data.episode import discover_episodes, load_episode
from .tactile import load_gelsight_pca


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("dataset")
    p.add_argument("--pca-mode", choices=("official", "simulation_fitted"),
                   default="official")
    p.add_argument("--pca-artifact")
    p.add_argument("--pca-transform")
    p.add_argument("--pca-mean")
    a = p.parse_args(argv)
    shared = (load_gelsight_pca(
        mode="simulation_fitted", artifact_path=a.pca_artifact)
        if a.pca_mode == "simulation_fitted" else None)
    reports = []
    for path in discover_episodes(a.dataset, success_only=False):
        ep = load_episode(path)
        fields = ep.arrays
        marker = fields.get("tactile_marker_displacement_px")
        valid = fields.get("tactile_marker_valid")
        contract = ep.metadata.get("tactile_markers", {})
        pca_ready, pca_error = False, None
        try:
            pca = (shared if shared is not None else load_gelsight_pca(
                mode="official", transform_path=a.pca_transform,
                mean_path=a.pca_mean, marker_contract=contract))
            if marker is not None and valid is not None:
                both = np.flatnonzero(np.asarray(valid, bool).all(axis=1))
                if len(both):
                    projected = pca.transform(
                        np.asarray(marker)[both[:1]],
                        image_width_px=240, image_height_px=320)
                    pca_ready = projected.shape == (1, 2, 15) and np.isfinite(projected).all()
        except (OSError, ValueError, KeyError) as error:
            pca_error = f"{type(error).__name__}: {error}"
        report = {
            "episode": str(path),
            "has_marker_displacement": marker is not None,
            "marker_shape": None if marker is None else list(np.asarray(marker).shape),
            "valid_ratio": None if valid is None else float(np.asarray(valid).mean()),
            "pca_mode": a.pca_mode,
            "official_pca_compatible": bool(
                contract.get("official_pca_compatible", False)),
            "pca_ready": bool(pca_ready),
            "pca_error": pca_error,
            "pressure_not_used_as_marker": True,
        }
        report["ready"] = bool(
            marker is not None and valid is not None
            and np.asarray(marker).shape[1:] == (2, 63, 2)
            and np.asarray(valid).shape[1:] == (2,)
            and pca_ready)
        reports.append(report)
    print(json.dumps({
        "pca_mode": a.pca_mode, "episodes": reports,
        "ready_count": sum(x["ready"] for x in reports)}, indent=2))
    return 0 if reports and all(x["ready"] for x in reports) else 2


if __name__ == "__main__":
    raise SystemExit(main())
