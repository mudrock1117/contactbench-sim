"""GelSight marker-displacement contract and official 15-D PCA.

Pressure rasters are intentionally rejected.  Marker positions are derived
from Hydrogel's physical shear/normal-displacement state; no random or
force-filled surrogate is accepted.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np

MARKER_COUNT = 63
MARKER_COMPONENTS = ("x_right_px", "y_down_px")
PAD_ORDER = ("gel_left", "gel_right")
IMAGE_WIDTH_PX = 240
IMAGE_HEIGHT_PX = 320
ORDERING = "9x7 row-major in image y then x"


def marker_metadata(*, official_pca_compatible: bool = False) -> dict:
    return {
        "quantity": "marker_displacement",
        "shape": [2, MARKER_COUNT, 2],
        "dtype": "float32",
        "units": "image_pixels",
        "pad_order": list(PAD_ORDER),
        "components": list(MARKER_COMPONENTS),
        "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
        "marker_ordering": ORDERING,
        "source": "hydrogel physical normal/shear displacement fields",
        "official_pca_compatible": bool(official_pca_compatible),
        "compatibility_note": (
            "Official PCA requires the exact real-detector marker ID ordering. "
            "The simulator has the same 63-marker cardinality but no upstream "
            "evidence that identities/order match; false is fail-closed."
        ),
    }


def marker_window_indices(marker_timestamps, marker_valid, query_timestamps):
    """Return causal indices, or None when any required sensor sample is invalid.

    Validity is independent of displacement magnitude: all-zero no-contact
    marker observations remain valid.
    """
    from force_bench.data.timebase import causal_indices
    timestamps = np.asarray(marker_timestamps, dtype=np.float64)
    query = np.asarray(query_timestamps, dtype=np.float64)
    valid = np.asarray(marker_valid, dtype=bool)
    if valid.ndim != 2 or valid.shape[1] != 2 or len(valid) != len(timestamps):
        raise ValueError("marker validity must have shape [N,2] aligned to timestamps")
    indices = causal_indices(timestamps, query)
    if np.any(indices < 0) or np.any(indices >= len(valid)):
        return None
    return indices if np.all(valid[indices]) else None


def frame_marker_displacement(frame, *, projector=None) -> np.ndarray:
    """Return [63,2] (x right, y down) displacement in image pixels."""
    if not hasattr(frame, "shear_displacement_m") or not hasattr(frame, "normal_displacement_m"):
        raise TypeError("frame lacks physical GelSight displacement fields")
    shear = np.asarray(frame.shear_displacement_m, dtype=np.float32)
    normal = np.asarray(frame.normal_displacement_m, dtype=np.float32)
    if shear.ndim != 3 or shear.shape[0] != 2 or normal.shape != shear.shape[1:]:
        raise ValueError("invalid Hydrogel displacement field shapes")
    # This path never constructs the optical Taxim renderer. Marker mechanics
    # remain available when RGB/video rendering is disabled.
    from hydrogel.render import (
        GS_H, GS_W, MARKER_MODEL, advect_markers, marker_grid,
        project_markers, raster_to_image)
    from hydrogel.markers import HydroShearMarkers, field_markers
    rest_hw = np.asarray(marker_grid(), dtype=np.float32)
    height = raster_to_image(normal)
    if MARKER_MODEL == "local":
        current_hw = project_markers(advect_markers(shear, rest_hw), height)
    elif MARKER_MODEL == "hydroshear":
        if projector is None:
            projector = HydroShearMarkers(
                rest_hw, (GS_H, GS_W), (0.0186, 0.0143))
        current_hw = project_markers(projector.displace(height, shear)[0], height)
    else:
        current_hw = project_markers(
            field_markers(rest_hw, shear, (GS_H, GS_W)), height)
    current_hw = np.asarray(current_hw, dtype=np.float32)
    if rest_hw.shape != (MARKER_COUNT, 2) or current_hw.shape != rest_hw.shape:
        raise ValueError("Hydrogel marker model is not the required 9x7 grid")
    delta_hw = current_hw - rest_hw
    # Hydrogel stores (image row h, image column w); official convention is x,y.
    return delta_hw[:, [1, 0]].copy()


def read_rig_markers(rig) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((2, MARKER_COUNT, 2), dtype=np.float32)
    projector = getattr(rig, "_rdp_marker_projector", None)
    if projector is None:
        from hydrogel.markers import HydroShearMarkers
        from hydrogel.render import GS_H, GS_W, marker_grid
        projector = HydroShearMarkers(
            marker_grid(), (GS_H, GS_W), (0.0186, 0.0143))
        rig._rdp_marker_projector = projector
    valid = np.zeros(2, dtype=bool)
    for index, pad in enumerate(PAD_ORDER):
        frames = getattr(rig, "tactile_frames", {}).get(pad, ())
        if frames:
            values[index] = frame_marker_displacement(
                frames[-1], projector=projector)
            valid[index] = bool(np.isfinite(values[index]).all())
    return values, valid


class OfficialGelSightPCA:
    """Exact affine projection used by upstream PCAEmbedding.pca_reduction."""

    def __init__(self, transform_path: str | Path, mean_path: str | Path,
                 *, marker_contract: dict):
        if not marker_contract.get("official_pca_compatible", False):
            raise ValueError(
                "marker ordering is not certified compatible with the official PCA basis")
        self.W = np.load(transform_path)
        self.mean = np.load(mean_path)
        expected = MARKER_COUNT * 2
        if self.W.shape != (expected, 15) or self.mean.shape not in ((expected,), (1, expected)):
            raise ValueError(
                f"official PCA assets must be mean [{expected}] and W [{expected},15], "
                f"got {self.mean.shape} and {self.W.shape}")
        self.mean = self.mean.reshape(expected).astype(np.float32)
        self.W = self.W.astype(np.float32)
        self.fingerprint = pca_basis_fingerprint(
            mode="official", transform_path=transform_path, mean_path=mean_path)

    @classmethod
    def from_upstream(cls, root: str | Path, *, marker_contract: dict):
        base = Path(root) / "data" / "PCA_Transform_GelSight"
        return cls(base / "pca_transform_matrix.npy", base / "pca_mean_matrix.npy",
                   marker_contract=marker_contract)

    def transform(self, displacement_px: np.ndarray,
                  *, image_width_px: int, image_height_px: int) -> np.ndarray:
        value = np.asarray(displacement_px, dtype=np.float32)
        if value.shape[-2:] != (MARKER_COUNT, 2):
            raise ValueError("marker displacement must end in [63,2]; pressure is not accepted")
        if image_width_px <= 0 or image_height_px <= 0:
            raise ValueError("image dimensions must be positive")
        normalized = value.copy()
        normalized[..., 0] /= float(image_width_px)
        normalized[..., 1] /= float(image_height_px)
        flat = normalized.reshape(*normalized.shape[:-2], MARKER_COUNT * 2)
        return ((flat - self.mean) @ self.W).astype(np.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pca_basis_fingerprint(*, mode: str,
                          transform_path: str | Path | None = None,
                          mean_path: str | Path | None = None,
                          artifact_path: str | Path | None = None) -> str:
    """Fingerprint the exact PCA mode and numeric basis used at deployment."""
    digest = hashlib.sha256()
    digest.update(mode.encode())
    if mode == "official":
        if transform_path is None or mean_path is None:
            raise ValueError("official PCA fingerprint requires both array paths")
        digest.update(_sha256(Path(transform_path)).encode())
        digest.update(_sha256(Path(mean_path)).encode())
    elif mode == "simulation_fitted":
        if artifact_path is None:
            raise ValueError("simulation PCA fingerprint requires artifact path")
        manifest = json.loads((Path(artifact_path) / "manifest.json").read_text())
        digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    else:
        raise ValueError("unknown PCA mode")
    return digest.hexdigest()


class SimulationFittedGelSightPCA:
    """Per-pad PCA fitted only from ContactBench training marker data."""

    FORMAT_VERSION = 1

    def __init__(self, artifact_dir: str | Path):
        self.artifact_dir = Path(artifact_dir)
        manifest_path = self.artifact_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if self.manifest.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("unsupported simulation PCA artifact version")
        if self.manifest.get("mode") != "simulation_fitted":
            raise ValueError("PCA artifact is not simulation_fitted")
        contract = self.manifest.get("marker_contract", {})
        expected = marker_metadata(official_pca_compatible=False)
        for key in ("shape", "components", "image_size_px", "marker_ordering"):
            if contract.get(key) != expected[key]:
                raise ValueError(f"simulation PCA marker contract mismatch: {key}")
        if self.manifest.get("normalization") != {
                "x": "divide_by_image_width_px",
                "y": "divide_by_image_height_px"}:
            raise ValueError("simulation PCA normalization contract mismatch")
        self.fingerprint = pca_basis_fingerprint(
            mode="simulation_fitted", artifact_path=self.artifact_dir)
        self.means, self.W = [], []
        hashes = self.manifest.get("sha256", {})
        for side in ("left", "right"):
            mean_path = self.artifact_dir / f"{side}_mean.npy"
            transform_path = self.artifact_dir / f"{side}_transform.npy"
            for value in (mean_path, transform_path):
                if hashes.get(value.name) != _sha256(value):
                    raise ValueError(f"simulation PCA checksum mismatch: {value.name}")
            mean = np.load(mean_path, allow_pickle=False)
            transform = np.load(transform_path, allow_pickle=False)
            if mean.shape != (MARKER_COUNT * 2,) or transform.shape != (MARKER_COUNT * 2, 15):
                raise ValueError(f"invalid simulation PCA arrays for {side}")
            self.means.append(mean.astype(np.float32))
            self.W.append(transform.astype(np.float32))
        self.means = np.stack(self.means)
        self.W = np.stack(self.W)

    def transform(self, displacement_px: np.ndarray,
                  *, image_width_px: int, image_height_px: int) -> np.ndarray:
        value = np.asarray(displacement_px, dtype=np.float32)
        if value.shape[-3:] != (2, MARKER_COUNT, 2):
            raise ValueError("dual marker displacement must end in [2,63,2]")
        manifest_size = self.manifest["marker_contract"]["image_size_px"]
        if [int(image_width_px), int(image_height_px)] != manifest_size:
            raise ValueError("runtime image size differs from simulation PCA fit")
        normalized = value.copy()
        normalized[..., 0] /= float(image_width_px)
        normalized[..., 1] /= float(image_height_px)
        flat = normalized.reshape(*normalized.shape[:-2], MARKER_COUNT * 2)
        return np.einsum("...sd,sdk->...sk", flat - self.means, self.W).astype(np.float32)


def load_gelsight_pca(*, mode: str, marker_contract: dict | None = None,
                      transform_path: str | Path | None = None,
                      mean_path: str | Path | None = None,
                      artifact_path: str | Path | None = None):
    """Single PCA construction point shared by training and deployment."""
    if mode == "official":
        if transform_path is None or mean_path is None or marker_contract is None:
            raise ValueError("official PCA requires transform, mean and certified marker contract")
        return OfficialGelSightPCA(
            transform_path, mean_path, marker_contract=marker_contract)
    if mode == "simulation_fitted":
        if artifact_path is None:
            raise ValueError("simulation_fitted PCA requires an artifact directory")
        return SimulationFittedGelSightPCA(artifact_path)
    raise ValueError("pca_mode must be official or simulation_fitted")
