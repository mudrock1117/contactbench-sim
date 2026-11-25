"""GelSight marker-displacement contract and official 15-D PCA.

Pressure rasters are intentionally rejected.  Marker positions are derived
from Hydrogel's physical shear/normal-displacement state; no random or
force-filled surrogate is accepted.
"""
from __future__ import annotations
from pathlib import Path
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
