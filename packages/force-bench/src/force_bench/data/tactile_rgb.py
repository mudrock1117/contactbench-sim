"""Opt-in physical Hydrogel/Taxim tactile RGB episode capture (no display required)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

PAD_ORDER = ("gel_left", "gel_right")
SENSOR_ID = "GelSightMini"


class TactileRGBCapture:
    """Render real gel indentation through the calibrated Taxim polynomial renderer."""

    def __init__(self, rig, output, episode_id, *, fps=24.0):
        if not np.isfinite(fps) or fps <= 0 or fps > float(rig.fps):
            raise ValueError("tactile RGB FPS must be finite, positive, and <= physics FPS")
        sensors = getattr(rig, "gel_sensors", {})
        frames = getattr(rig, "tactile_frames", {})
        if any(name not in sensors or name not in frames for name in PAD_ORDER):
            raise RuntimeError("left/right Hydrogel GelSight sensor frames are required")
        # This is the repository's optical sensor path: normal-displacement raster
        # -> raster_to_image (GS Mini orientation) -> calibrated Taxim RGB.
        from hydrogel.render import TaximRenderer, raster_to_image
        self.renderer = TaximRenderer()
        self._raster_to_image = raster_to_image
        self.rig = rig
        self.fps = float(fps)
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=".tactile-rgb-", dir=self.output))
        self.images = {pad: self.staging / pad / "images" for pad in PAD_ORDER}
        for directory in self.images.values():
            directory.mkdir(parents=True)
        self.timestamps = {pad: [] for pad in PAD_ORDER}
        self._frame_counts = {pad: 0 for pad in PAD_ORDER}
        self._closed = False
        self.metadata = {
            "enabled": True, "fps": self.fps, "timebase": "simulation_seconds",
            "renderer": "hydrogel.render.TaximRenderer.render",
            "input": "HydroelasticGelSight TactileFrame.normal_displacement_m",
            "optical_pipeline": "raster_to_image -> polycalib.npz + bg.png Taxim shading",
            "sensor_id": SENSOR_ID, "pad_order": list(PAD_ORDER),
            "images": "rgb/tactile/<pad>/images/%06d.png",
            "timestamps": "rgb/tactile/<pad>/timestamps.npy",
            "resize": "native GelSight Mini 240x320 RGB; official Zarr loader resizes to 224x224",
            "headless": True,
        }

    def _due(self, timestamp):
        next_frame = len(self.timestamps[PAD_ORDER[0]]) + 1
        return timestamp + 1e-12 >= next_frame / self.fps

    def capture(self, timestamp):
        if self._closed:
            raise RuntimeError("tactile RGB capture is closed")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp):
            raise ValueError("tactile RGB timestamp must be finite")
        # Align both pads at the same completed physics step. Sampling uses the
        # current HydroelasticGelSight output, never a future frame or a pressure map.
        if not self._due(timestamp):
            return False
        for pad in PAD_ORDER:
            frames = self.rig.tactile_frames[pad]
            if not frames or len(frames) <= self._frame_counts[pad]:
                raise RuntimeError(f"no new Hydrogel displacement frame for {pad}")
            frame = frames[-1]
            indent = self._raster_to_image(np.asarray(frame.normal_displacement_m))
            image = np.asarray(self.renderer.render(indent)).copy()
            if image.dtype != np.uint8 or image.shape != (240, 320, 3):
                raise ValueError(f"Taxim returned invalid RGB for {pad}: {image.shape} {image.dtype}")
            import imageio.v2 as iio
            index = len(self.timestamps[pad])
            iio.imwrite(self.images[pad] / f"{index:06d}.png", image)
            self.timestamps[pad].append(timestamp)
            self._frame_counts[pad] = len(frames)
        return True

    def finish(self):
        if self._closed:
            raise RuntimeError("tactile RGB capture is closed")
        for pad in PAD_ORDER:
            if not self.timestamps[pad]:
                raise RuntimeError(f"no tactile RGB frames captured for {pad}")
            folder = self.staging / pad
            np.save(folder / "timestamps.npy", np.asarray(self.timestamps[pad], dtype=np.float64),
                    allow_pickle=False)
            paths = [folder / "timestamps.npy", *sorted(self.images[pad].glob("*.png"))]
            (folder / "manifest.json").write_text(json.dumps({
                "files_sha256": {p.relative_to(folder).as_posix():
                    hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
            }, indent=2) + "\n")
        self._closed = True
        return self.staging

    def abort(self):
        if self.staging.parent == self.output and self.staging.name.startswith(".tactile-rgb-"):
            shutil.rmtree(self.staging, ignore_errors=True)
        self._closed = True
