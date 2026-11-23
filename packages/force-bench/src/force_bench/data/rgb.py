"""Optional fixed-world RGB observations for demonstration episodes.

Frames are written immediately into a hidden staging directory.  The
directory is moved into the episode by :class:`EpisodeWriter` only after all
files pass validation, so interrupted capture is never discoverable as a
complete camera stream.
"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path
import shutil
import tempfile

import imageio.v3 as iio
import numpy as np

from force_bench.teleop.camera import TeleopCamera

from .episode import atomic_json

CAMERA_NAME = "fixed"


def _positive_int(environment, name, default):
    value = int(environment.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def rgb_config_from_env(environment=None):
    """Return disabled/validated RGB settings without importing a renderer."""
    environment = os.environ if environment is None else environment
    enabled = environment.get("TB_COLLECT_RGB", "0") == "1"
    if environment.get("TB_COLLECT_RGB", "0") not in ("0", "1"):
        raise ValueError("TB_COLLECT_RGB must be 0 or 1")
    if not enabled:
        return {"enabled": False}
    return {"enabled": True,
            "fps": _positive_int(environment, "TB_COLLECT_RGB_FPS", 12),
            "width": _positive_int(environment, "TB_COLLECT_RGB_WIDTH", 320),
            "height": _positive_int(environment, "TB_COLLECT_RGB_HEIGHT", 240)}


class FixedRGBCapture:
    """Stream fixed-camera PNGs aligned to post-physics simulation times."""

    def __init__(self, rig, output, episode_id, camera, config, *, camera_factory=TeleopCamera):
        self.rig = rig
        self.output = Path(output)
        self.fps = float(config["fps"])
        if self.fps > float(rig.fps):
            raise ValueError("TB_COLLECT_RGB_FPS cannot exceed the simulation state rate")
        self.width, self.height = int(config["width"]), int(config["height"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.staging = Path(tempfile.mkdtemp(prefix=f".rgb-{episode_id}-", dir=self.output))
        self.images = self.staging / "images"
        self.images.mkdir()
        self.timestamps = []
        self._next_index = 0
        self._closed = False
        try:
            pos = np.asarray(camera["position"], dtype=float)
            target = np.asarray(camera["target"], dtype=float)
            fov = float(camera.get("fov_deg", 40.0))
            self.camera = camera_factory(rig.model, pos, target, width=self.width,
                                         height=self.height, fov_deg=fov)
            self.parameters = {
                "name": CAMERA_NAME,
                "mount": "fixed_world",
                "frame": "world",
                "position": pos.tolist(),
                "target": target.tolist(),
                "fov_deg": fov,
                "width": self.width,
                "height": self.height,
                "image_format": "png",
                "color_space": "RGB",
                "dtype": "uint8",
            }
            self.capture(rig.s0, 0.0, force=True)
        except BaseException:
            self.abort()
            raise

    @property
    def metadata(self):
        return {"camera_names": [CAMERA_NAME], "camera_type": "fixed_world",
                "fps": self.fps, "timebase": "simulation_seconds",
                "timestamps": f"rgb/{CAMERA_NAME}/timestamps.npy",
                "parameters": f"rgb/{CAMERA_NAME}/camera.json",
                "manifest": f"rgb/{CAMERA_NAME}/manifest.json",
                "images": f"rgb/{CAMERA_NAME}/images/%06d.png",
                "alignment": "initial frame at t=0; later frames after completed physics updates"}

    def _due(self, timestamp):
        return self._next_index == 0 or timestamp + 1e-12 >= self._next_index / self.fps

    def capture(self, state, timestamp, *, force=False):
        if self._closed:
            raise RuntimeError("RGB capture is closed")
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.timestamps and timestamp <= self.timestamps[-1]):
            raise ValueError("RGB timestamps must be finite and strictly increasing")
        if not force and not self._due(timestamp):
            return False
        # TeleopCamera returns a view of a reusable host buffer.  Copy before
        # handing it to an encoder that may retain or mutate the array.
        image = np.asarray(self.camera.render(state)).copy()
        if image.dtype != np.uint8 or image.shape != (self.height, self.width, 3):
            raise ValueError("camera must return uint8 RGB with configured height and width")
        iio.imwrite(self.images / f"{len(self.timestamps):06d}.png", image)
        self.timestamps.append(timestamp)
        self._next_index += 1
        while self._next_index / self.fps <= timestamp + 1e-12:
            self._next_index += 1
        return True

    def finish(self):
        if self._closed:
            raise RuntimeError("RGB capture is closed")
        np.save(self.staging / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float64),
                allow_pickle=False)
        atomic_json(self.staging / "camera.json", self.parameters)
        files = [self.staging / "timestamps.npy", self.staging / "camera.json",
                 *sorted(self.images.glob("*.png"))]
        atomic_json(self.staging / "manifest.json", {
            "files_sha256": {str(path.relative_to(self.staging)).replace("\\", "/"):
                              hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in files}})
        validate_rgb_directory(self.staging, self.parameters)
        self._closed = True
        return self.staging

    def abort(self):
        staging = getattr(self, "staging", None)
        if staging is not None and staging.parent == self.output and staging.name.startswith(".rgb-"):
            shutil.rmtree(staging, ignore_errors=True)
        self._closed = True


def validate_rgb_directory(directory, parameters=None):
    directory = Path(directory)
    parameters = parameters or json.loads((directory / "camera.json").read_text())
    timestamps = np.load(directory / "timestamps.npy", allow_pickle=False)
    if timestamps.dtype != np.float64 or timestamps.ndim != 1 or len(timestamps) == 0:
        raise ValueError("RGB timestamps must be a nonempty float64 vector")
    if timestamps[0] != 0.0 or (len(timestamps) > 1 and not (np.diff(timestamps) > 0).all()):
        raise ValueError("RGB timestamps must start at zero and increase strictly")
    files = sorted((directory / "images").glob("*.png"))
    expected = [f"{index:06d}.png" for index in range(len(timestamps))]
    if [path.name for path in files] != expected:
        raise ValueError("RGB image sequence does not match timestamps")
    shape = (int(parameters["height"]), int(parameters["width"]), 3)
    for path in files:
        image = iio.imread(path)
        if image.dtype != np.uint8 or image.shape != shape:
            raise ValueError(f"invalid RGB image {path.name}")
    if parameters.get("mount") != "fixed_world" or parameters.get("color_space") != "RGB":
        raise ValueError("RGB camera must be recorded as fixed_world RGB")
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_hashes = manifest.get("files_sha256", {})
    hashed_files = [directory / "timestamps.npy", directory / "camera.json", *files]
    if set(expected_hashes) != {
            str(path.relative_to(directory)).replace("\\", "/") for path in hashed_files}:
        raise ValueError("RGB manifest does not match the camera files")
    for path in hashed_files:
        name = str(path.relative_to(directory)).replace("\\", "/")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[name]:
            raise ValueError(f"RGB checksum mismatch: {name}")
    return len(files)


def validate_rgb_episode(path, metadata, state_timestamps=None):
    rgb = metadata.get("rgb")
    if rgb is None:
        return
    if rgb.get("camera_names") != [CAMERA_NAME] or rgb.get("camera_type") != "fixed_world":
        raise ValueError("unsupported RGB camera metadata")
    directory = Path(path) / "rgb" / CAMERA_NAME
    validate_rgb_directory(directory)
    if state_timestamps is not None:
        rgb_timestamps = np.load(directory / "timestamps.npy", allow_pickle=False)
        state_timestamps = np.asarray(state_timestamps, dtype=np.float64)
        indices = np.searchsorted(state_timestamps, rgb_timestamps)
        if np.any(indices >= len(state_timestamps)) or not np.allclose(
                state_timestamps[indices], rgb_timestamps, rtol=0.0, atol=1e-12):
            raise ValueError("RGB timestamps are not aligned to simulator state timestamps")

