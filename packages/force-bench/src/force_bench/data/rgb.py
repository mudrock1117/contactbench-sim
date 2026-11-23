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
from force_bench.geometry import world_T

from .episode import atomic_json

CAMERA_NAME = "fixed"


def _validate_rigid_transform(transform, name):
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-10):
        raise ValueError(f"{name} must have homogeneous last row [0,0,0,1]")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)):
        raise ValueError(f"{name} rotation must be right-handed and orthonormal")
    return transform


def camera_lookat_matrix(position, target, up=(0.0, 1.0, 0.0)):
    """Return ``T_parent_camera`` for the renderer's camera convention.

    Camera ``+X`` is image-right, ``+Y`` is image-up and optical forward is
    ``-Z``.  ``position``, ``target`` and ``up`` are expressed in the same
    parent frame.  This is a calibration helper only; it is never evaluated
    against a task object during capture.
    """
    position, target, up = (np.asarray(value, dtype=float)
                            for value in (position, target, up))
    forward = target - position
    if position.shape != (3,) or target.shape != (3,) or up.shape != (3,):
        raise ValueError("look-at position, target, and up must have shape [3]")
    if np.linalg.norm(forward) < 1e-9 or np.linalg.norm(up) < 1e-9:
        raise ValueError("look-at direction and up vector must be nonzero")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("look-at up vector is parallel to the optical axis")
    right /= np.linalg.norm(right)
    image_up = np.cross(right, forward)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((right, image_up, -forward))
    transform[:3, 3] = position
    return transform


def wrist_mount_from_tcp(T_link_tcp, position_tcp, target_tcp, *, up_tcp=(0.0, 1.0, 0.0)):
    """Build a rigid link-to-camera calibration via the FR3 TCP frame."""
    T_link_tcp = _validate_rigid_transform(T_link_tcp, "T_link_tcp")
    return _validate_rigid_transform(
        T_link_tcp @ camera_lookat_matrix(position_tcp, target_tcp, up_tcp),
        "T_link_camera")


def _intrinsics(width, height, fov_deg):
    fy = 0.5 * height / np.tan(0.5 * np.radians(fov_deg))
    return {"model": "pinhole", "fx": float(fy), "fy": float(fy),
            "cx": 0.5 * width, "cy": 0.5 * height,
            "vertical_fov_deg": float(fov_deg)}


def _positive_int(environment, name, default):
    value = int(environment.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def rgb_config_from_env(environment=None):
    """Return disabled/validated RGB settings without importing a renderer."""
    environment = os.environ if environment is None else environment
    selection = environment.get("TB_COLLECT_RGB_CAMERAS", "").strip().lower()
    enabled = environment.get("TB_COLLECT_RGB", "0") == "1" or selection not in ("", "none")
    if environment.get("TB_COLLECT_RGB", "0") not in ("0", "1"):
        raise ValueError("TB_COLLECT_RGB must be 0 or 1")
    if not enabled:
        return {"enabled": False}
    selection = selection or "fixed"
    if selection not in ("fixed", "wrist", "both"):
        raise ValueError("TB_COLLECT_RGB_CAMERAS must be none, fixed, wrist, or both")
    config = {"enabled": True,
              "fps": _positive_int(environment, "TB_COLLECT_RGB_FPS", 12),
              "width": _positive_int(environment, "TB_COLLECT_RGB_WIDTH", 320),
              "height": _positive_int(environment, "TB_COLLECT_RGB_HEIGHT", 240)}
    # Keep the exact legacy fixed-only dictionary for existing callers/tests.
    if environment.get("TB_COLLECT_RGB_CAMERAS", "").strip():
        config["cameras"] = ["fixed", "wrist"] if selection == "both" else [selection]
    return config


class FixedRGBCapture:
    """Stream fixed-camera PNGs aligned to post-physics simulation times."""

    def __init__(self, rig, output, episode_id, camera, config, *, camera_factory=TeleopCamera,
                 camera_name=CAMERA_NAME):
        self.rig = rig
        self.output = Path(output)
        self.fps = float(config["fps"])
        if self.fps > float(rig.fps):
            raise ValueError("TB_COLLECT_RGB_FPS cannot exceed the simulation state rate")
        self.width, self.height = int(config["width"]), int(config["height"])
        self.output.mkdir(parents=True, exist_ok=True)
        self.name = str(camera_name)
        self.staging = Path(tempfile.mkdtemp(prefix=f".rgb-{self.name}-{episode_id}-", dir=self.output))
        self.images = self.staging / "images"
        self.images.mkdir()
        self.timestamps = []
        self.poses = []
        self._next_index = 0
        self._closed = False
        try:
            pos = np.asarray(camera["position"], dtype=float)
            target = np.asarray(camera["target"], dtype=float)
            fov = float(camera.get("fov_deg", 40.0))
            self.camera = camera_factory(rig.model, pos, target, width=self.width,
                                         height=self.height, fov_deg=fov)
            self.parameters = {
                "name": self.name,
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
                "intrinsics": _intrinsics(self.width, self.height, fov),
                "extrinsics": "world_from_camera xyzw; poses.npy row [x,y,z,qx,qy,qz,qw]",
            }
            self.capture(rig.s0, 0.0, force=True)
        except BaseException:
            self.abort()
            raise

    @property
    def metadata(self):
        return {"camera_names": [self.name], "camera_type": self.parameters["mount"],
                "fps": self.fps, "timebase": "simulation_seconds",
                "timestamps": f"rgb/{self.name}/timestamps.npy",
                "poses": f"rgb/{self.name}/poses.npy",
                "parameters": f"rgb/{self.name}/camera.json",
                "manifest": f"rgb/{self.name}/manifest.json",
                "images": f"rgb/{self.name}/images/%06d.png",
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
        T = self.camera.transform() if hasattr(self.camera, "transform") else np.eye(4)
        from force_bench.teleop.camera import _mat_to_quat
        quat = _mat_to_quat(T[:3, :3])
        if self.poses and np.dot(self.poses[-1][3:], quat) < 0:
            quat = -quat
        self.poses.append(np.r_[T[:3, 3], quat])
        self._next_index += 1
        while self._next_index / self.fps <= timestamp + 1e-12:
            self._next_index += 1
        return True

    def finish(self):
        if self._closed:
            raise RuntimeError("RGB capture is closed")
        np.save(self.staging / "timestamps.npy", np.asarray(self.timestamps, dtype=np.float64),
                allow_pickle=False)
        np.save(self.staging / "poses.npy", np.asarray(self.poses, dtype=np.float64), allow_pickle=False)
        atomic_json(self.staging / "camera.json", self.parameters)
        files = [self.staging / "timestamps.npy", self.staging / "poses.npy",
                 self.staging / "camera.json",
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


class WristRGBCapture(FixedRGBCapture):
    """RGB camera rigidly attached to an actual simulated link pose.

    ``T_world_camera = T_world_link @ T_link_camera`` is evaluated from
    ``state.body_q`` immediately before every rendered frame.
    """

    def __init__(self, rig, output, episode_id, camera, config, *, camera_factory=TeleopCamera):
        self.link_label = camera.get("link_label", "fr3/fr3_link7")
        self.link_index = list(rig.model.body_label).index(self.link_label)
        if "T_link_camera" in camera:
            self.T_link_camera = np.asarray(camera["T_link_camera"], dtype=float)
        else:
            self.T_link_camera = camera_lookat_matrix(
                camera.get("position_link", [0.04, 0.0, 0.04]),
                camera.get("target_link", [0.12, 0.0, 0.0]),
                camera.get("up_link", [0.0, 0.0, 1.0]))
        if self.T_link_camera.shape != (4, 4):
            raise ValueError("T_link_camera must be 4x4")
        self.T_link_camera = _validate_rigid_transform(self.T_link_camera, "T_link_camera")
        T_world_link = world_T(rig.s0.body_q.numpy()[self.link_index])
        T_world_camera = T_world_link @ self.T_link_camera
        initial = {"position": T_world_camera[:3, 3],
                   "target": T_world_camera[:3, 3] - T_world_camera[:3, 2],
                   "fov_deg": camera.get("fov_deg", 60.0)}
        super().__init__(rig, output, episode_id, initial, config, camera_factory=camera_factory,
                         camera_name="wrist")
        self.parameters.update({"mount": "wrist_link", "link_label": self.link_label,
                                "T_link_camera": self.T_link_camera.tolist(),
                                "transform_rule": "T_world_camera = T_world_link @ T_link_camera",
                                **{key: (np.asarray(camera[key], dtype=float).tolist()
                                         if key in ("position_tcp", "target_tcp", "up_tcp")
                                         else camera[key])
                                   for key in ("mount_frame", "position_tcp", "target_tcp",
                                               "up_tcp", "optical_axis") if key in camera}})
        # Replace the look-at initial frame with the exact mount transform so
        # link roll is preserved from frame zero.
        for path in self.images.glob("*.png"):
            path.unlink()
        self.timestamps, self.poses, self._next_index = [], [], 0
        self.capture(rig.s0, 0.0, force=True)

    def capture(self, state, timestamp, *, force=False):
        if not force and not self._due(float(timestamp)):
            return False
        T_world_link = world_T(state.body_q.numpy()[self.link_index])
        self.camera.set_transform(T_world_link @ self.T_link_camera)
        return super().capture(state, timestamp, force=True)


class LiveRGBCamera:
    """Latest-frame live camera using the same definitions as collection.

    The image is sampled on the simulator clock.  ``observe`` never renders a
    future frame and may therefore return the same frame at consecutive policy
    observations.
    """

    def __init__(self, rig, name, camera, *, fps=12.0, width=320, height=240,
                 camera_factory=TeleopCamera):
        if name not in ("fixed", "wrist"):
            raise ValueError("live RGB camera must be fixed or wrist")
        self.rig, self.name = rig, name
        self.fps, self.width, self.height = float(fps), int(width), int(height)
        if self.fps <= 0 or self.fps > float(rig.fps):
            raise ValueError("RGB frequency must be positive and no greater than physics_hz")
        self.link_index = None
        self.T_link_camera = None
        fov = float(camera.get("fov_deg", 60.0 if name == "wrist" else 40.0))
        if name == "fixed":
            pos, target = np.asarray(camera["position"], float), np.asarray(camera["target"], float)
            mount = "fixed_world"
        else:
            link_label = camera.get("link_label", "fr3/fr3_link7")
            self.link_index = list(rig.model.body_label).index(link_label)
            if "T_link_camera" in camera:
                self.T_link_camera = _validate_rigid_transform(
                    camera["T_link_camera"], "T_link_camera")
            else:
                self.T_link_camera = camera_lookat_matrix(
                    camera.get("position_link", [0.04, 0, 0.04]),
                    camera.get("target_link", [0.12, 0, 0]),
                    camera.get("up_link", [0.0, 0.0, 1.0]))
            T = world_T(rig.s0.body_q.numpy()[self.link_index]) @ self.T_link_camera
            pos, target = T[:3, 3], T[:3, 3] - T[:3, 2]
            mount = "wrist_link"
        self.camera = camera_factory(rig.model, pos, target, width=self.width,
                                     height=self.height, fov_deg=fov)
        self.calibration = {"name": name, "mount": mount, "width": self.width,
                            "height": self.height, "fov_deg": fov,
                            "color_space": "RGB", "dtype": "uint8", "policy_layout": "CHW",
                            "intrinsics": _intrinsics(self.width, self.height, fov),
                            "T_link_camera": (None if self.T_link_camera is None
                                               else self.T_link_camera.tolist()),
                            "transform_rule": ("constant T_world_camera" if name == "fixed" else
                                               "T_world_camera = T_world_link @ T_link_camera")}
        if name == "wrist":
            self.calibration.update({
                key: (np.asarray(camera[key], dtype=float).tolist()
                      if key in ("position_tcp", "target_tcp", "up_tcp") else camera[key])
                for key in ("mount_frame", "position_tcp", "target_tcp", "up_tcp",
                            "optical_axis") if key in camera})
        self._next_index, self.image, self.timestamp, self.pose = 0, None, None, None

    def _update_pose(self, state):
        if self.link_index is not None:
            T = world_T(state.body_q.numpy()[self.link_index]) @ self.T_link_camera
            self.camera.set_transform(T)

    def sample(self, state, timestamp, *, force=False):
        timestamp = float(timestamp)
        if not force and timestamp + 1e-12 < self._next_index / self.fps:
            return False
        self._update_pose(state)
        image = np.asarray(self.camera.render(state)).copy()
        if image.dtype != np.uint8 or image.shape != (self.height, self.width, 3):
            raise ValueError("camera must return configured uint8 RGB")
        self.image = image.transpose(2, 0, 1).copy()
        self.timestamp = timestamp
        self.pose = self.camera.transform().copy()
        self._next_index += 1
        while self._next_index / self.fps <= timestamp + 1e-12:
            self._next_index += 1
        return True

    def observe(self, timestamp):
        valid = self.image is not None and self.timestamp <= float(timestamp) + 1e-12
        return {"rgb": None if self.image is None else self.image.copy(),
                "rgb_timestamp": np.float64(np.nan if self.timestamp is None else self.timestamp),
                "rgb_valid": bool(valid),
                "rgb_age_s": np.float64(np.nan if self.timestamp is None else timestamp - self.timestamp),
                "camera_name": self.name, "calibration": dict(self.calibration),
                "T_world_camera": None if self.pose is None else self.pose.copy()}


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
    manifest = json.loads((directory / "manifest.json").read_text())
    expected_hashes = manifest.get("files_sha256", {})
    poses_path = directory / "poses.npy"
    hashed_files = [directory / "timestamps.npy", *([poses_path] if poses_path.exists() else []),
                    directory / "camera.json", *files]
    if set(expected_hashes) != {
            str(path.relative_to(directory)).replace("\\", "/") for path in hashed_files}:
        raise ValueError("RGB manifest does not match the camera files")
    for path in hashed_files:
        name = str(path.relative_to(directory)).replace("\\", "/")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hashes[name]:
            raise ValueError(f"RGB checksum mismatch: {name}")
    shape = (int(parameters["height"]), int(parameters["width"]), 3)
    for path in files:
        image = iio.imread(path)
        if image.dtype != np.uint8 or image.shape != shape:
            raise ValueError(f"invalid RGB image {path.name}")
    if parameters.get("mount") != "fixed_world" or parameters.get("color_space") != "RGB":
        if parameters.get("mount") != "wrist_link" or parameters.get("color_space") != "RGB":
            raise ValueError("unsupported RGB camera mount or color space")
    if poses_path.exists():
        poses = np.load(poses_path, allow_pickle=False)
        if poses.dtype != np.float64 or poses.shape != (len(timestamps), 7) or not np.isfinite(poses).all():
            raise ValueError("camera poses must be finite float64 [N,7]")
        if not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1.0, atol=1e-6):
            raise ValueError("camera pose quaternions must be normalized")
        if len(poses) > 1 and np.any(np.sum(poses[:-1, 3:] * poses[1:, 3:], axis=1) < -1e-9):
            raise ValueError("camera quaternion sign flips are forbidden")
        if parameters.get("mount") == "fixed_world" and not np.allclose(poses, poses[:1], atol=1e-12):
            raise ValueError("fixed-world camera extrinsics changed")
    elif parameters.get("mount") == "wrist_link":
        raise ValueError("wrist RGB requires per-frame poses.npy")
    return len(files)


def validate_rgb_episode(path, metadata, state_timestamps=None):
    rgb = metadata.get("rgb")
    if rgb is None:
        return
    names = rgb.get("camera_names")
    if not isinstance(names, list) or not names or any(name not in ("fixed", "wrist") for name in names):
        raise ValueError("unsupported RGB camera metadata")
    for name in names:
        directory = Path(path) / "rgb" / name
        validate_rgb_directory(directory)
        if state_timestamps is not None:
            rgb_timestamps = np.load(directory / "timestamps.npy", allow_pickle=False)
            state_timestamps = np.asarray(state_timestamps, dtype=np.float64)
            indices = np.searchsorted(state_timestamps, rgb_timestamps)
            if np.any(indices >= len(state_timestamps)) or not np.allclose(
                    state_timestamps[indices], rgb_timestamps, rtol=0.0, atol=1e-12):
                raise ValueError("RGB timestamps are not aligned to simulator state timestamps")
