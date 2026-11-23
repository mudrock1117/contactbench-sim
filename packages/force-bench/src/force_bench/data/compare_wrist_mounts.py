"""Render rigid wrist-camera candidates from one recorded expert trajectory.

The tool never simulates or interpolates a state.  It rebuilds the recorded
task scene, assigns a complete ``body_q/body_qd`` row from
``artifacts/**/rollout_states.npz``, and renders every candidate from that
identical state.  An episode without this full-state artifact is rejected.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageDraw

from force_bench.geometry import world_T
from force_bench.teleop.camera import TeleopCamera

from .episode import atomic_json, load_episode
from .rgb import wrist_mount_from_tcp


DEFAULT_CANDIDATES = {
    "A": {"position": [0.055, 0.0, -0.035], "target": [0.0, 0.0, 0.110], "fov_deg": 72.0},
    "B": {"position": [0.080, 0.0, -0.035], "target": [0.0, 0.0, 0.110], "fov_deg": 72.0},
    "C": {"position": [0.080, 0.0, -0.035], "target": [0.0, 0.0, 0.140], "fov_deg": 72.0},
    "D": {"position": [0.080, 0.0, -0.035], "target": [0.0, 0.0, 0.140], "fov_deg": 80.0},
}


def _only(path, pattern, description):
    matches = sorted(Path(path).glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {description}; found {len(matches)}")
    return matches[0]


def load_full_states(episode_path, states_path=None):
    """Load numerical render state only; never unpickle recorder labels."""
    path = (Path(states_path) if states_path else
            _only(Path(episode_path) / "artifacts", "**/rollout_states.npz",
                  "artifacts/**/rollout_states.npz full-state artifact"))
    with np.load(path, allow_pickle=False) as archive:
        required = {"t", "body_q", "body_qd"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: full-state artifact is missing {sorted(required - set(archive.files))}")
        times = np.asarray(archive["t"], dtype=np.float64)
        body_q = np.asarray(archive["body_q"])
        body_qd = np.asarray(archive["body_qd"])
    if (times.ndim != 1 or len(times) == 0 or body_q.shape[:1] != times.shape
            or body_qd.shape[:1] != times.shape or body_q.shape[-1] != 7
            or body_qd.shape[-1] != 6 or not np.isfinite(body_q).all()
            or not np.isfinite(body_qd).all() or not np.isfinite(times).all()
            or (len(times) > 1 and not np.all(np.diff(times) > 0))):
        raise ValueError(f"{path}: invalid or incomplete renderer state arrays")
    return path, times, body_q, body_qd


def load_candidates(path=None):
    values = DEFAULT_CANDIDATES if path is None else json.loads(Path(path).read_text())
    if not isinstance(values, dict) or not values:
        raise ValueError("candidate JSON must be a nonempty name -> parameters object")
    result = {}
    for name, value in values.items():
        if not isinstance(value, dict) or set(value) < {"position", "target", "fov_deg"}:
            raise ValueError(f"candidate {name!r} requires position, target, and fov_deg")
        position, target = np.asarray(value["position"], float), np.asarray(value["target"], float)
        fov = float(value["fov_deg"])
        if (position.shape != (3,) or target.shape != (3,) or not np.isfinite(position).all()
                or not np.isfinite(target).all() or not 1.0 < fov < 179.0):
            raise ValueError(f"candidate {name!r} has invalid parameters")
        result[str(name)] = {"position": position, "target": target, "fov_deg": fov}
    return result


def _phase_samples(episode_path, state_times):
    try:
        path = _only(Path(episode_path) / "artifacts", "**/expert_log.npz",
                     "artifacts/**/expert_log.npz")
        with np.load(path, allow_pickle=False) as archive:
            log_t = np.asarray(archive["t"], float)
            phase = np.asarray(archive["ph"], int)
        if log_t.ndim != 1 or phase.shape != log_t.shape or len(log_t) == 0:
            raise ValueError("invalid expert phase timebase")
        from force_bench.appendix_tasks._allen_key import ORDER
        choices = (("approach", ("hover", "align")),
                   ("contact", ("insert", "reinsert")),
                   ("rotation", ("turn",)),
                   ("exit", ("extract", "retreat")))
        selected = []
        for label, names in choices:
            ids = [ORDER.index(name) for name in names]
            rows = np.flatnonzero(np.isin(phase, ids))
            if len(rows) == 0:
                raise ValueError(f"phase log has no {label} rows")
            timestamp = float(log_t[rows[len(rows) // 2]])
            state_index = int(np.argmin(np.abs(state_times - timestamp)))
            selected.append((label, state_index))
        return "expert_phase", selected
    except (OSError, KeyError, ValueError, ImportError):
        indices = np.linspace(0, len(state_times) - 1, 4).round().astype(int)
        return "uniform_time", [(f"time sample {i + 1}", int(index))
                                for i, index in enumerate(indices)]


def _diagnostic_points(episode_path, timestamps):
    """Allen-only privileged diagnostic points, never policy observations."""
    try:
        path = _only(Path(episode_path) / "artifacts", "**/expert_log.npz", "expert_log.npz")
        with np.load(path, allow_pickle=False) as archive:
            log_t = np.asarray(archive["t"], float)
            tip = np.column_stack([archive["Ex"], archive["Ey"], archive["Ez"]]).astype(float)
            rise = np.asarray(archive["z"], float)
        from force_bench.appendix_tasks._allen_key import RECESS_FLOOR_Z, SCREW_XY, Z0
        result = []
        for timestamp in timestamps:
            row = int(np.argmin(np.abs(log_t - timestamp)))
            result.append({"tool_tip": tip[row],
                           "screw_recess": np.r_[SCREW_XY, Z0 + RECESS_FLOOR_Z + rise[row]]})
        return result
    except (OSError, KeyError, ValueError, ImportError):
        return [{} for _ in timestamps]


@contextmanager
def _recorded_environment(metadata, output_root):
    recorded = metadata.get("physics", {}).get("environment", {})
    values = {str(key): str(value) for key, value in recorded.items()}
    values["TB_OUT_ROOT"] = str(output_root)
    previous = {key: os.environ.get(key) for key in values}
    try:
        os.environ.update(values)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _scene_for_episode(episode, output_root):
    task = episode.metadata.get("task")
    args = tuple(episode.metadata.get("collection", {}).get("task_arguments", ()))
    if not task:
        raise ValueError("episode metadata has no task name; scene cannot be reconstructed")
    from force_bench.teleop.scenes import load_scene
    with _recorded_environment(episode.metadata, output_root):
        return load_scene(task, args=args)


def _validate_recorded_wrist(episode_path, state_times, body_q, link_index):
    """Use the published wrist poses to verify restored link identity/order."""
    directory = Path(episode_path) / "rgb" / "wrist"
    required = [directory / "camera.json", directory / "timestamps.npy", directory / "poses.npy"]
    if not all(path.is_file() for path in required):
        return "unavailable: episode has no recorded wrist calibration/poses"
    parameters = json.loads(required[0].read_text())
    if "T_link_camera" not in parameters:
        return "unavailable: recorded wrist camera has no T_link_camera"
    T_link_camera = np.asarray(parameters["T_link_camera"], float)
    rgb_times = np.load(required[1], allow_pickle=False)
    poses = np.load(required[2], allow_pickle=False)
    if poses.shape != (len(rgb_times), 7):
        raise ValueError("recorded wrist poses do not match wrist timestamps")
    # A bounded spread across the trajectory catches a wrong body index while
    # keeping validation cheap for long episodes.
    available = np.flatnonzero((rgb_times >= state_times[0] - 1e-12)
                               & (rgb_times <= state_times[-1] + 1e-12))
    if len(available) == 0:
        raise ValueError("recorded wrist poses do not overlap rollout_states timebase")
    pick = np.linspace(0, len(available) - 1, min(32, len(available))).round().astype(int)
    rows = available[pick]
    for row in rows:
        state_index = int(np.searchsorted(state_times, rgb_times[row]))
        if state_index >= len(state_times) or not np.isclose(
                state_times[state_index], rgb_times[row], rtol=0.0, atol=1e-12):
            raise ValueError("recorded wrist timestamp is absent from rollout_states timebase")
        expected = world_T(body_q[state_index, link_index]) @ T_link_camera
        recorded = world_T(poses[row])
        if not np.allclose(expected, recorded, rtol=0.0, atol=2e-7):
            raise ValueError("reconstructed link7 does not reproduce recorded wrist poses")
    return f"passed on {len(rows)} recorded wrist poses"


def compare_wrist_mounts(episode_path, output, *, candidates_path=None, states_path=None,
                         width=320, height=240):
    episode_path, output = Path(episode_path), Path(output)
    episode = load_episode(episode_path)
    state_path, state_times, body_q, body_qd = load_full_states(episode_path, states_path)
    candidates = load_candidates(candidates_path)
    if int(width) < 1 or int(height) < 1:
        raise ValueError("width and height must be positive")
    selection, samples = _phase_samples(episode_path, state_times)
    timestamps = [float(state_times[index]) for _, index in samples]
    points = _diagnostic_points(episode_path, timestamps)

    with tempfile.TemporaryDirectory(prefix="contactbench-camera-scene-") as temporary:
        scene = _scene_for_episode(episode, temporary)
        rig = scene.rig
        labels = list(rig.model.body_label)
        if body_q.shape[1:] != (rig.model.body_count, 7) or body_qd.shape[1:] != (rig.model.body_count, 6):
            raise ValueError("recorded body arrays do not match the reconstructed scene topology")
        link_label = "fr3/fr3_link7"
        if link_label not in labels:
            raise ValueError("reconstructed scene has no fr3/fr3_link7")
        link_index = labels.index(link_label)
        wrist_validation = _validate_recorded_wrist(
            episode_path, state_times, body_q, link_index)
        T_link_tcp = (np.linalg.inv(world_T(rig.ik.body_pose(link_label)))
                      @ world_T(rig.ik.tcp_pose()))

        cameras = {}
        for name, candidate in candidates.items():
            T_link_camera = wrist_mount_from_tcp(
                T_link_tcp, candidate["position"], candidate["target"], up_tcp=(0, 1, 0))
            first_T = world_T(body_q[samples[0][1], link_index]) @ T_link_camera
            position = first_T[:3, 3]
            target = position - first_T[:3, 2]
            camera = TeleopCamera(rig.model, position, target, width=width, height=height,
                                  fov_deg=candidate["fov_deg"])
            cameras[name] = (camera, T_link_camera)

        rows, diagnostics = [], []
        for sample_number, ((sample_label, state_index), privileged) in enumerate(zip(samples, points)):
            rig.s0.body_q.assign(body_q[state_index])
            rig.s0.body_qd.assign(body_qd[state_index])
            cells, row_diagnostics = [], {}
            for name, candidate in candidates.items():
                camera, T_link_camera = cameras[name]
                camera.set_transform(world_T(body_q[state_index, link_index]) @ T_link_camera)
                image = np.asarray(camera.render(rig.s0)).copy()
                shape_ids = np.asarray(camera.sidx.numpy())[0, 0]
                cell = Image.new("RGB", (width, height + 42), "white")
                cell.paste(Image.fromarray(image), (0, 42))
                text = (f"{name}  t={state_times[state_index]:.3f}s  {sample_label}\n"
                        f"p={candidate['position'].tolist()} target={candidate['target'].tolist()} "
                        f"FOV={candidate['fov_deg']:g}")
                ImageDraw.Draw(cell).text((4, 3), text, fill="black")
                cells.append(cell)
                projected = {}
                for point_name, point in privileged.items():
                    pixels, front = camera.project(np.asarray(point)[None])
                    pixel = pixels[0]
                    projected[point_name] = {
                        "pixel_xy": [float(pixel[0]), float(pixel[1])],
                        "in_front": bool(front[0]),
                        "inside_image": bool(front[0] and 0 <= pixel[0] < width
                                             and 0 <= pixel[1] < height),
                    }
                    if projected[point_name]["inside_image"]:
                        x = int(np.clip(round(float(pixel[0])), 0, width - 1))
                        y = int(np.clip(round(float(pixel[1])), 0, height - 1))
                        shape_id = int(shape_ids[y, x])
                        if shape_id != 0xFFFFFFFF and shape_id < rig.model.shape_count:
                            body_id = int(rig.model.shape_body.numpy()[shape_id])
                            body_label = (str(rig.model.body_label[body_id])
                                          if body_id >= 0 else None)
                            projected[point_name].update(
                                center_ray_hit_shape=str(rig.model.shape_label[shape_id]),
                                center_ray_hit_body=body_label,
                                possible_robot_occlusion=bool(
                                    body_label and body_label.startswith("fr3/")))
                        else:
                            projected[point_name].update(
                                center_ray_hit_shape=None, center_ray_hit_body=None,
                                possible_robot_occlusion=False)
                row_diagnostics[name] = projected
            rows.append(cells)
            diagnostics.append({"label": sample_label, "timestamp": timestamps[sample_number],
                                "state_index": state_index, "projections": row_diagnostics})

    sheet = Image.new("RGB", (len(candidates) * width, len(rows) * (height + 42)), "white")
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            sheet.paste(cell, (column_index * width, row_index * (height + 42)))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    report = {
        "episode": str(episode_path), "full_state_artifact": str(state_path),
        "selection": selection, "image": str(output), "image_size": [width, height],
        "scene_identity_validation": wrist_validation,
        "candidates": {name: {"position_tcp": value["position"].tolist(),
                               "target_tcp": value["target"].tolist(),
                               "fov_deg": value["fov_deg"]}
                       for name, value in candidates.items()},
        "samples": diagnostics,
        "projection_scope": "offline privileged diagnostics only; not policy or training input",
        "occlusion": ("center-pixel shape IDs flag possible robot occlusion only; "
                      "projection and one pixel do not prove target visibility, so confirm rendered images"),
    }
    report_path = output.with_suffix(".json")
    atomic_json(report_path, report)
    return output, report_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("output")
    parser.add_argument("--candidates-json")
    parser.add_argument("--states", help="explicit rollout_states.npz when it is not inside episode/artifacts")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    args = parser.parse_args(argv)
    compare_wrist_mounts(args.episode, args.output, candidates_path=args.candidates_json,
                         states_path=args.states, width=args.width, height=args.height)


if __name__ == "__main__":
    main()
