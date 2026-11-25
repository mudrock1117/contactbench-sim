"""Closed-loop ContactBench client for the official slow/fast RDP policy."""
from __future__ import annotations
import argparse
import json
import socket
from pathlib import Path
import numpy as np

from force_bench.data.timebase import PolicyTimeConfig
from force_bench.dp.transport import receive_arrays, send_arrays
from force_bench.policy.env import ClosedLoopPolicyEnv, PolicyEnvConfig
from force_bench.policy.evaluation import CardRetrievalPolicyEvaluator
from .tactile import load_gelsight_pca, read_rig_markers
from .timebase import RDPTimeConfig


def _mode(value):
    return np.frombuffer(value.encode(), dtype=np.uint8)


def _response(conn, request):
    send_arrays(conn, request)
    result = receive_arrays(conn)
    if "error" in result:
        raise RuntimeError(bytes(np.asarray(result["error"], np.uint8)).decode())
    return result


def _marker_status(rig):
    marker, valid = read_rig_markers(rig)
    frames = getattr(rig, "tactile_frames", {})
    counts = {
        pad: len(frames.get(pad, ())) for pad in ("gel_left", "gel_right")}
    return marker, valid, counts


def _marker_embedding(env, pca):
    marker, valid, counts = _marker_status(env.rig)
    if not np.all(valid):
        raise RuntimeError(
            "both GelSight marker streams must be valid; "
            f"valid={valid.tolist()} frame_counts={counts}")
    return pca.transform(
        marker, image_width_px=240, image_height_px=320).reshape(30)


def _rgb_is_current(observation):
    valid = observation.get("rgb_valid", False)
    timestamps = observation.get("rgb_timestamp", np.nan)
    now = float(observation["timestamp"])
    if isinstance(valid, dict):
        return (all(bool(value) for value in valid.values()) and
                all(np.isclose(float(value), now, atol=1e-12, rtol=0.0)
                    for value in timestamps.values()))
    return (bool(valid) and
            np.isclose(float(timestamps), now, atol=1e-12, rtol=0.0))


def _initialize_marker_history(env, observation, pca, timing, max_frames):
    """Wait for real dual-pad frames and one timestamp-aligned RGB state."""
    def ready(rig, current):
        _, valid, _ = _marker_status(rig)
        return bool(np.all(valid) and _rgb_is_current(current))

    warmup_frames = 0
    if not ready(env.rig, observation):
        try:
            observation, warmup_frames = env.warmup_sensors(
                ready, max_physics_frames=max_frames)
        except TimeoutError as error:
            _, valid, counts = _marker_status(env.rig)
            raise RuntimeError(
                f"GelSight initialization timed out after {max_frames} "
                f"physics frames; valid={valid.tolist()} "
                f"frame_counts={counts}") from error
    marker = _marker_embedding(env, pca)
    history = [marker.copy() for _ in range(timing.dataset_obs_steps)]
    return observation, history, warmup_frames


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="retrieve_card_from_deck")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--max-fast-steps", type=int, default=600)
    p.add_argument("--sensor-warmup-frames", type=int, default=120)
    p.add_argument("--pca-mode", choices=("official", "simulation_fitted"),
                   default="simulation_fitted")
    p.add_argument("--pca-artifact")
    p.add_argument("--pca-transform")
    p.add_argument("--pca-mean")
    p.add_argument("--marker-contract",
                   help="JSON certification required only for official PCA")
    a = p.parse_args(argv)
    contract = (json.loads(Path(a.marker_contract).read_text())
                if a.marker_contract else None)
    pca = load_gelsight_pca(
        mode=a.pca_mode, artifact_path=a.pca_artifact,
        transform_path=a.pca_transform, mean_path=a.pca_mean,
        marker_contract=contract)
    timing = RDPTimeConfig()
    env_timing = PolicyTimeConfig(
        physics_hz=60, policy_hz=24,
        observation_horizon=timing.dataset_obs_steps,
        action_horizon=timing.action_horizon)
    env = ClosedLoopPolicyEnv(
        a.scene, evaluator=CardRetrievalPolicyEvaluator(),
        config=PolicyEnvConfig(
            control_hz=24, visual_mode="wrist", rgb_fps=12,
            rgb_width=320, rgb_height=240,
            observation_horizon=timing.dataset_obs_steps,
            action_type="eef_relative_10d", include_tactile=True,
            tactile_hz=24, time_config=env_timing,
            max_steps=a.max_fast_steps))
    conn = socket.create_connection((a.host, a.port))
    executed = replans = 0
    try:
        observation = env.reset()
        observation, marker_history, warmup_frames = _initialize_marker_history(
            env, observation, pca, timing, a.sensor_warmup_frames)
        print(json.dumps({
            "event": "sensors_ready",
            "warmup_physics_frames": warmup_frames,
            "policy_fast_steps": executed,
            "timestamp": float(observation["timestamp"]),
            "rgb_timestamp": float(observation["rgb_timestamp"]),
            "marker_valid": [True, True],
        }), flush=True)
        while not env.done:
            fast_history = observation["history"]
            slow = fast_history[::-2][::-1]
            wrist = np.stack([np.asarray(item["rgb"], np.uint8) for item in slow])
            if not all(bool(item["rgb_valid"]) for item in slow):
                raise RuntimeError("causal wrist image is invalid")
            agent = np.asarray(observation["eef_relative_history"], np.float32)[::-2][::-1]
            tactile = np.stack(marker_history[::-2][::-1]).astype(np.float32)
            _response(conn, {
                "mode": _mode("plan"),
                "pca_fingerprint": _mode(pca.fingerprint),
                "rgb_wrist": wrist, "agent_pos": agent,
                "tactile_marker_pca": tactile})
            replans += 1
            extended = list(marker_history)
            for fast_index in range(timing.execution_horizon):
                result = _response(conn, {
                    "mode": _mode("decode"),
                    "extended_tactile_marker_pca": np.stack(extended).astype(np.float32)})
                action = np.asarray(result["actions"], np.float32)
                if action.shape != (1, 10) or not np.isfinite(action).all():
                    raise RuntimeError(f"invalid reactive action {action.shape}")
                if fast_index == 0:
                    env.replan(action)
                    observation, _, terminated, truncated, info = env.step_next()
                else:
                    # Preserve the fixed anchor established for this slow chunk.
                    observation, _, terminated, truncated, info = env.step(action[0])
                executed += 1
                marker = _marker_embedding(env, pca)
                extended.append(marker)
                marker_history = (marker_history + [marker])[-timing.dataset_obs_steps:]
                print(json.dumps({
                    "event": "fast_step", "step": executed, "replan": replans,
                    "timestamp": float(observation["timestamp"]),
                    "action_success": info["action_success"],
                    "terminated": terminated, "truncated": truncated,
                }), flush=True)
                if terminated or truncated:
                    break
        print(json.dumps({
            "event": "finished", "fast_steps": executed, "replans": replans,
            "evaluation": env.evaluate()}), flush=True)
    finally:
        conn.close()
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
