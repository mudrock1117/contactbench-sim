"""Process-isolated official RDP inference server."""
from __future__ import annotations
import argparse
import json
import socket
import numpy as np
import torch
from force_bench.dp.transport import receive_arrays, send_arrays
from .official_predict import _load


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args(argv)
    ws, policy = _load(a.config, a.checkpoint, a.device)
    obs_steps = int(ws.cfg.n_obs_steps)
    horizon = int(ws.cfg.horizon)
    ratio = int(ws.cfg.dataset_obs_temporal_downsample_ratio)
    print(json.dumps({
        "event": "ready",
        "implementation": "xiaoxiaoxh/reactive_diffusion_policy@8ed82a3",
        "slow_policy_hz": 12, "fast_action_hz": 24,
        "execute_actions_per_plan": 2,
    }), flush=True)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((a.host, a.port))
        listener.listen(1)
        while True:
            conn, _ = listener.accept()
            with conn:
                latent = None
                while True:
                    try:
                        request = receive_arrays(conn)
                    except ConnectionError:
                        break
                    try:
                        mode = bytes(np.asarray(request["mode"], np.uint8)).decode()
                        if mode == "plan":
                            wrist = np.asarray(request["rgb_wrist"])
                            agent = np.asarray(request["agent_pos"])
                            tactile = np.asarray(request["tactile_marker_pca"])
                            if wrist.shape != (obs_steps, 3, 240, 320):
                                raise ValueError(f"rgb_wrist has shape {wrist.shape}")
                            if agent.shape != (obs_steps, 10):
                                raise ValueError(f"agent_pos has shape {agent.shape}")
                            if tactile.shape != (obs_steps, 30):
                                raise ValueError(f"tactile PCA has shape {tactile.shape}")
                            obs = {
                                "wrist_rgb": torch.as_tensor(
                                    wrist, device=a.device).float().unsqueeze(0) / 255.0,
                                "agent_pos": torch.as_tensor(
                                    agent, device=a.device).float().unsqueeze(0),
                                "tactile_marker_pca": torch.as_tensor(
                                    tactile, device=a.device).float().unsqueeze(0),
                            }
                            with torch.inference_mode():
                                result = policy.predict_action(
                                    obs, dataset_obs_temporal_downsample_ratio=ratio,
                                    return_latent_action=True)
                            latent = result["action_pred"][:, 0].detach()
                            send_arrays(conn, {"planned": np.ones(1, np.uint8)})
                        elif mode == "decode":
                            if latent is None:
                                raise RuntimeError("decode requested before plan")
                            extended = np.asarray(request["extended_tactile_marker_pca"])
                            if extended.ndim != 2 or extended.shape[1] != 30:
                                raise ValueError(f"extended tactile has shape {extended.shape}")
                            if not 1 <= len(extended) <= horizon:
                                raise ValueError("extended tactile length is outside the chunk")
                            ext = {"tactile_marker_pca": torch.as_tensor(
                                extended, device=a.device).float().unsqueeze(0)}
                            with torch.inference_mode():
                                result = policy.predict_from_latent_action(
                                    latent, extended_obs_dict=ext,
                                    extended_obs_last_step=len(extended),
                                    dataset_obs_temporal_downsample_ratio=ratio,
                                    extend_obs_pad_after=True)
                            offset = len(extended) - int(ws.cfg.dataset_obs_steps)
                            action = result["action"][0].cpu().numpy()
                            if offset < 0 or offset >= len(action):
                                raise RuntimeError("reactive action offset is outside prediction")
                            send_arrays(conn, {"actions": action[offset:offset + 1]})
                        else:
                            raise ValueError(f"unknown request mode {mode!r}")
                    except Exception as error:
                        send_arrays(conn, {"error": np.frombuffer(
                            f"{type(error).__name__}: {error}".encode(),
                            dtype=np.uint8)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
