"""Process-isolated official OpenPI pi0.5 inference server."""
from __future__ import annotations

import argparse
import json
import socket
import numpy as np

from force_bench.dp.transport import receive_arrays, send_arrays
from .config import DEFAULT_DATA, DEFAULT_MODEL_ROOT
from .dataset import PROMPT
from .inference import load_policy


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    _, policy = load_policy(
        a.profile, a.checkpoint, device=a.device,
        data_path=a.data, model_root=a.model_root)
    print(json.dumps({
        "event": "ready", "port": a.port,
        "implementation": "Physical-Intelligence/openpi@15a9616",
        "action_shape": [32, 10], "policy_hz": 24,
    }), flush=True)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((a.host, a.port))
        listener.listen(1)
        while True:
            conn, _ = listener.accept()
            with conn:
                while True:
                    try:
                        request = receive_arrays(conn)
                    except ConnectionError:
                        break
                    try:
                        wrist = np.asarray(request["rgb_wrist"], np.uint8)
                        state = np.asarray(request["state"], np.float32)
                        if wrist.shape != (3, 240, 320):
                            raise ValueError(f"rgb_wrist has shape {wrist.shape}")
                        if state.shape != (10,):
                            raise ValueError(f"state has shape {state.shape}")
                        result = policy.infer({
                            "wrist_rgb": wrist.transpose(1, 2, 0),
                            "state": state,
                            "prompt": PROMPT,
                        })
                        actions = np.asarray(result["actions"], np.float32)
                        if actions.shape != (32, 10) or not np.isfinite(actions).all():
                            raise ValueError(f"invalid actions {actions.shape}")
                        send_arrays(conn, {"actions": actions})
                    except Exception as error:
                        send_arrays(conn, {"error": np.frombuffer(
                            f"{type(error).__name__}: {error}".encode(),
                            dtype=np.uint8)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
