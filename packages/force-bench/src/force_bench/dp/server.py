from __future__ import annotations

import argparse
import json
import socket

import numpy as np
import torch

from .checkpoint import load_policy
from .transport import receive_arrays, send_arrays


def main(argv=None):
    parser = argparse.ArgumentParser(description="Diffusion Policy inference server")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    model, config, checkpoint = load_policy(args.checkpoint, args.device)
    print(json.dumps({"event": "ready", "host": args.host, "port": args.port,
                      "step": checkpoint["step"], "camera": config.camera}), flush=True)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((args.host, args.port)); listener.listen(1)
        while True:
            connection, address = listener.accept()
            with connection:
                print(json.dumps({"event": "connected", "peer": address[0]}), flush=True)
                while True:
                    try:
                        request = receive_arrays(connection)
                    except ConnectionError:
                        break
                    try:
                        proprio = torch.as_tensor(request["proprio"], device=args.device).float()
                        if proprio.ndim == 2:
                            proprio = proprio.unsqueeze(0)
                        images = {}
                        for name in config.camera_names:
                            image = torch.as_tensor(request[f"rgb_{name}"], device=args.device)
                            images[name] = image.unsqueeze(0) if image.ndim == 4 else image
                        with torch.inference_mode():
                            actions = model.predict_actions(proprio, images)[0].cpu().numpy()
                        send_arrays(connection, {"actions": actions})
                    except Exception as error:
                        send_arrays(connection, {"error": np.frombuffer(
                            f"{type(error).__name__}: {error}".encode(), dtype=np.uint8)})


if __name__ == "__main__":
    raise SystemExit(main())
