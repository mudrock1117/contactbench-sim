"""Persistent FTP-1 inference socket service."""
import argparse
import socket
import numpy as np
from force_bench.dp.transport import receive_arrays, send_arrays
from .official import load_policy, predict

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8768)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    wrapper = load_policy(a.checkpoint, a.device)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((a.host,a.port))
        listener.listen(1)
        print(f"FTP-1 ready on {a.host}:{a.port}", flush=True)
        while True:
            conn,_ = listener.accept()
            with conn:
                while True:
                    try:
                        req = receive_arrays(conn)
                    except ConnectionError:
                        break
                    try:
                        _, actions = predict(wrapper, req["rgb_wrist"], req["tactile_rgb"],
                                             float(req["width"].item()))
                        send_arrays(conn, {"actions":actions})
                    except Exception as error:
                        send_arrays(conn, {"error":np.frombuffer(str(error).encode(),dtype=np.uint8)})

if __name__ == "__main__":
    main()
