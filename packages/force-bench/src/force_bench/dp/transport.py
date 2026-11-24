"""Length-prefixed NPZ transport between simulation and PyTorch environments."""
from __future__ import annotations

from io import BytesIO
import socket
import struct

import numpy as np


def _receive_exact(sock, size):
    chunks = []
    while size:
        chunk = sock.recv(size)
        if not chunk:
            raise ConnectionError("inference connection closed")
        chunks.append(chunk); size -= len(chunk)
    return b"".join(chunks)


def send_arrays(sock, arrays):
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    sock.sendall(struct.pack("!Q", len(payload)) + payload)


def receive_arrays(sock):
    size = struct.unpack("!Q", _receive_exact(sock, 8))[0]
    if size > 512 * 1024 * 1024:
        raise ValueError("inference message exceeds 512 MiB")
    with np.load(BytesIO(_receive_exact(sock, size)), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class InferenceClient:
    def __init__(self, host="127.0.0.1", port=8765, timeout=120.0):
        self.sock = socket.create_connection((host, int(port)), timeout=timeout)

    def predict(self, proprio, images):
        payload = {"proprio": np.asarray(proprio, dtype=np.float32)}
        payload.update({f"rgb_{name}": np.asarray(value, dtype=np.uint8)
                        for name, value in images.items()})
        send_arrays(self.sock, payload)
        response = receive_arrays(self.sock)
        if "error" in response:
            raise RuntimeError(bytes(response["error"]).decode())
        return np.asarray(response["actions"], dtype=np.float32)

    def close(self):
        self.sock.close()
