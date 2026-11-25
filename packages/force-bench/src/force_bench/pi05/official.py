"""Activate the immutable OpenPI checkout used by ContactBench B3."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import yaml

OPENPI_NAME = "openpi"
OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def repository_root() -> Path:
    return Path(__file__).resolve().parents[5]


def activate_openpi(*, verify: bool = True) -> Path:
    root = repository_root()
    lock = yaml.safe_load((root / "upstreams.lock.yaml").read_text())
    spec = lock["upstreams"][OPENPI_NAME]
    if str(spec["commit"]) != OPENPI_COMMIT:
        raise RuntimeError("upstreams.lock.yaml does not pin required OpenPI commit")
    path = root / spec["path"]
    if not path.is_dir():
        raise FileNotFoundError(f"missing locked OpenPI checkout: {path}")
    if verify:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if head != OPENPI_COMMIT:
            raise RuntimeError(f"OpenPI revision mismatch: {head} != {OPENPI_COMMIT}")
    source = str(path / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    return path
