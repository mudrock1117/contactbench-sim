"""Delegate ContactBench pi0.5 fine-tuning to the locked official JAX trainer."""
from __future__ import annotations

import argparse
import importlib.util
import os

from .config import DEFAULT_DATA, DEFAULT_MODEL_ROOT, install_dataset_factory, make_config
from .official import activate_openpi


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=("smoke", "paper"), required=True)
    p.add_argument("--exp-name", required=True)
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)
    root = activate_openpi()
    config = install_dataset_factory(make_config(
        a.profile, exp_name=a.exp_name, data_path=a.data,
        model_root=a.model_root, resume=a.resume, overwrite=a.overwrite))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.90")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.join(
        a.model_root, "cache", "jax"))
    os.environ.setdefault("WANDB_DIR", os.path.join(a.model_root, "logs"))
    spec = importlib.util.spec_from_file_location(
        "_contactbench_openpi_train", root / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
