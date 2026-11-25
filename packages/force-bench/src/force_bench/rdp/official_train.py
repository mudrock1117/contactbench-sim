from __future__ import annotations
import argparse
from .official import run_workspace


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train locked official RDP AT or latent diffusion workspace")
    parser.add_argument("--stage", choices=("at", "ldp"), required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run_workspace(args.config, stage=args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
