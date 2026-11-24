"""Fetch exact, reviewable upstream source revisions (no dependencies or weights)."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import yaml

ROOT = Path(__file__).resolve().parents[1]

def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", action="append", help="Only fetch named upstream (repeatable)")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    lock = yaml.safe_load((ROOT / "upstreams.lock.yaml").read_text())
    selected = set(args.name or lock["upstreams"])
    unknown = selected - set(lock["upstreams"])
    if unknown:
        parser.error(f"unknown upstreams: {sorted(unknown)}")
    for name, spec in lock["upstreams"].items():
        if name not in selected:
            continue
        path = ROOT / spec["path"]
        if not path.exists():
            if args.verify_only:
                raise SystemExit(f"{name}: missing {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            run("git", "clone", "--filter=blob:none", "--no-checkout", spec["repository"], str(path))
        if not args.verify_only:
            run("git", "fetch", "--depth=1", "origin", spec["commit"], cwd=path)
            run("git", "checkout", "--detach", "FETCH_HEAD", cwd=path)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        expected = subprocess.check_output(
            ["git", "rev-parse", f"{spec['commit']}^{{commit}}"], cwd=path, text=True).strip()
        if head != expected:
            raise SystemExit(f"{name}: expected {expected}, found {head}")
        print(f"{name}: {head}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
