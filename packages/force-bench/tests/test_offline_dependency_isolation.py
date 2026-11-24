"""Regression tests for training-only imports without simulation packages."""
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[3]
SRC=ROOT/"packages/force-bench/src"

def _run_guarded_import(modules):
    code=r"""
import builtins
real_import=builtins.__import__
blocked={"warp","newton","mujoco","mujoco_warp","trimesh"}
def guarded(name,*args,**kwargs):
    if name.split(".",1)[0] in blocked:
        raise AssertionError("offline import attempted simulation dependency: "+name)
    return real_import(name,*args,**kwargs)
builtins.__import__=guarded
"""+"\n".join(f"__import__({module!r})" for module in modules)+r"""
import sys
assert not (blocked & set(sys.modules))
"""
    env=dict(os.environ)
    env["PYTHONPATH"]=str(SRC)+os.pathsep+env.get("PYTHONPATH","")
    result=subprocess.run([sys.executable,"-c",code],env=env,text=True,
                          capture_output=True,check=False)
    assert result.returncode==0,result.stdout+result.stderr

def test_offline_rgb_dataset_and_eef_math_do_not_import_simulation_stack():
    _run_guarded_import([
        "force_bench.data.rgb",
        "force_bench.data.episode",
        "force_bench.data.dataset",
        "force_bench.robot.eef_action",
    ])

def test_official_dataset_import_is_simulation_free_when_training_stack_is_installed():
    required=("torch","hydra","omegaconf","diffusers","robomimic")
    if any(importlib.util.find_spec(name) is None for name in required):
        pytest.skip("official DP Python dependencies are not installed")
    if not (ROOT/"third_party/diffusion_policy").is_dir():
        pytest.skip("pinned diffusion_policy checkout is not present")
    _run_guarded_import(["force_bench.dp.official_dataset"])
