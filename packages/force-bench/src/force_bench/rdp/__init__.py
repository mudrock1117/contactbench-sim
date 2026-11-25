"""Pinned official Reactive Diffusion Policy integration.

The implementation deliberately imports the upstream AT, latent policy and
workspaces from xiaoxiaoxh/reactive_diffusion_policy@8ed82a3.  No model core
is copied into force_bench.
"""
from .official import activate_upstream, workspace_from_yaml
from .timebase import RDPTimeConfig

__all__ = ["activate_upstream", "workspace_from_yaml", "RDPTimeConfig"]
