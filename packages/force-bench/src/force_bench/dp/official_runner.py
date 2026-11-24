"""No-simulation runner satisfying the pinned upstream workspace contract."""
from __future__ import annotations
from force_bench.dp.official import activate_upstream
activate_upstream()
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class ContactBenchOfflineImageRunner(BaseImageRunner):
    """Workspace-compatible runner; closed-loop evaluation remains a separate process."""
    def __init__(self, output_dir):
        super().__init__(output_dir)

    def run(self, policy):
        # Epoch-zero workspace rollout must be valid without starting Newton.
        return {"offline_runner": 1.0}
