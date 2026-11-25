"""ContactBench adapter for the official RDP AT/LDP workspaces."""
from __future__ import annotations
import copy
import numpy as np
import torch

from force_bench.data.dataset import EpisodeWindowDataset, WindowSpec
from force_bench.data.episode import discover_episodes, load_episode
from force_bench.data.timebase import PolicyTimeConfig, causal_indices
from force_bench.dp.official_dataset import _pose_matrix
from force_bench.dp.official_contract import validate_eef_episode_arrays
from force_bench.robot.eef_action import encode_relative_eef, relative_eef_history
from .official import activate_upstream
from .tactile import load_gelsight_pca, marker_window_indices
from .timebase import RDPTimeConfig

activate_upstream()
from reactive_diffusion_policy.dataset.base_dataset import BaseImageDataset
from reactive_diffusion_policy.model.common.normalizer import (
    LinearNormalizer, SingleFieldLinearNormalizer)


class ContactBenchOfficialRDPDataset(BaseImageDataset):
    """24 Hz actions/tactile, causal 12 Hz wrist images, fixed chunk anchor."""

    REQUIRED_MARKER_FIELDS = (
        "tactile_marker_displacement_px", "tactile_marker_valid")

    def __init__(self, data_path, pca_transform_path=None, pca_mean_path=None,
                 pca_mode="official", pca_artifact_path=None,
                 observation_horizon=2, horizon=32, success_only=True,
                 seed=0, val_ratio=0.0, at=None,
                 use_latent_action_before_vq=False):
        super().__init__()
        self.time = RDPTimeConfig(
            observation_horizon=int(observation_horizon),
            action_horizon=int(horizon))
        timing = PolicyTimeConfig(
            physics_hz=60, policy_hz=24,
            observation_horizon=self.time.dataset_obs_steps,
            action_horizon=int(horizon))
        spec = WindowSpec(
            observation_horizon=self.time.dataset_obs_steps,
            action_horizon=int(horizon), camera_name="wrist",
            include_privileged=False, policy_time_config=timing)
        paths = discover_episodes(data_path, success_only=bool(success_only))
        if not paths:
            raise ValueError(f"no episodes under {data_path}")
        self.base = EpisodeWindowDataset(paths, spec)
        self.episodes = {}
        self.pca = {}
        self.pca_mode = str(pca_mode)
        shared_pca = (load_gelsight_pca(
            mode=self.pca_mode, artifact_path=pca_artifact_path)
            if self.pca_mode == "simulation_fitted" else None)
        for path in paths:
            ep = load_episode(path)
            validate_eef_episode_arrays(ep.arrays, len(ep["actions"]))
            missing = [key for key in self.REQUIRED_MARKER_FIELDS if key not in ep.arrays]
            if missing:
                raise ValueError(
                    f"{path}: missing {missing}; pressure grids are not marker displacement")
            contract = ep.metadata.get("tactile_markers", {})
            if self.pca_mode == "official":
                pca = load_gelsight_pca(
                    mode="official", transform_path=pca_transform_path,
                    mean_path=pca_mean_path, marker_contract=contract)
            else:
                artifact_contract = shared_pca.manifest["marker_contract"]
                for key in ("shape", "components", "image_size_px", "marker_ordering"):
                    if contract.get(key) != artifact_contract.get(key):
                        raise ValueError(f"{path}: marker contract differs from PCA artifact: {key}")
                pca = shared_pca
            self.pca[str(ep.metadata["episode_id"])] = pca
            self.episodes[str(ep.metadata["episode_id"])] = ep
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.val_ratio = float(val_ratio)
        self.at = at
        if self.at is not None:
            self.at.eval()
        self.use_latent_action_before_vq = bool(use_latent_action_before_vq)
        before = len(self.base.samples)
        self.base.samples = [
            pair for pair in self.base.samples if self._marker_window_valid(*pair)]
        self.filtered_invalid_marker_windows = before - len(self.base.samples)
        if not self.base.samples:
            raise ValueError("no RDP windows have a complete valid marker history")

    def _marker_window_valid(self, episode_index, anchor):
        """Validity is sensor availability, never contact/nonzero pressure."""
        item = self.base.episodes[episode_index]
        ep = item["episode"]
        anchor_s = float(item["policy_ticks"][anchor])
        query = self.time.fast_query_times(anchor_s)
        marker_times = np.asarray(ep["timestamps"][1:], dtype=np.float64)
        indices = marker_window_indices(
            marker_times, ep["tactile_marker_valid"], query)
        # A valid all-zero displacement is an important no-contact sample.
        return indices is not None

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        sample = self.base[index]
        eid = str(sample["episode_id"])
        ep = self.episodes[eid]
        anchor_s = float(sample["timestamp"][-1])
        query = self.time.fast_query_times(anchor_s)
        action_times = np.asarray(ep["timestamps"][:-1], dtype=np.float64)
        action_idx = causal_indices(action_times, query)
        if np.any(action_idx < 0) or np.any(action_idx >= len(ep["actions"])):
            raise IndexError("RDP window crosses episode boundary")
        anchor_i = int(action_idx[self.time.action_start_index])
        anchor = _pose_matrix(ep["action_anchor_tcp_pose_world_xyzw"][anchor_i])
        actions = np.stack([
            encode_relative_eef(
                anchor, _pose_matrix(ep["action_tcp_target_world_xyzw"][i]),
                ep["action_gripper_width_m"][i])
            for i in action_idx]).astype(np.float32)

        slow_times = self.time.slow_observation_times(anchor_s)
        state_idx = causal_indices(action_times, slow_times)
        poses = np.stack([
            _pose_matrix(ep["action_anchor_tcp_pose_world_xyzw"][i])
            for i in state_idx])
        widths = np.asarray(ep["action_anchor_gripper_width_m"])[state_idx]
        agent = relative_eef_history(poses, widths, anchor).astype(np.float32)

        # EpisodeWindowDataset returns the four causal 24 Hz image slots.
        # [::-2][::-1] is the exact upstream downsampling and ends at anchor.
        rgb_fast = np.asarray(sample["rgb"], dtype=np.float32) / 255.0
        wrist = rgb_fast[::-2][::-1].copy()
        if wrist.shape[0] != self.time.observation_horizon:
            raise RuntimeError("causal wrist history has an unexpected length")

        marker_times = np.asarray(ep["timestamps"][1:], dtype=np.float64)
        marker_idx = marker_window_indices(
            marker_times, ep["tactile_marker_valid"], query)
        if marker_idx is None:
            raise RuntimeError("filtered RDP sample unexpectedly has an invalid marker frame")
        marker = np.asarray(ep["tactile_marker_displacement_px"])[marker_idx]
        emb = self.pca[eid].transform(
            marker, image_width_px=240, image_height_px=320)
        # Keep the two physical pads distinct: [left 15, right 15].
        emb = emb.reshape(self.horizon, 30).astype(np.float32)
        obs_emb = emb[:self.time.dataset_obs_steps][::-2][::-1].copy()

        return {
            "obs": {
                "wrist_rgb": torch.from_numpy(wrist),
                "agent_pos": torch.from_numpy(agent),
                "tactile_marker_pca": torch.from_numpy(obs_emb),
            },
            "extended_obs": {
                "tactile_marker_pca": torch.from_numpy(emb),
            },
            "action": torch.from_numpy(actions),
        }

    def get_validation_dataset(self):
        # Splits are episode-level in the A4 discovery layer; a one-episode
        # smoke run intentionally has no statistically independent test set.
        return copy.copy(self)

    def get_normalizer(self, **kwargs):
        actions, agents, tactile = [], [], []
        for i in range(len(self)):
            item = self[i]
            actions.append(item["action"].numpy())
            agents.append(item["obs"]["agent_pos"].numpy())
            tactile.append(item["extended_obs"]["tactile_marker_pca"].numpy())
        normalizer = LinearNormalizer()
        normalizer.fit({
            "action": np.concatenate(actions),
            "agent_pos": np.concatenate(agents),
            "tactile_marker_pca": np.concatenate(tactile),
        }, last_n_dims=1, mode=kwargs.pop("mode", "limits"), **kwargs)
        normalizer["wrist_rgb"] = SingleFieldLinearNormalizer.create_identity()
        if self.at is not None:
            import einops
            latent_all = []
            for action in actions:
                value = torch.from_numpy(action).to(self.at.device).unsqueeze(0)
                value = normalizer["action"].normalize(value)
                latent = self.at.encoder(self.at.preprocess(value / self.at.act_scale))
                if self.at.use_vq:
                    if not self.use_latent_action_before_vq:
                        latent, _, _ = self.at.quant_state_with_vq(latent)
                else:
                    latent, _ = self.at.quant_state_without_vq(latent)
                steps = self.at.downsampled_input_h if self.at.use_conv_encoder else 1
                latent = einops.rearrange(latent, "n (t a) -> n t a", t=steps)
                latent_all.append(latent[0].detach().cpu().numpy())
            normalizer["latent_action"] = SingleFieldLinearNormalizer.create_fit(
                np.concatenate(latent_all))
        return normalizer

    def get_all_actions(self):
        return torch.cat([self[i]["action"] for i in range(len(self))], dim=0)
