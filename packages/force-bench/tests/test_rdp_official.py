import importlib
import sys
import numpy as np
import pytest


def test_rdp_offline_contract_imports_without_simulator():
    before = set(sys.modules)
    tactile = importlib.import_module("force_bench.rdp.tactile")
    importlib.import_module("force_bench.rdp.timebase")
    added = set(sys.modules) - before
    assert "warp" not in added
    assert "newton" not in added
    assert tactile.MARKER_COMPONENTS == ("x_right_px", "y_down_px")


def test_rdp_12_24_60_timing_and_action_anchor():
    from force_bench.rdp.timebase import RDPTimeConfig
    cfg = RDPTimeConfig()
    first = [cfg.physics_steps_for_fast_tick(i) for i in range(4)]
    assert first == [2, 3, 2, 3]
    steps_24 = [cfg.physics_steps_for_fast_tick(i) for i in range(24)]
    steps_2400 = [cfg.physics_steps_for_fast_tick(i) for i in range(2400)]
    assert sum(steps_24) == 60
    assert sum(steps_2400) == 6000
    assert all(sum(steps_2400[i:i + 2]) == 5 for i in range(0, 2400, 2))

    from force_bench.data.timebase import PolicyFrameScheduler, PolicyTimeConfig
    online = PolicyFrameScheduler(PolicyTimeConfig(60, 24, 4, 32))
    assert [online.pop() for _ in range(24)] == steps_24
    # The established DP clock remains exactly five physics frames per action.
    dp = PolicyFrameScheduler(PolicyTimeConfig(60, 12, 2, 16))
    assert [dp.pop() for _ in range(24)] == [5] * 24

    query = cfg.fast_query_times(1.0)
    assert query.shape == (32,)
    assert query[cfg.action_start_index] == pytest.approx(1.0)
    slow = cfg.slow_observation_times(1.0)
    assert slow.shape == (2,)
    assert slow[-1] == pytest.approx(1.0)
    assert np.all(np.diff(slow) > 0)


def test_official_pca_is_exact_affine_projection(tmp_path):
    from force_bench.rdp.tactile import OfficialGelSightPCA, marker_metadata
    mean = np.arange(126, dtype=np.float32) / 1000
    W = np.arange(126 * 15, dtype=np.float32).reshape(126, 15) / 10000
    mp, wp = tmp_path / "mean.npy", tmp_path / "w.npy"
    np.save(mp, mean)
    np.save(wp, W)
    pca = OfficialGelSightPCA(
        wp, mp, marker_contract={
            **marker_metadata(), "official_pca_compatible": True})
    marker = np.zeros((2, 63, 2), dtype=np.float32)
    marker[..., 0] = 24
    marker[..., 1] = 32
    got = pca.transform(marker, image_width_px=240, image_height_px=320)
    normalized = marker.copy()
    normalized[..., 0] /= 240
    normalized[..., 1] /= 320
    expected = (normalized.reshape(2, 126) - mean) @ W
    np.testing.assert_allclose(got, expected)


def test_official_pca_fails_closed_on_uncertified_ordering(tmp_path):
    from force_bench.rdp.tactile import OfficialGelSightPCA, marker_metadata
    np.save(tmp_path / "mean.npy", np.zeros(126, np.float32))
    np.save(tmp_path / "w.npy", np.zeros((126, 15), np.float32))
    with pytest.raises(ValueError, match="not certified"):
        OfficialGelSightPCA(
            tmp_path / "w.npy", tmp_path / "mean.npy",
            marker_contract=marker_metadata())


def test_pressure_grid_cannot_be_projected_as_marker_data(tmp_path):
    from force_bench.rdp.tactile import OfficialGelSightPCA, marker_metadata
    np.save(tmp_path / "mean.npy", np.zeros(126, np.float32))
    np.save(tmp_path / "w.npy", np.zeros((126, 15), np.float32))
    pca = OfficialGelSightPCA(
        tmp_path / "w.npy", tmp_path / "mean.npy",
        marker_contract={**marker_metadata(), "official_pca_compatible": True})
    with pytest.raises(ValueError, match=r"\[63,2\]"):
        pca.transform(np.zeros((2, 16, 16), np.float32),
                      image_width_px=240, image_height_px=320)


def test_rdp_profiles_use_locked_official_workspaces():
    from pathlib import Path
    import yaml
    root = Path(__file__).parents[1] / "config" / "rdp"
    expected = {
        "at_smoke.yaml": "train_at_workspace",
        "at_paper.yaml": "train_at_workspace",
        "ldp_smoke.yaml": "train_latent_diffusion_unet_real_image_workspace",
        "ldp_paper.yaml": "train_latent_diffusion_unet_real_image_workspace",
    }
    for name, upstream in expected.items():
        cfg = yaml.safe_load((root / name).read_text())
        assert cfg["upstream_config"] == upstream
        shape = cfg["config"]["shape_meta"]
        assert shape["action"]["shape"] == [10]
        assert shape["obs"]["tactile_marker_pca"]["shape"] == [30]
        assert shape["obs"].get("object_pose") is None


def test_marker_window_filter_keeps_valid_no_contact_and_rejects_missing():
    from force_bench.rdp.tactile import marker_window_indices
    timestamps = np.arange(8, dtype=np.float64) / 24.0
    query = timestamps[2:6]
    valid = np.ones((8, 2), dtype=bool)
    # Displacement magnitude is intentionally absent from this decision:
    # valid zero/no-contact marker samples are retained.
    indices = marker_window_indices(timestamps, valid, query)
    np.testing.assert_array_equal(indices, np.arange(2, 6))

    valid[4, 1] = False
    assert marker_window_indices(timestamps, valid, query) is None
    assert marker_window_indices(timestamps, np.ones((8, 2), bool),
                                 np.array([-1.0 / 24.0])) is None
