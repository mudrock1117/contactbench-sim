"""Pure contract tests; do not import simulator or FTP-1 model runtimes."""
import numpy as np
import pytest
from force_bench.ftp1.data import causal_indices
from force_bench.ftp1.preprocess import decode_action_chunk, PROMPT

def test_causal_indices_never_select_future():
    source=np.asarray([0.0, 0.05, 0.10],dtype=np.float64)
    target=np.asarray([0.0, 0.049, 0.10],dtype=np.float64)
    idx=causal_indices(source,target)
    assert np.all(source[idx] <= target)
    assert idx.tolist()==[0,0,2]

def test_causal_indices_reject_missing_past_sample():
    with pytest.raises(ValueError):
        causal_indices([0.1,0.2],[0.0])

def test_action_decode_uses_pose_and_absolute_gripper_slot():
    actions=np.zeros((32,120),dtype=np.float32)
    actions[:, 0:3] = 11.0
    actions[:, 3:9] = 22.0
    actions[:, 9:16] = 33.0
    actions[:, 16:48] = 44.0
    actions[:, 48:57] = 55.0
    actions[:, 44] = 0.04
    decoded=decode_action_chunk(actions)
    assert decoded.shape==(32,10)
    assert np.all(decoded[:, :3] == 11.0)
    assert np.all(decoded[:, 3:9] == 22.0)
    assert np.all(decoded[:, 9] == 0.04)

def test_prompt_has_required_punctuation():
    assert PROMPT=="pick up the thin object on the table."
