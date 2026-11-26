"""Load the pinned official FTP-1 policy once for inference."""
from pathlib import Path
import numpy as np
from .preprocess import prepare_observation, decode_action_chunk, _right_action_slices

DOMAIN = "ContactBench_retrieve_card_from_deck"

def load_policy(checkpoint, device="cuda:0", steps=10):
    from openpi.policies import FTP1InferenceWrapper
    root = Path(checkpoint)
    for name in ("model.safetensors", "model_config.json", "train_config.json",
                 "tactile_input_config_file.json",
                 "hpt_tokenizer/GelSightMini_image_224_224_3.safetensors",
                 "hpt_tokenizer/shared_image_chunk_encoder.safetensors",
                 "normalization/share_norm_stats_all_t0_zscore.json",
                 f"normalization/{DOMAIN}/independent_norm_stats_all_t0_zscore.json"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    wrapper = FTP1InferenceWrapper(checkpoint_dir=str(root), domain_name=DOMAIN,
        tactile_input_config_file=str(root / "tactile_input_config_file.json"),
        device=device, num_inference_steps=steps)
    if not wrapper.model_config.use_tactile_input or wrapper.domain_norm_stats is None:
        raise RuntimeError("required tactile model or domain normalization missing")
    pg = wrapper.model.paligemma_with_expert.paligemma
    if pg.get_input_embeddings().weight.data_ptr() != pg.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("PaliGemma embedding and LM head are not tied")
    return wrapper

def predict(wrapper, wrist, tactile, width):
    if not np.isfinite(width) or width < 0:
        raise ValueError("invalid gripper width")
    inputs = prepare_observation(wrapper, wrist, tactile, width)
    pos, rot, grip = _right_action_slices(wrapper.get_action_dim())
    mask = np.zeros(wrapper.get_action_dim(), dtype=np.float32)
    mask[pos] = mask[rot] = 1
    mask[grip] = 1
    raw = np.asarray(wrapper.infer(**inputs, action_mask=mask), dtype=np.float32)
    if not np.isfinite(raw).all():
        raise ValueError("nonfinite FTP-1 prediction")
    return raw, decode_action_chunk(raw)
