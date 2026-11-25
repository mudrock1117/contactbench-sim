"""Restore and validate normalizers omitted by the upstream nested AT state."""
from __future__ import annotations

import torch


_REQUIRED_FIELDS = {
    "action": 10,
    "tactile_marker_pca": 30,
}


def _field(normalizer, key, *, owner):
    params = getattr(normalizer, "params_dict", None)
    if params is None or key not in params:
        raise ValueError(f"{owner} normalizer is missing {key!r}")
    field = params[key]
    scale = field.get("scale") if hasattr(field, "get") else field["scale"]
    if scale is None:
        raise ValueError(f"{owner} normalizer {key!r} has no scale")
    expected = _REQUIRED_FIELDS[key]
    if int(scale.numel()) != expected:
        raise ValueError(
            f"{owner} normalizer {key!r} has width {scale.numel()}, "
            f"expected {expected}")
    for name, value in field.state_dict().items():
        if not torch.isfinite(value).all():
            raise ValueError(
                f"{owner} normalizer {key!r}.{name} contains non-finite values")
    return field


def validate_policy_normalizers(policy, *, owner="policy"):
    """Validate parent and nested AT fields required by reactive decoding."""
    for key in _REQUIRED_FIELDS:
        parent = _field(policy.normalizer, key, owner=owner)
        nested = _field(policy.at.normalizer, key, owner=f"{owner}.at")
        parent_state = parent.state_dict()
        nested_state = nested.state_dict()
        if parent_state.keys() != nested_state.keys():
            raise ValueError(f"{owner} nested AT {key!r} normalizer schema differs")
        for name in parent_state:
            if not torch.equal(parent_state[name], nested_state[name]):
                raise ValueError(
                    f"{owner} nested AT {key!r}.{name} differs from parent")
    return policy


def restore_nested_at_normalizer(policy, *, owner="policy"):
    """Restore AT stats from the serialized parent LDP normalizer.

    Upstream VAE.state_dict() intentionally serializes network weights only.
    The enclosing LDP normalizer is serialized in the LDP checkpoint and is
    therefore the exact training-time source for both action and tactile stats.
    """
    for key in _REQUIRED_FIELDS:
        _field(policy.normalizer, key, owner=owner)
    if not hasattr(policy, "at") or not hasattr(policy.at, "set_normalizer"):
        raise ValueError(f"{owner} has no restorable nested AT")
    policy.at.set_normalizer(policy.normalizer)
    return validate_policy_normalizers(policy, owner=owner)


def restore_workspace_normalizers(workspace):
    """Repair and validate ordinary and EMA policies after checkpoint loading."""
    policies = [("model", workspace.model)]
    ema = getattr(workspace, "ema_model", None)
    if ema is not None:
        policies.append(("ema_model", ema))
    for owner, policy in policies:
        restore_nested_at_normalizer(policy, owner=owner)

    if ema is not None:
        for key in _REQUIRED_FIELDS:
            model_state = _field(
                workspace.model.normalizer, key, owner="model").state_dict()
            ema_state = _field(
                ema.normalizer, key, owner="ema_model").state_dict()
            if model_state.keys() != ema_state.keys():
                raise ValueError(f"EMA {key!r} normalizer schema differs")
            for name in model_state:
                if not torch.equal(model_state[name], ema_state[name]):
                    raise ValueError(
                        f"EMA {key!r}.{name} normalizer differs from model")
    return workspace
