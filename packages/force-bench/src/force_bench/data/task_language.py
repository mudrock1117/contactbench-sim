"""Outcome-free task language shared by offline and online policy inputs."""

_INSTRUCTIONS = {
    "tighten_allen_key": "Use the Allen key to tighten the socket-head screw.",
    "loosen_allen_key": "Use the Allen key to loosen the socket-head screw.",
}


def task_instruction(task):
    """Return stable task semantics without state, phase, or outcome leakage."""
    key = str(task or "").strip().lower().replace("-", "_").replace(" ", "_")
    if "allen_key" in key and "tighten" in key:
        return _INSTRUCTIONS["tighten_allen_key"]
    if "allen_key" in key and "loosen" in key:
        return _INSTRUCTIONS["loosen_allen_key"]
    for token, instruction in _INSTRUCTIONS.items():
        if token in key:
            return instruction
    readable = key.replace("_", " ").strip()
    return f"Complete the {readable} task." if readable else "Complete the assigned task."

