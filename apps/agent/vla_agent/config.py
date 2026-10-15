from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    value = json.loads(source.read_text())
    for key in ("control_plane_url", "name", "roots", "output_root", "state_dir", "profiles"):
        if key not in value:
            raise ValueError(f"Missing agent configuration: {key}")
    if not value["roots"] or not isinstance(value["profiles"], list):
        raise ValueError("roots must be nonempty and profiles must be a list")
    value["roots"] = [str(Path(root).expanduser().resolve(strict=True)) for root in value["roots"]]
    for key in ("output_root", "state_dir"):
        resolved = Path(value[key]).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        value[key] = str(resolved)
    output = Path(value["output_root"])
    if not any(output == Path(root) or Path(root) in output.parents for root in value["roots"]):
        raise ValueError("output_root must be inside a configured root so produced assets can be reused")
    value["config_path"] = str(source)
    value.setdefault("poll_interval_seconds", 2)
    value.setdefault("heartbeat_interval_seconds", 10)
    value.setdefault("http_timeout_seconds", 15)
    value.setdefault("event_batch_size", 100)
    if any(float(value[k]) <= 0 for k in ("poll_interval_seconds", "heartbeat_interval_seconds", "http_timeout_seconds")):
        raise ValueError("Intervals and timeouts must be positive")
    ids = [p["id"] for p in value["profiles"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Profile IDs must be unique")
    return value


def get_profile(config: dict, profile_id: str) -> dict:
    for profile in config["profiles"]:
        if profile["id"] == profile_id:
            if profile.get("execution_mode", "native") != "native":
                raise ValueError("This agent supports native execution only")
            return profile
    raise ValueError(f"Unknown runtime profile: {profile_id}")

