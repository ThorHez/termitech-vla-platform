"""Transport helpers shared by the control plane and node agent; no ORM imports."""

import hashlib
import json
from typing import Any

PROTOCOL_VERSION = 1


def canonical_digest(value: Any) -> str:
    """Stable fingerprint for JSON-compatible command and configuration values."""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

