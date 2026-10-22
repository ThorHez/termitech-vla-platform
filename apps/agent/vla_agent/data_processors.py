"""Read processor constructors through the configured, CPU-only training Python."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from .recipe_details import _git_pin, _source
from .recipe_suggestions import SOURCE_PATHS
from .safety import AgentError, allowed_path
from .training_validation import _probe_process


PROBE_PREFIX = "VLA_DATA_PROCESSORS:"
PROBE_TIMEOUT_SECONDS = 45
_CREDENTIALS = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "WANDB_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"}


def data_processors(config: dict, payload: dict) -> dict:
    """Catalog metadata is discovery evidence, not training acceptance."""
    profile_id = payload.get("profile_id")
    if (not isinstance(profile_id, str) or not 1 <= len(profile_id) <= 200
            or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in profile_id)):
        raise AgentError("INVALID_DATA_PROCESSORS", "A configured training environment is required")
    try:
        expires = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
        if expires.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("UTC required")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise AgentError("INVALID_DATA_PROCESSORS", "Catalog request needs a UTC expiration") from exc
    if expires <= datetime.now(timezone.utc):
        raise AgentError("DATA_PROCESSORS_EXPIRED", "Catalog query expired; submit a new query")
    profile = next((item for item in config["profiles"] if item.get("id") == profile_id), None)
    if profile is None or profile.get("adapter") != "openpi-sft" or profile.get("execution_mode", "native") != "native":
        raise AgentError("PROFILE_UNAVAILABLE", "Catalog needs a configured native OpenPI environment")
    try:
        checkout = allowed_path(profile["code_dir"], config["roots"])
        interpreter = Path(profile["python"])
        if not interpreter.is_absolute() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise AgentError("RUNTIME_UNAVAILABLE", "Configured training Python is unavailable")
        source_name = next((name for name in SOURCE_PATHS if (checkout / name).exists()), None)
        if source_name is None:
            raise AgentError("RECIPE_SOURCE_UNAVAILABLE", "OpenPI config.py is unavailable")
        source_path = checkout / source_name
        pin = _git_pin(checkout, config["roots"])
        content, identity, resolved = _source(source_path, checkout, config["roots"])
        import vla_adapters
        helper = Path(vla_adapters.__file__).resolve().parent / "data_processors.py"
        spec = {"code_dir": str(checkout), "allowed_roots": list(config["roots"])}
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith("VLA_") and key not in _CREDENTIALS}
        environment.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", PYTHONDONTWRITEBYTECODE="1",
                           WANDB_MODE="disabled", WANDB_DISABLED="true", HF_HUB_OFFLINE="1",
                           HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
        returncode, stdout, _ = _probe_process(
            [str(interpreter), "-B", str(helper), "--catalog-json", json.dumps(spec)], str(checkout), environment,
            timeout=PROBE_TIMEOUT_SECONDS, output_limit=1024 * 1024)
        responses = [line[len(PROBE_PREFIX):] for line in stdout.splitlines() if line.startswith(PROBE_PREFIX)]
        if returncode != 0 or len(responses) != 1:
            raise AgentError("DATA_PROCESSORS_FAILED", "Training Python could not enumerate data processors")
        result = json.loads(responses[0])
        if not isinstance(result, dict) or not isinstance(result.get("entries"), list) or len(result["entries"]) > 128:
            raise AgentError("DATA_PROCESSORS_INVALID_RESULT", "Training Python returned an invalid catalog")
        reported = result.get("source")
        if (not isinstance(reported, dict) or reported.get("path") != source_name
                or reported.get("code_dir") != str(checkout)
                or reported.get("sha256") != hashlib.sha256(content).hexdigest()):
            raise AgentError("RECIPE_SOURCE_CHANGED", "Catalog source differs from the configured checkout")
        current, current_identity, current_resolved = _source(source_path, checkout, config["roots"])
        if (current != content or current_identity != identity or current_resolved != resolved
                or _git_pin(checkout, config["roots"]) != pin):
            raise AgentError("RECIPE_SOURCE_CHANGED", "Training source changed while cataloging processors")
        return {"profile_id": profile_id, "entries": result["entries"],
                "source": {"path": source_name, "sha256": hashlib.sha256(content).hexdigest(), "code_dir": str(checkout), **pin}}
    except AgentError as exc:
        translated = {"PREFLIGHT_TIMEOUT": "DATA_PROCESSORS_TIMEOUT", "PREFLIGHT_OUTPUT_LIMIT": "DATA_PROCESSORS_OUTPUT_LIMIT"}
        if exc.code in translated:
            raise AgentError(translated[exc.code], "Data processor inspection exceeded its time or output limit") from exc
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AgentError("DATA_PROCESSORS_FAILED", "Data processors could not be read from the training environment") from exc
