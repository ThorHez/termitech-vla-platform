"""PyTorch checkpoint evidence without loading tensor data or importing torch."""

import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat

if __package__:
    from .recap import recap_model_role
else:
    from recap import recap_model_role


RECAP_TORCH_FORMAT = "openpi-pytorch"
RECAP_TORCH_WEIGHTS = "model.safetensors"
RECAP_TORCH_COMMIT = "recap_checkpoint.json"
_MAX_HEADER_BYTES = 16 * 1024 * 1024


def recap_weight_file(asset):
    """Resolve the conventional weight file beneath a registered asset path."""
    path = Path(asset)
    return path / RECAP_TORCH_WEIGHTS if path.is_dir() else path


def safetensors_header(path):
    """Read only the header and prove its intervals cover a nonempty payload."""
    if Path(path).name != RECAP_TORCH_WEIGHTS:
        raise ValueError("MODEL_FORMAT: ReCap 需要包含 model.safetensors 的 PyTorch 权重资产")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            count = int.from_bytes(stream.read(8), byteorder="little")
            if (not stat.S_ISREG(info.st_mode) or not 2 <= count <= _MAX_HEADER_BYTES
                    or count + 8 >= info.st_size):
                raise ValueError("invalid safetensors header size")
            raw = stream.read(count)
        header = json.loads(raw)
        if len(raw) != count or not isinstance(header, dict) or not 1 <= len(header) <= 20000:
            raise ValueError("invalid safetensors tensor directory")
        tensors = {name: entry for name, entry in header.items() if name != "__metadata__"}
        intervals = []
        for entry in tensors.values():
            if not isinstance(entry, dict) or not isinstance(entry.get("dtype"), str):
                raise ValueError("invalid tensor entry")
            shape, interval = entry.get("shape"), entry.get("data_offsets")
            if (not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape)
                    or not isinstance(interval, list) or len(interval) != 2
                    or any(type(offset) is not int for offset in interval)):
                raise ValueError("invalid tensor shape or data interval")
            intervals.append(tuple(interval))
        end = 0
        for start, stop in sorted(intervals):
            if start != end or stop < start:
                raise ValueError("non-contiguous safetensors data intervals")
            end = stop
        if end != info.st_size - count - 8:
            raise ValueError("safetensors data size does not match the file")
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError("MODEL_FORMAT: ReCap 需要有效的 PyTorch model.safetensors：" + str(path)) from exc
    return tensors, info.st_size, info.st_mtime_ns, hashlib.sha256(raw).hexdigest()


def recap_commit_matches(receipt, step, stage=None):
    return (isinstance(receipt, dict) and receipt.get("schema_version") == 1
            and receipt.get("backend") == "pytorch" and type(receipt.get("step")) is int
            and receipt["step"] == step and receipt.get("stage") in ("value", "stage_progress", "offline_rl")
            and (stage is None or receipt["stage"] == stage))


def recap_model_evidence_valid(model, asset_path):
    if (not isinstance(model, dict) or not isinstance(asset_path, str) or not asset_path.startswith("/")
            or model.get("format") != RECAP_TORCH_FORMAT or model.get("checkpoint_path") != asset_path
            or type(model.get("parameter_bytes")) is not int or model["parameter_bytes"] <= 0):
        return False
    asset = PurePosixPath(asset_path)
    weights = model.get("path")
    expected = str(asset) if asset.name == RECAP_TORCH_WEIGHTS else str(asset / RECAP_TORCH_WEIGHTS)
    metadata = model.get("metadata")
    return (weights == expected and isinstance(metadata, list) and 1 <= len(metadata) <= 2
            and isinstance(metadata[0], dict) and metadata[0].get("path") == weights
            and metadata[0].get("scope") == "safetensors_header"
            and isinstance(metadata[0].get("sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", metadata[0]["sha256"]) is not None)


def recap_torch_artifact_valid(output, output_dir, recipe, stage, *, final_step=None):
    if not isinstance(output, dict) or output.get("kind") != "MODEL" or not isinstance(output_dir, str) or not isinstance(recipe, str):
        return False
    metadata, version = output.get("metadata"), output.get("version")
    role = recap_model_role(stage)
    return (isinstance(version, str) and version.isascii() and version.isdigit() and len(version) <= 16
            and (final_step is None or version == str(final_step)) and role is not None
            and output.get("path") == str(PurePosixPath(output_dir) / "checkpoints" / recipe / "platform" / version)
            and isinstance(metadata, dict) and metadata.get("format") == RECAP_TORCH_FORMAT
            and metadata.get("backend") == "pytorch" and metadata.get("model_role") == role
            and type(metadata.get("step")) is int and str(metadata["step"]) == version)
