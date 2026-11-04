"""Bind completed model checkpoints to their own recorded statistics.

Only the Agent's successful final manifest can publish this association. Input
dataset defaults never select a model's normalization file.
"""

from pathlib import PurePosixPath
import re

from .common import absolute_path, now, uid
from .models import Asset


def _descriptor(output):
    if output.get("kind") != "MODEL":
        return None
    return output.get("metadata", {}).get("norm_stats")


def model_norm_outputs_error(outputs):
    """Validate all announced bindings before any final output is registered."""
    seen = {}
    for output in outputs:
        descriptor = _descriptor(output)
        if descriptor is None:
            continue  # Historical manifests do not announce model statistics.
        path = descriptor.get("path") if isinstance(descriptor, dict) else None
        model_path = PurePosixPath(output["path"])
        method = descriptor.get("normalization_method") if isinstance(descriptor, dict) else None
        valid_path = (isinstance(path, str) and len(path) <= 4096 and absolute_path(path)
                      and "\\" not in path and not any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in path))
        norm_path = PurePosixPath(path) if valid_path else None
        if (not isinstance(descriptor, dict) or not valid_path
                or norm_path.name != "norm_stats.json" or model_path / "assets" not in norm_path.parents
                or not isinstance(descriptor.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is None
                or method is not None and (not isinstance(method, str) or method not in {"quantile", "minmax", "zscore"})
                or descriptor.get("shared") is not True
                or norm_path != model_path / "assets/shared/norm_stats.json"):
            return {"code": "MODEL_NORM_INVALID", "message": "模型归一化参数必须是该 checkpoint 内实际保存的共享 norm_stats"}
        identity = str(norm_path)
        signature = (descriptor["sha256"], descriptor.get("normalization_method"))
        if identity in seen and seen[identity] != signature:
            return {"code": "MODEL_NORM_INVALID", "message": "同一模型归一化路径报告了相互冲突的统计"}
        seen[identity] = signature
    return None


def bind_model_norm_outputs(session, *, node, attempt, outputs, existing):
    """Publish same-node, same-attempt NORM assets in the final transaction."""
    for output in outputs:
        descriptor = _descriptor(output)
        if descriptor is None:
            continue
        model = existing[("MODEL", output["path"], output["version"])]
        path = str(PurePosixPath(descriptor["path"]))
        identity = ("NORM", path, output["version"])
        norm = existing.get(identity)
        metadata = {"format": "openpi-norm-stats", "registration_source": "model_training",
                    "model_checkpoint_path": model.path, "sha256": descriptor["sha256"], "shared": True,
                    **({"normalization_method": descriptor["normalization_method"]}
                       if descriptor.get("normalization_method") is not None else {})}
        if norm is None:
            suffix = " · norm_stats"
            norm = Asset(id=uid(), node_id=node.id, name=model.name[:200-len(suffix)] + suffix,
                         kind="NORM", path=path, version=output["version"], state="READY",
                         metadata_json=metadata, created_at=now(), origin_attempt_id=attempt.id)
            session.add(norm)
            existing[identity] = norm
        else:
            if norm.state == "ARCHIVED":
                # A user may archive a discovered NORM while this run is still
                # active. Completion cannot undo that explicit choice.
                model.metadata_json = {**model.metadata_json,
                    "norm_binding": {"status": "SKIPPED", "reason": "NORM_ASSET_ARCHIVED"}}
                continue
            norm.metadata_json = {**norm.metadata_json, **metadata}
            if norm.state == "UNAVAILABLE":
                norm.state = "READY"
        session.flush()
        model.norm_asset_id = norm.id
