"""Frozen identity and preflight proof for a platform SFT checkpoint."""

import ast
from pathlib import PurePosixPath
import re

from . import canonical_digest
from .config_preview import validate_generated_config


def validate_resume_source(value):
    fields = {"asset_id", "run_id", "attempt_id", "path", "output_dir", "step"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Resume source must name one frozen platform checkpoint")
    for key in ("asset_id", "run_id", "attempt_id"):
        if not isinstance(value[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", value[key]):
            raise ValueError("Invalid resume source identity")
    paths = []
    for key in ("path", "output_dir"):
        raw = value[key]
        if (not isinstance(raw, str) or not 1 <= len(raw) <= 4096
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in raw)):
            raise ValueError("Invalid resume source path")
        path = PurePosixPath(raw)
        if not path.is_absolute() or raw.startswith("//") or str(path) != raw or ".." in path.parts:
            raise ValueError("Resume source paths must be canonical and absolute")
        paths.append(path)
    checkpoint, output = paths
    step = value["step"]
    if (type(step) is not int or not 0 <= step < 2**53 or output.name != value["attempt_id"]
            or checkpoint.name != str(step) or checkpoint.parent.name != "platform"
            or checkpoint.parent.parent.parent != output / "checkpoints"):
        raise ValueError("Resume source must be a checkpoint step under its owning attempt")
    return value


def validate_resume_evidence(plan, source):
    """Confirm the checkpoint proof and the resume source in the reviewed config."""
    if not isinstance(plan, dict):
        raise ValueError("Missing resume plan")
    proof = plan.get("resume_checkpoint")
    if source is None:
        if proof is not None or plan.get("resume_source") is not None:
            raise ValueError("Unexpected resume plan for a fresh run")
        return None
    validate_resume_source(source)
    if canonical_digest(plan.get("resume_source")) != canonical_digest(source):
        raise ValueError("Resume checkpoint differs from the requested source")
    digests = {"checkpoint_metadata_sha256", "params_manifest_digest", "train_state_metadata_sha256",
               "train_state_manifest_digest", "source_effective_config_sha256", "source_execution_plan_digest"}
    counters = {"commit_timestamp_nsecs", "train_state_bytes"}
    if (not isinstance(proof, dict) or not digests | counters <= set(proof)
            or any(not isinstance(proof[key], str) or re.fullmatch(r"[a-f0-9]{64}", proof[key]) is None for key in digests)
            or any(type(proof[key]) is not int or proof[key] <= 0 for key in counters)):
        raise ValueError("Missing committed resume checkpoint or train state evidence")
    generated = validate_generated_config(plan.get("generated_config"))
    tree = ast.parse(generated["source"])
    frozen = [node.value for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == "FROZEN_REQUEST" for target in node.targets)]
    if (len(frozen) == 1 and isinstance(frozen[0], ast.Dict)
            and all(isinstance(key, ast.Constant) and isinstance(key.value, str) for key in frozen[0].keys)):
        values = [value for key, value in zip(frozen[0].keys, frozen[0].values)
                  if isinstance(key, ast.Constant) and key.value == "resume_source"]
        if len(values) == 1:
            try:
                if canonical_digest(ast.literal_eval(values[0])) == canonical_digest(source):
                    return proof
            except (TypeError, ValueError, SyntaxError, RecursionError):
                pass
    raise ValueError("Reviewed config does not bind the requested resume source")
