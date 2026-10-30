"""Versioned platform recipes: data-only edits over a pinned code recipe."""

import copy
import dataclasses
import hashlib
import json
import math
import re


OVERRIDE_RULES = {
    "num_train_steps": {"type": "integer", "min": 1, "max": 1_000_000},
    "batch_size": {"type": "integer", "min": 1, "max": 4096},
    "lr_schedule.peak_lr": {"type": "number", "min": 1e-8, "max": 0.1},
    "ema_decay": {"type": "number", "min": 0, "max": 0.999999999},
    "num_workers": {"type": "integer", "min": 0, "max": 128},
    "log_interval": {"type": "integer", "min": 1, "max": 1_000_000},
    "save_interval": {"type": "integer", "min": 1, "max": 1_000_000},
    "keep_period": {"type": "integer", "min": 1, "max": 10_000_000},
    "seed": {"type": "integer", "min": 0, "max": 4_294_967_295},
    "fsdp_devices": {"type": "integer", "min": 1, "max": 4096},
    "lr_schedule.warmup_steps": {"type": "integer", "min": 0, "max": 10_000_000},
    "lr_schedule.decay_steps": {"type": "integer", "min": 1, "max": 100_000_000},
    "lr_schedule.decay_lr": {"type": "number", "min": 0, "max": 0.1},
    "optimizer.clip_gradient_norm": {"type": "number", "min": 1e-8, "max": 1000},
}
PARAMETER_PATHS = {"steps": "num_train_steps", "batch_size": "batch_size", "learning_rate": "lr_schedule.peak_lr",
                   "save_interval": "save_interval", "keep_period": "keep_period", "num_workers": "num_workers",
                   "warmup_steps": "lr_schedule.warmup_steps", "decay_steps": "lr_schedule.decay_steps",
                   "decay_lr": "lr_schedule.decay_lr", "clip_gradient_norm": "optimizer.clip_gradient_norm",
                   "ema_decay": "ema_decay", "fsdp_devices": "fsdp_devices", "log_interval": "log_interval"}
SOURCE_PATHS = {"src/openpi/training/config.py", "openpi/training/config.py"}
# Discovery still starts at the compatibility facade. Only these explicit
# recipe parts may provide a saved definition or receive a source writeback.
RECIPE_PART_PATHS = {f"{prefix}openpi/training/recipes/{method}.py"
                     for prefix in ("src/", "") for method in ("sft", "recap", "mem", "rlt", "uot")}
RECIPE_SOURCE_PATHS = SOURCE_PATHS | RECIPE_PART_PATHS
LITERAL_ROOTS = {"data", "model", "optimizer", "lr_schedule"}
_PROTECTED_FIELDS = {"name", "code_dir", "python", "entrypoint", "output_dir", "output_root", "checkpoint_base_dir",
                     "exp_name", "resume", "overwrite", "wandb_enabled"}
_FIELD_PATH = re.compile(r"[A-Za-z][A-Za-z0-9_]*(?:(?:\.[A-Za-z][A-Za-z0-9_]*)|(?:\[(?:0|[1-9][0-9]?)\]))*")
_ASSET_FIELDS = {"repo_id", "repo_ids", "assets", "assets_dir", "asset_id", "asset_ids", "composition",
                 "norm_stats", "norm_stats_path", "norm_stats_dir", "norm_path", "norm_dir",
                 "normalization_path", "normalization_dir", "dataset_path", "dataset_dir",
                 "data_path", "data_dir", "rlds_data_dir", "filter_dict_path", "weight_loader", "weight_loaders", "init_weights",
                 "pretrained_path", "pretrained_dir", "weights_path", "weight_path",
                 "model_path", "model_dir", "checkpoint_path", "checkpoint_dir", "params_path", "params_dir"}


def _asset_managed_tokens(tokens: tuple) -> bool:
    if any(token in _ASSET_FIELDS for token in tokens if isinstance(token, str)):
        return True
    if tokens and tokens[0] in {"data", "model"} and "weights" in tokens:
        return True
    # The list (or a whole factory) could otherwise replace bound asset leaves.
    return bool(tokens and tokens[0] == "data" and "datasets" in tokens
                and (tokens[-1] == "datasets" or isinstance(tokens[-1], int) and tokens[-2] == "datasets"))


def is_asset_managed_field(path: str) -> bool:
    """Fields whose source of truth is the run's explicit asset bindings."""
    if not isinstance(path, str) or len(path) > 512 or _FIELD_PATH.fullmatch(path) is None:
        return False
    tokens = tuple(int(index) if index else name for name, index in re.findall(r"([A-Za-z][A-Za-z0-9_]*)|\[([0-9]+)\]", path))
    return _asset_managed_tokens(tokens)


def _contains_asset_managed_value(tokens, value, depth=0):
    if _asset_managed_tokens(tokens):
        return True
    if depth > 4:
        return False  # The literal validator rejects deeper structures.
    if isinstance(value, dict):
        return any(_contains_asset_managed_value((*tokens, key), item, depth + 1) for key, item in value.items())
    if isinstance(value, list):
        return any(_contains_asset_managed_value((*tokens, index), item, depth + 1) for index, item in enumerate(value))
    return False


def override_path_tokens(path: str) -> tuple:
    """Parse data-only field paths, without attribute lookup or expression evaluation."""
    if not isinstance(path, str) or not 1 <= len(path) <= 512:
        raise ValueError("RECIPE_OVERRIDES_INVALID: unsupported configuration field")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:(?:\.[A-Za-z][A-Za-z0-9_]*)|(?:\[(?:0|[1-9][0-9]?)\]))*", path):
        raise ValueError("RECIPE_OVERRIDES_INVALID: unsupported configuration field")
    tokens = tuple(int(index) if index else name for name, index in re.findall(r"([A-Za-z][A-Za-z0-9_]*)|\[([0-9]+)\]", path))
    if (len(tokens) > 16 or any(isinstance(token, int) and token >= 64 for token in tokens)
            or any(isinstance(token, str) and (token.startswith("_") or token in _PROTECTED_FIELDS) for token in tokens)
            or path not in OVERRIDE_RULES and (tokens[0] not in LITERAL_ROOTS or len(tokens) < 2)):
        raise ValueError("RECIPE_OVERRIDES_INVALID: unsupported configuration field")
    return tokens


def _literal_value(value, depth=0):
    if depth > 4:
        return False
    if value is None or isinstance(value, bool):
        return True
    if isinstance(value, str):
        return len(value) <= 1024 and not any(ord(char) < 32 and char not in "\n\t" for char in value)
    if isinstance(value, int):
        return abs(value) <= 2 ** 53 - 1
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return len(value) <= 64 and all(_literal_value(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return (len(value) <= 64 and all(isinstance(key, str) and len(key) <= 128 and not key.startswith("_")
                                       and _literal_value(item, depth + 1) for key, item in value.items()))
    return False


def _literal_shape(value, base):
    if base is None:
        return True  # An explicitly null optional value has no more precise static type.
    if isinstance(base, bool):
        return isinstance(value, bool)
    if isinstance(base, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(base, float):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(base, str):
        return isinstance(value, str)
    if isinstance(base, (list, tuple)):
        if not isinstance(value, list):
            return False
        if not base:
            return True
        if len(value) == len(base):
            return all(_literal_shape(item, old) for item, old in zip(value, base))
        # Homogeneous lists such as mixing weights may change length.
        return all(_literal_shape(item, base[0]) for item in [*base, *value])
    if isinstance(base, dict):
        return isinstance(value, dict) and set(value) == set(base) and all(_literal_shape(value[key], old) for key, old in base.items())
    return False


def validate_overrides(overrides: dict, *, allow_asset_managed=False) -> dict:
    if not isinstance(overrides, dict) or len(overrides) > 128:
        raise ValueError("RECIPE_OVERRIDES_INVALID: unsupported configuration field")
    clean = {}
    paths = []
    for path, value in overrides.items():
        tokens = override_path_tokens(path)
        if not allow_asset_managed and _contains_asset_managed_value(tokens, value):
            raise ValueError(f"RECIPE_ASSET_FIELDS_MANAGED: {path} is controlled by the run's asset bindings")
        if any(tokens[:len(other)] == other or other[:len(tokens)] == tokens for other in paths):
            raise ValueError("RECIPE_OVERRIDES_INVALID: overlapping configuration fields")
        paths.append(tokens)
        rule = OVERRIDE_RULES.get(path)
        if rule and (isinstance(value, bool) or not isinstance(value, (int, float)) or isinstance(value, float) and not math.isfinite(value)
                or rule["type"] == "integer" and not isinstance(value, int)
                or not rule["min"] <= value <= rule["max"]):
            raise ValueError(f"RECIPE_OVERRIDES_INVALID: invalid value for {path}")
        if path == "data.weights" and value is not None and (not isinstance(value, list) or not 1 <= len(value) <= 64
                or any(isinstance(item, bool) or not isinstance(item, (int, float))
                       or isinstance(item, float) and not math.isfinite(item) or item <= 0 for item in value)):
            raise ValueError("RECIPE_OVERRIDES_INVALID: data.weights requires positive finite mixing weights")
        if not _literal_value(value) or len(json.dumps(value, ensure_ascii=True, allow_nan=False)) > 8192:
            raise ValueError(f"RECIPE_OVERRIDES_INVALID: invalid literal for {path}")
        clean[path] = copy.deepcopy(value)
    if len(json.dumps(clean, ensure_ascii=True, allow_nan=False)) > 65536:
        raise ValueError("RECIPE_OVERRIDES_INVALID: configuration edits exceed their size limit")
    return clean


def validate_overrides_against_fields(overrides: dict, fields: list[dict]) -> dict:
    """Generic edits must name an observed literal; legacy numeric rules stay bounded."""
    clean = validate_overrides(overrides)
    declared = {item.get("path"): item for item in fields if isinstance(item, dict) and isinstance(item.get("path"), str)}
    for path, value in clean.items():
        if path in OVERRIDE_RULES:
            continue
        field = declared.get(path)
        if (field is None or field.get("kind") != "literal" or "value" not in field
                or path != "data.weights" and not _literal_shape(value, field["value"])):
            raise ValueError(f"RECIPE_FIELD_UNSUPPORTED: {path} must be an existing literal of the same type")
    return clean


def definition_digest(definition: dict) -> str:
    value = {key: value for key, value in definition.items() if key != "digest"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_definition(definition: dict) -> dict:
    fields = {"schema_version", "id", "recipe_id", "name", "revision", "base_recipe", "node_id", "profile_id", "source", "overrides", "digest"}
    if (not isinstance(definition, dict) or set(definition) != fields
            or type(definition.get("schema_version")) is not int or definition["schema_version"] != 1):
        raise ValueError("RECIPE_DEFINITION_INVALID: invalid versioned recipe")
    for key, bound in [("id", 36), ("recipe_id", 36), ("name", 200), ("node_id", 36), ("profile_id", 200)]:
        value = definition[key]
        if not isinstance(value, str) or not 1 <= len(value) <= bound or any(ord(c) < 32 or 127 <= ord(c) < 160 for c in value):
            raise ValueError("RECIPE_DEFINITION_INVALID: invalid recipe identity")
    if (isinstance(definition["revision"], bool) or not isinstance(definition["revision"], int) or definition["revision"] < 1
            or not isinstance(definition["base_recipe"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,255}", definition["base_recipe"])):
        raise ValueError("RECIPE_DEFINITION_INVALID: invalid base recipe or revision")
    source = definition["source"]
    if not isinstance(source, dict) or not isinstance(source.get("path"), str) or source["path"] not in RECIPE_SOURCE_PATHS:
        raise ValueError("RECIPE_DEFINITION_INVALID: invalid source path")
    for key, pattern in [("sha256", r"[a-f0-9]{64}"), ("changes_digest", r"[a-f0-9]{64}"), ("commit_sha", r"[a-f0-9]{40}|[a-f0-9]{64}")]:
        if not isinstance(source.get(key), str) or re.fullmatch(pattern, source[key]) is None:
            raise ValueError("RECIPE_DEFINITION_INVALID: invalid source version")
    code_dir = source.get("code_dir")
    if (not isinstance(code_dir, str) or not code_dir.startswith("/") or len(code_dir) > 4096 or ".." in code_dir.split("/")
            or any(ord(c) < 32 for c in code_dir) or not isinstance(source.get("dirty"), bool)):
        raise ValueError("RECIPE_DEFINITION_INVALID: invalid code directory")
    for key in ("line", "end_line"):
        if isinstance(source.get(key), bool) or not isinstance(source.get(key), int) or not 1 <= source[key] <= 1_000_000:
            raise ValueError("RECIPE_DEFINITION_INVALID: invalid source location")
    if source["line"] > source["end_line"]:
        raise ValueError("RECIPE_DEFINITION_INVALID: invalid source location")
    # Old revisions remain inspectable and keep their original digest. Execution
    # and mutation paths call the stricter validator before using their values.
    validate_overrides(definition["overrides"], allow_asset_managed=True)
    if definition.get("digest") != definition_digest(definition):
        raise ValueError("RECIPE_DEFINITION_INVALID: recipe content digest does not match")
    return copy.deepcopy(definition)


def resolve_definition(profile: dict, definition: dict, recipe: str | None = None, *, require_validated=True) -> str:
    definition = validate_definition(definition)
    validate_overrides(definition["overrides"])
    if (profile.get("adapter") != "openpi-sft" or definition["profile_id"] != profile.get("id")
            or recipe is not None and recipe != definition["base_recipe"]):
        raise ValueError("RECIPE_DEFINITION_MISMATCH: recipe does not belong to this training environment")
    approved = (profile.get("metadata") or {}).get("validated_recipe_revisions", [])
    if require_validated and (not isinstance(approved, list) or definition["digest"] not in approved):
        raise ValueError("PROFILE_NOT_VALIDATED: this saved recipe revision must be validated independently")
    return definition["base_recipe"]


def recipe_parameters(definition: dict | None) -> dict:
    overrides = definition["overrides"] if definition else {}
    return {name: overrides[path] for name, path in PARAMETER_PATHS.items() if path in overrides}


def apply_overrides(base, overrides: dict):
    """Replace dataclass and sequence leaves without mutating the base recipe."""
    overrides = validate_overrides(overrides)
    def replace(obj, tokens, value, legacy):
        if not tokens:
            if not legacy and not _literal_shape(value, obj):
                raise ValueError("RECIPE_FIELD_UNSUPPORTED: configuration leaf has a different type")
            if isinstance(obj, tuple) and isinstance(value, list):
                return tuple(value)
            return copy.deepcopy(value)
        token, *tail = tokens
        if isinstance(token, int) and isinstance(obj, (list, tuple)) and token < len(obj):
            result = list(obj)
            result[token] = replace(obj[token], tail, value, legacy)
            return tuple(result) if isinstance(obj, tuple) else result
        if isinstance(token, str) and dataclasses.is_dataclass(obj) and not isinstance(obj, type) and token in {f.name for f in dataclasses.fields(obj)}:
            return dataclasses.replace(obj, **{token: replace(getattr(obj, token), tail, value, legacy)})
        raise ValueError("RECIPE_FIELD_UNSUPPORTED: configuration field cannot apply these overrides")
    result = base
    for path, value in overrides.items():
        result = replace(result, override_path_tokens(path), value, path in OVERRIDE_RULES or path == "data.weights")
    return result
