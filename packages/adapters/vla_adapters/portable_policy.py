"""Bounded, non-executable inference configurations for verified checkpoints.

Only explicitly allowed local dataclasses can be reconstructed. Training data,
initial-weight paths, Python source and pickle objects are never transported.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
from types import SimpleNamespace


MAX_BYTES = 512 * 1024
MODEL_MODULES = frozenset({
    "openpi.models.pi0_config", "openpi.models.pi0_mem", "openpi.models.pi0_mem_compress",
    "openpi.models.pi0_mem_fixed_grid_temporal", "openpi.models.pi0_mem_fixed_grid_query_action",
    "openpi.models.pi0_mem_post_transformer", "openpi.models.pi0_mem_pf", "openpi.models.pi0_mem_pf_safe",
    "openpi.models.pi0_advantage", "openpi.models.pi0_gripper", "openpi.models.pi0_discrete",
})
DATACLASS_MODULES = MODEL_MODULES | frozenset({
    "openpi.transforms", "openpi.policies.aloha_policy", "openpi.policies.droid_policy",
    "openpi.policies.libero_policy", "openpi.policies.umi_policy", "vla_adapters.policy_inference",
})
ADAPTER_FILES = ("policy_inference.py", "composition_spec.py", "portable_policy.py")


class PortableRepresentationUnsupported(ValueError):
    """A trusted local config needs constructors the portable codec cannot express.

    Integrity, schema, normalization and numeric errors deliberately keep the
    ordinary ValueError type so callers cannot mistake corruption for this
    compatibility boundary.
    """


def _unrepresentable(message):
    raise PortableRepresentationUnsupported("PORTABLE_POLICY_UNSUPPORTED: " + message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def portable_digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _fail(message):
    raise ValueError("PORTABLE_POLICY_UNSUPPORTED: " + message)


def _class(module, name):
    if module not in DATACLASS_MODULES or not isinstance(name, str) or not name.isidentifier():
        _unrepresentable("Unsupported constructor " + str(module) + "." + str(name))
    value = getattr(importlib.import_module(module), name)
    if not isinstance(value, type) or not dataclasses.is_dataclass(value) or value.__module__ != module:
        _fail("Constructor is not a declared dataclass")
    if name == "Pi0ValueConfig":
        _fail("Value models are not action policies")
    return value


class _Codec:
    def __init__(self, code_dir=None):
        self.nodes = 0
        self.code_dir = Path(code_dir).resolve() if code_dir is not None else None

    def _origin(self, module):
        if self.code_dir is not None and module.startswith("openpi."):
            imported = importlib.import_module(module)
            expected = self.code_dir / "src" / (module.replace(".", "/") + ".py")
            if Path(imported.__file__).resolve() != expected:
                _fail("Imported module resolves outside its verified checkout: " + module)

    def _bound(self, depth):
        self.nodes += 1
        if self.nodes > 20000 or depth > 30:
            _fail("Configuration exceeds structural limits")

    def encode(self, value, depth=0):
        self._bound(depth)
        if value is None or type(value) in (bool, int, str):
            if isinstance(value, str) and len(value) > 16384:
                _fail("Configuration text is too long")
            return value
        if type(value) is float:
            if not math.isfinite(value):
                _fail("Configuration numbers must be finite")
            return value
        if isinstance(value, enum.Enum):
            if type(value).__module__ != "openpi.models.model" or type(value).__name__ != "ModelType":
                _unrepresentable("Unsupported enum")
            self._origin(type(value).__module__)
            return {"kind": "model_type", "value": value.value}
        module, name = type(value).__module__, type(value).__name__
        if module == "openpi.models.tokenizer" and name == "PaligemmaTokenizer":
            self._origin(module)
            tokenizer = value._tokenizer
            return {"kind": "paligemma_tokenizer", "max_len": value._max_len,
                    "vocabulary_sha256": hashlib.sha256(tokenizer.serialized_model_proto()).hexdigest()}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            _class(module, name)
            self._origin(module)
            return {"kind": "dataclass", "module": module, "name": name,
                    "fields": {field.name: self.encode(getattr(value, field.name), depth + 1)
                               for field in dataclasses.fields(value) if field.init}}
        if type(value) in (list, tuple, set, frozenset):
            if len(value) > 4096:
                _fail("Configuration sequence is too long")
            values = [self.encode(item, depth + 1) for item in value]
            if isinstance(value, (set, frozenset)):
                values.sort(key=canonical)
            return {"kind": type(value).__name__, "items": values}
        if type(value) is dict:
            if len(value) > 4096 or not all(isinstance(key, str) for key in value):
                _fail("Configuration maps require bounded string keys")
            return {"kind": "dict", "items": [[key, self.encode(item, depth + 1)] for key, item in sorted(value.items())]}
        # A handful of existing transforms carry literal NumPy masks. No tensor,
        # tokenizer state, callable or file path is generalized into executable code.
        if module == "numpy" and name == "ndarray":
            if value.size > 65536 or value.dtype.kind not in "biuf" or not value.dtype.isnative:
                _fail("Only bounded numeric NumPy literals are supported")
            return {"kind": "array", "dtype": str(value.dtype), "values": self.encode(value.tolist(), depth + 1)}
        if module.startswith("numpy") and hasattr(value, "item"):
            return self.encode(value.item(), depth + 1)
        _unrepresentable("Cannot safely export " + module + "." + name)

    def decode(self, value, depth=0):
        self._bound(depth)
        if value is None or type(value) in (bool, int, str):
            if isinstance(value, str) and len(value) > 16384:
                _fail("Configuration text is too long")
            return value
        if type(value) is float and math.isfinite(value):
            return value
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            _fail("Invalid configuration literal")
        kind = value["kind"]
        if kind == "dataclass":
            if set(value) != {"kind", "module", "name", "fields"} or not isinstance(value["fields"], dict):
                _fail("Invalid dataclass description")
            cls = _class(value["module"], value["name"])
            self._origin(value["module"])
            fields = {field.name for field in dataclasses.fields(cls) if field.init}
            if set(value["fields"]) != fields:
                _fail("Target dataclass fields differ from the exported environment")
            return cls(**{key: self.decode(item, depth + 1) for key, item in value["fields"].items()})
        if kind in {"list", "tuple", "set", "frozenset"}:
            if set(value) != {"kind", "items"} or not isinstance(value["items"], list) or len(value["items"]) > 4096:
                _fail("Invalid sequence")
            items = [self.decode(item, depth + 1) for item in value["items"]]
            return {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}[kind](items)
        if kind == "dict":
            items = value.get("items")
            if (set(value) != {"kind", "items"} or not isinstance(items, list) or len(items) > 4096
                    or any(not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str) for item in items)
                    or len({item[0] for item in items}) != len(items)):
                _fail("Invalid map")
            return {key: self.decode(item, depth + 1) for key, item in items}
        if kind == "model_type":
            if set(value) != {"kind", "value"}:
                _fail("Invalid model enum")
            self._origin("openpi.models.model")
            return importlib.import_module("openpi.models.model").ModelType(value["value"])
        if kind == "paligemma_tokenizer":
            if (set(value) != {"kind", "max_len", "vocabulary_sha256"} or type(value["max_len"]) is not int
                    or not 1 <= value["max_len"] <= 16384):
                _fail("Invalid tokenizer description")
            self._origin("openpi.models.tokenizer")
            tokenizer = importlib.import_module("openpi.models.tokenizer").PaligemmaTokenizer(value["max_len"])
            if hashlib.sha256(tokenizer._tokenizer.serialized_model_proto()).hexdigest() != value["vocabulary_sha256"]:
                _fail("Target tokenizer vocabulary differs from training")
            return tokenizer
        if kind == "array":
            if set(value) != {"kind", "dtype", "values"} or value["dtype"] not in {
                    "bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float32", "float64"}:
                _fail("Invalid array literal")
            import numpy as np
            result = np.asarray(self.decode(value["values"], depth + 1), dtype=value["dtype"])
            if result.size > 65536 or not np.isfinite(result).all():
                _fail("Array exceeds bounds or contains non-finite values")
            return result
        _fail("Unknown literal type")


def _code_files(code_dir):
    root = Path(code_dir).resolve(strict=True)
    package = root / "src/openpi"
    if not package.is_dir():
        _fail("Registered checkout has no src/openpi package")
    files, total = {}, 0
    for path in sorted(package.rglob("*.py")):
        if path.is_symlink() or not path.is_file() or path.resolve() != path or any(parent.is_symlink() for parent in path.parents if parent != root):
            _fail("OpenPI source must contain regular files without escaping links")
        total += path.stat().st_size
        if len(files) >= 4096 or path.stat().st_size > 8 * 1024 * 1024 or total > 32 * 1024 * 1024:
            _fail("OpenPI source inventory exceeds bounds")
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        _fail("OpenPI source inventory is empty")
    return files


def _adapter_files():
    root = Path(__file__).resolve().parent
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ADAPTER_FILES}


def _versions():
    result = {}
    for name in ("jax", "jaxlib", "flax", "numpy", "sentencepiece"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


@dataclasses.dataclass(frozen=True)
class PortableData:
    asset_id: str
    data_transforms: object
    model_transforms: object
    use_quantile_norm: bool
    normalize_masks: object
    robot_type: str | None


@dataclasses.dataclass(frozen=True)
class ResolvedDataFactory:
    value: PortableData

    def create(self, *_args):
        return self.value


def export_portable_policy(train_config, code_dir, *, resolved_data=None):
    """Resolve training's data factory once on its owning machine, without weights."""
    if type(train_config.model).__name__ == "Pi0ValueConfig":
        _fail("Value-only models are not action policies")
    if type(train_config.model).__module__ not in MODEL_MODULES:
        _unrepresentable("Unsupported model family")
    if getattr(train_config.data, "training_stage", None) == "value":
        _fail("Value-only training is not an action policy")
    if len(getattr(train_config.data, "datasets", ())) > 1:
        _fail("Multi-dataset inference requires an explicit normalization binding")
    from .policy_inference import compile_inference_config
    if resolved_data is None:
        compiled = compile_inference_config(train_config)
        resolved_data = compiled.data.create(train_config.assets_dirs, train_config.model)
    data = resolved_data
    asset_id = data.asset_id
    if (not isinstance(asset_id, str) or not asset_id or PurePosixPath(asset_id).is_absolute()
            or ".." in PurePosixPath(asset_id).parts):
        _fail("Checkpoint normalization asset id must be a relative directory")
    codec = _Codec(code_dir)
    result = {
        "schema_version": 1, "model": codec.encode(train_config.model),
        "data": {"asset_id": asset_id, "data_transforms": codec.encode(data.data_transforms),
                 "model_transforms": codec.encode(data.model_transforms),
                 "use_quantile_norm": bool(data.use_quantile_norm),
                 "normalize_masks": codec.encode(getattr(data, "normalize_masks", None)),
                 "robot_type": getattr(data, "robot_type", None)},
        "policy_metadata": codec.encode(train_config.policy_metadata),
        "code_files": _code_files(code_dir), "adapter_files": _adapter_files(), "runtime_versions": _versions(),
        "model_family": type(train_config.model).__name__,
    }
    if len(canonical(result)) > MAX_BYTES:
        _fail("Portable policy exceeds 512 KiB")
    return result


def validate_portable_policy(config, digest):
    if not isinstance(config, dict) or len(canonical(config)) > MAX_BYTES or portable_digest(config) != digest:
        _fail("Portable policy digest or size is invalid")
    if set(config) != {"schema_version", "model", "data", "policy_metadata", "code_files", "adapter_files", "runtime_versions", "model_family"} or config["schema_version"] != 1:
        _fail("Unsupported portable policy schema")
    return config


def load_portable_policy(config, digest, code_dir):
    """Rebuild local trusted constructors after checking source and runtime identity."""
    validate_portable_policy(config, digest)
    if config["code_files"] != _code_files(code_dir):
        _fail("Target OpenPI source differs from the exported training checkout")
    if config["adapter_files"] != _adapter_files():
        _fail("Target inference adapter version differs from the exported version")
    if config["runtime_versions"] != _versions():
        _fail("Target runtime package versions differ from the source inference environment")
    codec = _Codec(code_dir)
    model = codec.decode(config["model"])
    if type(model).__module__ not in MODEL_MODULES or type(model).__name__ != config["model_family"]:
        _fail("Model family does not match exported evidence")
    data = config["data"]
    if not isinstance(data, dict) or set(data) != {"asset_id", "data_transforms", "model_transforms", "use_quantile_norm", "normalize_masks", "robot_type"}:
        _fail("Invalid resolved inference data")
    asset_id = data["asset_id"]
    if (not isinstance(asset_id, str) or not asset_id or PurePosixPath(asset_id).is_absolute()
            or ".." in PurePosixPath(asset_id).parts or type(data["use_quantile_norm"]) is not bool
            or data["robot_type"] is not None and not isinstance(data["robot_type"], str)):
        _fail("Invalid checkpoint normalization or robot contract")
    resolved = PortableData(asset_id, codec.decode(data["data_transforms"]), codec.decode(data["model_transforms"]),
        data["use_quantile_norm"], codec.decode(data["normalize_masks"]), data["robot_type"])
    return SimpleNamespace(model=model, data=ResolvedDataFactory(resolved), assets_dirs=(),
                           policy_metadata=codec.decode(config["policy_metadata"]))
