"""Stable, reviewable Python built from the real merged training configuration."""

import ast
import dataclasses
import hashlib
import json
import math
import pprint
import re
from pathlib import Path

if __package__:
    from .automatic_config import normalize_auto_config
else:
    from automatic_config import normalize_auto_config


_VALUE_FIELDS = {"c_fail_coef", "value_clip_min", "value_clip_max", "episode_metadata_path", "task_max_lengths"}
_EVAL_FIELDS = {"val_ratio", "eval_interval", "eval_batches", "eval_batch_size"}
_PYTORCH_FIELDS = {"pytorch_weight_path", "pytorch_training_precision"}
_TASK_FIELDS = {"shellgame_cup_eval", "shellgame_memory_classifier", "uot"}


def _recipe_family(spec, entrypoint):
    """Classify registered entrypoints; custom entrypoints remain conservative."""
    selected = entrypoint or spec.get("entrypoint")
    if selected is None:
        return "value" if spec.get("training_stage") == "value" else "sft"
    path = Path(selected)
    if path.is_absolute():
        try:
            selected = str(path.relative_to(Path(spec["code_dir"])))
        except ValueError:
            return "custom"
    else:
        selected = path.as_posix()
    if selected in {"scripts/train.py", "scripts/train_multi_dataset.py"}:
        return "value" if spec.get("training_stage") == "value" else "sft"
    if selected in {"scripts/train_value.py", "scripts/train_value_multi_dataset.py",
                    "scripts/acp/train_value.py", "scripts/acp/train_value_multi_dataset.py"}:
        return "value"
    if selected in {"scripts/train_pi0_mem_compress.py", "scripts/mem/train_pi0_mem_compress.py"}:
        return "mem"
    if selected in {"scripts/train_uot.py", "scripts/uot/train_uot.py"}:
        return "uot"
    if selected == "scripts/train_pytorch.py":
        return "pytorch"
    return "custom"


def _literal_equal(left, right, depth=0):
    """Compare only inert literals; never invoke user equality or factories."""
    if depth > 12 or type(left) is not type(right):
        return False
    if left is None or type(left) in {bool, int, str}:
        return left == right
    if type(left) is float:
        return math.isfinite(left) and math.isfinite(right) and left == right
    if type(left) in {list, tuple}:
        return len(left) <= 128 and len(left) == len(right) and all(_literal_equal(a, b, depth + 1) for a, b in zip(left, right))
    if type(left) is dict:
        return (len(left) <= 128 and all(type(key) in {str, int} for key in left) and all(type(key) in {str, int} for key in right)
                and left.keys() == right.keys() and all(_literal_equal(left[key], right[key], depth + 1) for key in left))
    return False


def _declared_default(value, field):
    if field.default is not dataclasses.MISSING:
        return _literal_equal(value, field.default)
    # Empty built-in collections and dataclass-class factories can be checked
    # from their declarations. No factory, including an arbitrary lambda, runs.
    if any(field.default_factory is builtin for builtin in (list, tuple, dict)):
        return type(value) is field.default_factory and not value
    if field.default_factory is type(value) and dataclasses.is_dataclass(value):
        fields = [item for item in dataclasses.fields(value) if item.init]
        return all(item.default is not dataclasses.MISSING and
                   _literal_equal(getattr(value, item.name), item.default) for item in fields)
    return False


def _resolved_digest(config, output_root):
    """Bind omitted defaults to confirmation without repeating their contents."""
    count = 0

    def project(value, depth=0):
        nonlocal count
        count += 1
        if count > 7000 or depth > 10:
            return {"unresolved": "bounded_projection_limit"}
        if value is None or type(value) in {bool, int}:
            return value
        if type(value) is float:
            return value if math.isfinite(value) else {"unresolved": "non_finite"}
        if isinstance(value, (str, Path)):
            text = str(value)
            if text == output_root or text.startswith(output_root + "/"):
                text = "<runtime_output_dir>" + text[len(output_root):]
            return text if len(text) <= 4096 else {"value_prefix": text[:4096], "unresolved": "long_string"}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
                    **{field.name: project(getattr(value, field.name), depth + 1)
                       for field in dataclasses.fields(value)[:128]}}
        if type(value) in {list, tuple}:
            rows = [project(item, depth + 1) for item in value[:128]]
            if len(value) > 128:
                rows.append({"unresolved": "sequence_truncated"})
            return rows
        if type(value) is dict:
            return {str(key)[:255]: project(item, depth + 1) for key, item in list(value.items())[:128]
                    if type(key) in {str, int}}
        return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
                "unresolved": "code_object_from_registered_project_and_runtime"}

    encoded = json.dumps(project(config), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def generated_training_config(config, spec, *, entrypoint=None):
    """Expose resolved values and a faithful builder without serializing code.

    Dynamic functions, filters and constructors keep their real source/runtime
    dependency. The Python file deliberately uses the same trusted resolver;
    its bounded projection is explanatory, never an executable replacement.
    """
    output_root = str(Path(spec["output_dir"]).resolve())
    family = _recipe_family(spec, entrypoint)
    parameters = spec.get("parameters") or {}
    root_explicit = {"steps": "num_train_steps", "learning_rate": "lr_schedule", "warmup_steps": "lr_schedule",
                     "decay_steps": "lr_schedule", "decay_lr": "lr_schedule", "clip_gradient_norm": "optimizer"}
    explicit_fields = {root_explicit.get(key, key) for key in parameters}
    explicit_fields.update(spec.get("stage_config") or {})

    def review_fields(value, path):
        """Show active overrides and required constructors, without task noise."""
        result = []
        for field in dataclasses.fields(value):
            if not field.init:
                continue
            item = getattr(value, field.name)
            required = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
            pinned = required
            if not path:
                known_root = type(value).__module__ == "openpi.training.config" and type(value).__name__ == "TrainConfig"
                if known_root and not required and family != "custom":
                    if field.name in _VALUE_FIELDS and family != "value":
                        continue
                    if field.name in _EVAL_FIELDS and family not in {"value", "mem"}:
                        continue
                    if field.name in _PYTORCH_FIELDS and family != "pytorch":
                        continue
                    if field.name == "log_perf_metrics" and family not in {"mem", "uot"}:
                        continue
                    if field.name in _TASK_FIELDS and (item is None or getattr(item, "enabled", None) is False):
                        continue
                pinned |= field.name in explicit_fields or field.name in {"name", "model", "data", "weight_loader", "optimizer", "lr_schedule"}
            if path == ("model",):
                pinned |= field.name in {"pi05", "action_dim", "action_horizon"}
            if (type(value).__module__ == "openpi.training.config"
                    and type(value).__name__ == "ComposableDataConfig" and family != "custom"):
                if field.name == "acp_dropout" and not required and value.training_stage != "offline_rl":
                    continue
                pinned |= field.name in {"training_stage", "acp_dropout"}
            # An ACP-specific legacy/custom processor may use dropout without
            # a stage switch. Preserve its knob instead of guessing it is idle.
            if field.name == "acp_dropout":
                pinned = True
            if path == ("lr_schedule",):
                pinned |= any(field.name == child and parameter in parameters for parameter, child in {
                    "learning_rate": "peak_lr", "warmup_steps": "warmup_steps", "decay_steps": "decay_steps",
                    "decay_lr": "decay_lr"}.items())
            if path == ("optimizer",) and "clip_gradient_norm" in parameters:
                pinned |= field.name == "clip_gradient_norm"
            if not pinned and _declared_default(item, field):
                continue
            result.append(field)
        return result

    # Keep the complete validated request needed by prepare_config. Exclude the
    # runtime attempt path and digest (which would otherwise be self-referential).
    keys = ("code_dir", "recipe", "inputs", "parameters", "gpu_count", "dataset_bindings", "auto_config",
            "recipe_definition", "entrypoint", "template_id", "training_stage", "stage_config", "allowed_roots")
    frozen = {key: spec[key] for key in keys if key in spec}
    automatic = normalize_auto_config(spec.get("auto_config"))
    if automatic is not None:
        frozen["auto_config"] = automatic
    freeze_source = ("model.get_freeze_filter() for LoRA; EMA disabled" if automatic and automatic["finetune_mode"] == "lora"
                     else "TrainConfig default freeze_filter; full value backbone and value head, not head-only training" if automatic and spec.get("training_stage") == "value"
                     else "TrainConfig default freeze_filter (full model)" if automatic
                     else "registered code recipe and its pinned overrides; see effective freeze_filter type")
    imports = {}
    aliases = set()
    render_count = 0

    def native(value, path=(), depth=0):
        """Render literal dataclasses; opaque objects refer to the real resolver."""
        nonlocal render_count
        render_count += 1
        fallback = "_at(_resolved, " + repr(path) + ")"
        if render_count > 7000 or depth > 10:
            return fallback
        if value is None or type(value) in (bool, int):
            return repr(value)
        if type(value) is float:
            return repr(value) if math.isfinite(value) else fallback
        if isinstance(value, (str, Path)):
            text = str(value)
            if len(text) > 4096:
                return fallback
            if text == output_root or text.startswith(output_root + "/"):
                suffix = text[len(output_root):].lstrip("/")
                expression = "Path(output_dir)" + (" / " + repr(suffix) if suffix else "")
                return expression if isinstance(value, Path) else "str(" + expression + ")"
            return "Path(" + repr(text) + ")" if isinstance(value, Path) else repr(text)
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            cls = type(value)
            if (re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", cls.__module__) is None
                    or re.fullmatch(r"[A-Za-z_]\w*", cls.__qualname__) is None):
                return fallback
            identity = (cls.__module__, cls.__qualname__)
            if identity not in imports:
                alias = cls.__name__
                if alias in aliases or alias in {"Path", "copy", "prepare_config"}:
                    alias += "_" + str(len(imports))
                aliases.add(alias)
                imports[identity] = alias
            values = review_fields(value, path)
            if len(values) > 128:
                return fallback
            indent = "    " * (depth + 1)
            arguments = [indent + field.name + "=" + native(getattr(value, field.name), path + (field.name,), depth + 1)
                         for field in values]
            return imports[identity] + "(\n" + ",\n".join(arguments) + ",\n" + "    " * depth + ")" if arguments else imports[identity] + "()"
        if isinstance(value, (tuple, list)) and len(value) <= 128:
            items = [native(item, path + (i,), depth + 1) for i, item in enumerate(value)]
            if isinstance(value, tuple):
                return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
            return "[" + ", ".join(items) + "]"
        if isinstance(value, dict) and len(value) <= 128 and all(type(key) in (str, int) for key in value):
            return "{" + ", ".join(repr(key) + ": " + native(item, path + (key,), depth + 1)
                                    for key, item in value.items()) + "}"
        return fallback

    # This native constructor view is generated from actual resolved objects,
    # not from recipe-name heuristics. The exported builder uses the resolver
    # as the authority, so opaque code objects are never replaced by this view.
    native_constructor = native(config)
    resolved_digest = _resolved_digest(config, output_root)
    imported = "".join(f"from {module} import {name}" + (f" as {alias}" if alias != name else "") + "\n"
                       for (module, name), alias in imports.items())
    format_value = lambda value: pprint.pformat(value, width=100, sort_dicts=True, compact=False)
    source = ('"""本次训练配置。运行输出由平台分配，通过 build_config(output_dir) 绑定。\n\n'
              'Requires the registered OpenPI project/runtime and platform adapter.\n'
              'Native constructors are a bounded review projection of this recipe, not standalone model code.\n'
              'Only active recipe settings and explicit overrides are expanded; other defaults remain in the pinned project.\n'
              'Unresolved filters/functions/constructors retain their stated source dependencies.\n'
              '"""\n\n'
              'import copy\nfrom pathlib import Path\n\n' + imported + '\n'
              'from vla_adapters.openpi_entry import prepare_config\n\n'
              'def native_configuration(output_dir, _resolved):\n'
              '    """Native constructors for review; opaque values depend on the checked resolver."""\n'
              '    # Freeze rule: ' + freeze_source + '\n'
              '    return ' + native_constructor.replace('\n', '\n    ') + '\n\n'
              'def _at(value, path):\n'
              '    for key in path:\n'
              '        value = value[key] if isinstance(value, (dict, list, tuple)) else getattr(value, key)\n'
              '    return value\n\n'
              'FROZEN_REQUEST = ' + format_value(frozen) + '\n\n'
              '# Bind all resolved defaults to this review without expanding them.\n'
              'RESOLVED_CONFIGURATION_SHA256 = ' + repr(resolved_digest) + '\n\n'
              'def build_config(output_dir):\n'
              '    """Rebuild through the same checked resolver; this does not start training."""\n'
              '    # The resolver installs per-dataset registered FPS for actual LeRobot loading.\n'
              '    # Source metadata/timestamps stay unchanged; this does not resample frames.\n'
              '    spec = copy.deepcopy(FROZEN_REQUEST)\n'
              '    spec["output_dir"] = str(Path(output_dir).resolve())\n'
              '    config, _ = prepare_config(spec)\n'
              '    return config\n')
    if len(source.encode()) > 384 * 1024:
        raise ValueError("GENERATED_CONFIG_LIMIT: 本次配置代码超过可审阅大小，请减少单次绑定数据组")
    ast.parse(source)
    return {"schema_version": 1, "language": "python", "filename": "training_config.py", "source": source,
            "sha256": hashlib.sha256(source.encode()).hexdigest()}
