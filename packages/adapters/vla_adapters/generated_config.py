"""Stable, reviewable Python built from the real merged training configuration."""

import ast
import dataclasses
import hashlib
import math
import pprint
import re
from itertools import islice
from pathlib import Path

if __package__:
    from .automatic_config import normalize_auto_config
    from .dataset_fps import resolve_registered_fps
else:
    from automatic_config import normalize_auto_config
    from dataset_fps import resolve_registered_fps


def generated_training_config(config, spec):
    """Expose resolved values and a faithful builder without serializing code.

    Dynamic functions, filters and constructors keep their real source/runtime
    dependency. The Python file deliberately uses the same trusted resolver;
    its bounded projection is explanatory, never an executable replacement.
    """
    output_root = str(Path(spec["output_dir"]).resolve())
    count = 0

    def project(value, depth=0):
        nonlocal count
        count += 1
        if count > 7000 or depth > 10:
            return {"unresolved": "bounded_projection_limit"}
        if value is None or isinstance(value, (bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {"unresolved": "non_finite"}
        if isinstance(value, (str, Path)):
            value = str(value)
            if value == output_root or value.startswith(output_root + "/"):
                value = "<runtime_output_dir>" + value[len(output_root):]
            return value if len(value) <= 4096 else {"value_prefix": value[:4096], "unresolved": "long_string"}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
                    **{field.name: project(getattr(value, field.name), depth + 1)
                       for field in dataclasses.fields(value)[:128]}}
        if isinstance(value, (tuple, list)):
            rows = [project(item, depth + 1) for item in value[:128]]
            if len(value) > 128:
                rows.append({"unresolved": "sequence_truncated"})
            return tuple(rows) if isinstance(value, tuple) else rows
        if isinstance(value, dict):
            return {str(key)[:255]: project(item, depth + 1) for key, item in islice(value.items(), 128)
                    if isinstance(key, (str, int))}
        return {"python_type": type(value).__module__ + "." + type(value).__qualname__,
                "unresolved": "code_object_from_registered_project_and_runtime"}

    # Keep the complete validated request needed by prepare_config. Exclude the
    # runtime attempt path and digest (which would otherwise be self-referential).
    keys = ("code_dir", "recipe", "inputs", "parameters", "gpu_count", "dataset_bindings", "auto_config",
            "recipe_definition", "entrypoint", "template_id", "training_stage", "stage_config", "allowed_roots")
    frozen = {key: spec[key] for key in keys if key in spec}
    effective = project(config)
    automatic = normalize_auto_config(spec.get("auto_config"))
    if automatic is not None:
        frozen["auto_config"] = automatic
    freeze_source = ("model.get_freeze_filter() for LoRA; EMA disabled" if automatic and automatic["finetune_mode"] == "lora"
                     else "TrainConfig default freeze_filter; full value backbone and value head, not head-only training" if automatic and spec.get("training_stage") == "value"
                     else "TrainConfig default freeze_filter (full model)" if automatic
                     else "registered code recipe and its pinned overrides; see effective freeze_filter type")
    dependencies = {"code_dir": spec["code_dir"], "named_code_recipe": None if automatic else spec["recipe"],
                    "dataset_fps_priority": "registered dataset fps; metadata fallback only when not configured",
                    "dataset_fps_policy": resolve_registered_fps(spec),
                    "saved_recipe_source": (spec.get("recipe_definition") or {}).get("source"),
                    "freeze_rule_source": freeze_source,
                    "optimizer_source": "effective optimizer/lr_schedule objects below; constructors remain in the registered project",
                    "execution": "requires the registered OpenPI checkout, its Python dependencies and bundled vla_adapters",
                    "scope": "resolved dataclass values before data-factory execution; unresolved code objects retain source dependencies"}
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
            values = [field for field in dataclasses.fields(value) if field.init]
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
    imported = "".join(f"from {module} import {name}" + (f" as {alias}" if alias != name else "") + "\n"
                       for (module, name), alias in imports.items())
    format_value = lambda value: pprint.pformat(value, width=100, sort_dicts=True, compact=False)
    source = ('"""本次训练配置。运行输出由平台分配，通过 build_config(output_dir) 绑定。\n\n'
              'Requires the registered OpenPI project/runtime and platform adapter.\n'
              'EFFECTIVE_CONFIGURATION is a bounded review projection, not standalone executable model code.\n'
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
              'DEPENDENCIES = ' + format_value(dependencies) + '\n\n'
              'EFFECTIVE_CONFIGURATION = ' + format_value(effective) + '\n\n'
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
