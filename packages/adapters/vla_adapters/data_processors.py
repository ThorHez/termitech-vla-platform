"""Registered data-processing factories and bounded, data-only constructor edits.

Catalog inspection never instantiates a factory or calls a default_factory.
Construction happens only in the registered training interpreter's preflight
and launch, and uses classes already exported by its local configuration module.
"""

import argparse
import ast
import collections.abc
import copy
import dataclasses
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import types
import typing

if __package__:
    from .composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from .dataset_contracts import static_dataset_contract
    from .recipe_definitions import is_asset_managed_field, validate_overrides
else:
    from composition_spec import COMPOSABLE_PROCESSOR, normalize_composition
    from dataset_contracts import static_dataset_contract
    from recipe_definitions import is_asset_managed_field, validate_overrides


_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,199}")
_UNSET = object()
_PRIMITIVES = {"bool": "boolean", "int": "integer", "float": "number", "str": "string", "None": "null", "NoneType": "null"}


def normalize_processor(value):
    if not isinstance(value, dict) or set(value) != {"name", "parameters"}:
        raise ValueError("DATA_PROCESSOR_INVALID: 数据处理器需要名称和参数对象")
    if not isinstance(value["name"], str) or _NAME.fullmatch(value["name"]) is None:
        raise ValueError("DATA_PROCESSOR_INVALID: 请选择已登记代码导出的简单处理器类名")
    parameters = value["parameters"]
    if not isinstance(parameters, dict) or len(parameters) > 128:
        raise ValueError("DATA_PROCESSOR_INVALID: 数据处理参数必须是受限的字段对象")
    if value["name"] == COMPOSABLE_PROCESSOR:
        if set(parameters) != {"composition"}:
            raise ValueError("COMPOSITION_INVALID: 组合处理器仅接受完整 composition，训练阶段由任务管理")
        return {"name": COMPOSABLE_PROCESSOR, "parameters": {"composition": normalize_composition(parameters["composition"])}}
    try:
        clean = validate_overrides({"data." + key: item for key, item in parameters.items()
                                    if isinstance(key, str)})
        if len(clean) != len(parameters):
            raise ValueError("invalid parameter path")
    except ValueError as exc:
        if str(exc).startswith("RECIPE_ASSET_FIELDS_MANAGED:"):
            raise ValueError("DATA_PROCESSOR_ASSET_FIELDS_MANAGED: 数据、归一化、模型路径及混采权重由任务资产绑定管理") from exc
        raise ValueError("DATA_PROCESSOR_INVALID: 参数只支持有界 JSON 字面量和普通字段路径") from exc
    return {"name": value["name"], "parameters": {path[5:]: item for path, item in clean.items()}}


def _source(cls, code_dir):
    root = Path(code_dir).resolve(strict=True)
    filename = inspect.getsourcefile(cls)
    if not filename:
        raise ValueError("DATA_PROCESSOR_UNSUPPORTED: 数据处理类缺少可追溯的本机源码")
    path = Path(filename).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("DATA_PROCESSOR_OUTSIDE_CODE_DIR: 数据处理类必须来自登记的训练代码目录")
    return {"path": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _class(config_module, name, code_dir):
    base = getattr(config_module, "DataConfigFactory", None)
    cls = vars(config_module).get(name)
    if (not inspect.isclass(base) or not inspect.isclass(cls) or cls is base
            or not issubclass(cls, base) or not dataclasses.is_dataclass(cls)
            or inspect.isabstract(cls) or cls.__name__ != name):
        raise ValueError("DATA_PROCESSOR_UNSUPPORTED: 处理器必须是配置模块导出的具体 DataConfigFactory 数据类")
    _source(cls, code_dir)
    if any(ancestor.__name__ in {"FakeDataConfig", "RLDSDroidDataConfig", "MultiDataConfigFactory"}
           for ancestor in cls.__mro__) or any(field.name == "datasets" for field in dataclasses.fields(cls)):
        raise ValueError("DATA_PROCESSOR_UNSUPPORTED: 合成数据、RLDS 和多数据集容器不能作为单个 LeRobot 处理器")
    return cls


def _union(schemas):
    if not schemas or any(schema is None for schema in schemas):
        return None
    nullable = any(schema["type"] == "null" for schema in schemas)
    live = [schema for schema in schemas if schema["type"] != "null"]
    if not live:
        return {"type": "null", "nullable": True}
    result = copy.deepcopy(live[0])
    if any(schema != live[0] for schema in live[1:]):
        if {schema["type"] for schema in live} <= {"integer", "number"}:
            result = {"type": "number"}
        else:
            return None
    if nullable:
        result["nullable"] = True
    return result


def _annotation_node(node, depth=0):
    if depth > 8:
        return None
    if isinstance(node, ast.Name):
        return {"type": _PRIMITIVES[node.id]} if node.id in _PRIMITIVES else None
    if isinstance(node, ast.Constant):
        if node.value is None:
            return {"type": "null", "nullable": True}
        if isinstance(node.value, str):
            try:
                return _annotation_node(ast.parse(node.value, mode="eval").body, depth + 1)
            except (SyntaxError, RecursionError):
                return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _union([_annotation_node(node.left, depth + 1), _annotation_node(node.right, depth + 1)])
    if not isinstance(node, ast.Subscript):
        return None
    name = node.value.id if isinstance(node.value, ast.Name) else node.value.attr if isinstance(node.value, ast.Attribute) else ""
    args = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
    if name in {"Suppress", "Annotated"} and args:
        return _annotation_node(args[0], depth + 1)
    if name in {"Optional", "Union"}:
        return _union([*[_annotation_node(arg, depth + 1) for arg in args], *([{"type": "null"}] if name == "Optional" else [])])
    if name == "Literal":
        try:
            choices = [ast.literal_eval(arg) for arg in args]
        except (ValueError, TypeError):
            return None
        if not choices or any(type(choice) not in (bool, int, float, str, type(None)) for choice in choices):
            return None
        schema = _union([_inferred(choice) for choice in choices])
        return {**schema, "choices": choices} if schema is not None else None
    if name in {"list", "List", "Sequence", "tuple", "Tuple"} and args:
        item = _annotation_node(args[0], depth + 1)
        if item is None:
            return None
        if len(args) > 1 and not (len(args) == 2 and isinstance(args[1], ast.Constant) and args[1].value is Ellipsis):
            return None
        return {"type": "array", "items": item, **({"container": "tuple"} if name in {"tuple", "Tuple"} else {})}
    if name in {"dict", "Dict", "Mapping"} and len(args) == 2 and _annotation_node(args[0], depth + 1) == {"type": "string"}:
        item = _annotation_node(args[1], depth + 1)
        return {"type": "object", "values": item} if item else None
    return None


def _schema(annotation, depth=0):
    if depth > 8:
        return None
    if isinstance(annotation, str):
        if len(annotation) > 1024:
            return None
        try:
            return _annotation_node(ast.parse(annotation, mode="eval").body, depth + 1)
        except (SyntaxError, RecursionError):
            return None
    if annotation in (bool, int, float, str, type(None)):
        return {"type": _PRIMITIVES[annotation.__name__]}
    origin, args = typing.get_origin(annotation), typing.get_args(annotation)
    if origin is typing.Annotated:
        return _schema(args[0], depth + 1)
    if origin in (typing.Union, types.UnionType):
        return _union([_schema(arg, depth + 1) for arg in args])
    if origin is typing.Literal:
        schema = _union([_inferred(item) for item in args])
        return {**schema, "choices": list(args)} if schema is not None else None
    if origin in (list, tuple, typing.Sequence, collections.abc.Sequence) and args:
        item = _schema(args[0], depth + 1)
        return {"type": "array", "items": item, **({"container": "tuple"} if origin is tuple else {})} if item else None
    if origin in (dict, typing.Mapping, collections.abc.Mapping) and len(args) == 2 and args[0] is str:
        item = _schema(args[1], depth + 1)
        return {"type": "object", "values": item} if item else None
    return None


def _inferred(value):
    if type(value) in (bool, int, float, str, type(None)):
        return {"type": _PRIMITIVES[type(value).__name__]}
    if isinstance(value, (list, tuple)) and value:
        item = _union([_inferred(item) for item in value])
        return {"type": "array", "items": item, **({"container": "tuple"} if isinstance(value, tuple) else {})} if item else None
    if isinstance(value, dict) and value and all(isinstance(key, str) for key in value):
        item = _union([_inferred(item) for item in value.values()])
        return {"type": "object", "values": item} if item else None
    return None


def _json_default(value, depth=0):
    if depth > 4:
        return _UNSET
    if type(value) in (bool, int, float, str, type(None)):
        return value
    if isinstance(value, (list, tuple)) and len(value) <= 64:
        result = [_json_default(item, depth + 1) for item in value]
        return result if all(item is not _UNSET for item in result) else _UNSET
    if isinstance(value, dict) and len(value) <= 64 and all(isinstance(key, str) for key in value):
        result = {key: _json_default(item, depth + 1) for key, item in value.items()}
        return result if all(item is not _UNSET for item in result.values()) else _UNSET
    return _UNSET


def _parameters(cls, config_module, depth=0):
    parameters, unsupported = [], []
    for field in dataclasses.fields(cls):
        if not field.init or is_asset_managed_field("data." + field.name):
            continue
        if field.name == "base_config":
            if depth >= 1:
                unsupported.append({"path": field.name, "required": False, "reason": "不支持递归 base_config 参数"})
                continue
            if field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
                unsupported.append({"path": field.name, "required": True, "reason": "必填 base_config 需要 Python DataConfig 对象，无法使用普通参数构造"})
            base_type = type(field.default) if dataclasses.is_dataclass(field.default) and not isinstance(field.default, type) else getattr(config_module, "DataConfig", None)
            if inspect.isclass(base_type) and dataclasses.is_dataclass(base_type):
                child, skipped = _parameters(base_type, config_module, depth + 1)
                for item in child:
                    nested = {**item, "path": "base_config." + item["path"], "required": False}
                    if field.default_factory is not dataclasses.MISSING:
                        # An unknown factory may change every DataConfig default.
                        # Let the normal constructor supply it during preflight.
                        nested.pop("default", None)
                    elif dataclasses.is_dataclass(field.default) and not isinstance(field.default, type):
                        nested.pop("default", None)
                        actual = _json_default(getattr(field.default, item["path"]))
                        if actual is not _UNSET:
                            try:
                                normalize_processor({"name": cls.__name__, "parameters": {nested["path"]: actual}})
                                nested["default"] = actual
                            except ValueError:
                                pass
                    parameters.append(nested)
                unsupported.extend({**item, "path": "base_config." + item["path"], "required": False} for item in skipped)
            continue
        required = field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING
        schema = _schema(field.type)
        default = _json_default(field.default) if field.default is not dataclasses.MISSING else _UNSET
        if schema is None and default is not _UNSET and default is not None:
            schema = _inferred(default)
        if schema is None:
            unsupported.append({"path": field.name, "required": required, "reason": "需要 Python 对象或无法静态确定参数类型；保留类默认值"})
            continue
        descriptor = {"path": field.name, **schema, "required": required}
        if default is not _UNSET:
            try:
                normalize_processor({"name": cls.__name__, "parameters": {field.name: default}})
            except ValueError:
                default = _UNSET
            if default is not _UNSET:
                descriptor["default"] = default
        parameters.append(descriptor)
    return parameters[:128], unsupported[:128]


def _descriptor(cls, config_module, name, code_dir):
    if name == COMPOSABLE_PROCESSOR:
        result = {"name": name, "description": (inspect.getdoc(cls) or "")[:2000], "parameters": [],
                  "unsupported_parameters": [], "available": False}
        try:
            _class(config_module, name, code_dir)
            if (type(getattr(cls, "COMPOSITION_SCHEMA_VERSION", None)) is not int
                    or cls.COMPOSITION_SCHEMA_VERSION != 1 or not callable(getattr(cls, "processing_contract", None))):
                raise ValueError("组合处理器不支持配置版本 1")
            result.update(source=_source(cls, code_dir), available=True, composition_schema_version=1)
        except (ValueError, OSError, TypeError) as exc:
            result["reason"] = str(exc).split(":", 1)[-1].strip()
        return result
    parameters, unsupported = _parameters(cls, config_module)
    required_unsupported = [item["path"] for item in unsupported if item["required"]]
    reason = None
    try:
        _class(config_module, name, code_dir)
    except (ValueError, OSError, TypeError) as exc:
        reason = str(exc).split(":", 1)[-1].strip()
    if required_unsupported:
        reason = "缺少可编辑的必填参数：" + ", ".join(required_unsupported)
    result = {"name": name, "description": (inspect.getdoc(cls) or "")[:2000], "parameters": parameters,
              "unsupported_parameters": unsupported, "available": reason is None}
    if reason:
        result["reason"] = reason
    try:
        result["source"] = _source(cls, code_dir)
    except (ValueError, OSError, TypeError):
        pass
    if reason is None:
        contract = static_dataset_contract(cls, code_dir)
        if contract is not None:
            result["dataset_contract"] = contract
    return result


def catalog_data_processors(config_module, code_dir):
    base = getattr(config_module, "DataConfigFactory", None)
    if not inspect.isclass(base):
        raise ValueError("DATA_PROCESSOR_UNSUPPORTED: 配置模块没有导出 DataConfigFactory")
    result = []
    for name, cls in sorted(vars(config_module).items()):
        if (_NAME.fullmatch(name) and inspect.isclass(cls) and cls is not base and cls.__name__ == name
                and issubclass(cls, base) and dataclasses.is_dataclass(cls) and not inspect.isabstract(cls)):
            result.append(_descriptor(cls, config_module, name, code_dir))
            if len(result) >= 128:
                break
    return result


def _coerce(value, schema, path):
    if value is None:
        if schema.get("nullable") or schema["type"] == "null":
            return None
        raise ValueError(f"DATA_PROCESSOR_PARAMETERS_INVALID: {path} 不接受 null")
    kind = schema["type"]
    valid = ((kind == "boolean" and isinstance(value, bool))
             or (kind == "integer" and type(value) is int)
             or (kind == "number" and type(value) in (int, float))
             or (kind == "string" and isinstance(value, str))
             or (kind == "array" and isinstance(value, list))
             or (kind == "object" and isinstance(value, dict)))
    if not valid or "choices" in schema and value not in schema["choices"]:
        raise ValueError(f"DATA_PROCESSOR_PARAMETERS_INVALID: {path} 参数类型或可选值不匹配")
    if kind == "array":
        result = [_coerce(item, schema["items"], path) for item in value]
        return tuple(result) if schema.get("container") == "tuple" else result
    if kind == "object":
        return {key: _coerce(item, schema["values"], path) for key, item in value.items()}
    return copy.deepcopy(value)


def instantiate_data_processor(processor, config_module, code_dir, *, dataset_path=None, norm_dir=None,
                               training_stage=None, stage_config=None):
    processor = normalize_processor(processor)
    cls = _class(config_module, processor["name"], code_dir)
    descriptor = _descriptor(cls, config_module, processor["name"], code_dir)
    if not descriptor["available"]:
        raise ValueError("DATA_PROCESSOR_UNSUPPORTED: " + descriptor["reason"])
    fields = {item["path"]: item for item in descriptor["parameters"]}
    missing = [path for path, field in fields.items() if field["required"] and path not in processor["parameters"]]
    if missing:
        raise ValueError("DATA_PROCESSOR_PARAMETERS_REQUIRED: 缺少数据处理参数 " + ", ".join(missing))
    kwargs, base_updates = {}, {}
    if processor["name"] == COMPOSABLE_PROCESSOR:
        kwargs.update(composition=processor["parameters"]["composition"], training_stage=training_stage or "sft")
        if training_stage == "offline_rl" and stage_config and "acp_dropout" in stage_config:
            kwargs["acp_dropout"] = stage_config["acp_dropout"]
    for path, value in processor["parameters"].items():
        if processor["name"] == COMPOSABLE_PROCESSOR and path == "composition":
            continue
        if path not in fields:
            raise ValueError(f"DATA_PROCESSOR_PARAMETERS_INVALID: 未支持的数据处理参数 {path}")
        value = _coerce(value, fields[path], path)
        if path.startswith("base_config."):
            base_updates[path[12:]] = value
        else:
            kwargs[path] = value
    managed = {field.name: field for field in dataclasses.fields(cls)}
    if "repo_id" not in managed or not managed["repo_id"].init or "assets" not in managed or not managed["assets"].init:
        raise ValueError("DATA_PROCESSOR_UNSUPPORTED: 数据处理构造器必须支持资产目录绑定")
    kwargs["repo_id"] = str(dataset_path) if dataset_path is not None else ""
    kwargs["assets"] = config_module.AssetsConfig(assets_dir=str(norm_dir) if norm_dir is not None else None, asset_id=".")
    try:
        factory = cls(**kwargs)
        if base_updates:
            base = getattr(factory, "base_config", None)
            if base is None:
                base = config_module.DataConfig()
            if not dataclasses.is_dataclass(base) or isinstance(base, type):
                raise ValueError("DATA_PROCESSOR_UNSUPPORTED: base_config 必须是 DataConfig 数据类")
            factory = dataclasses.replace(factory, base_config=dataclasses.replace(base, **base_updates))
    except TypeError as exc:
        raise ValueError("DATA_PROCESSOR_PARAMETERS_INVALID: 数据处理器无法使用所选参数构造") from exc
    return factory


def processor_identity(factory, code_dir, name=None):
    return {"name": name or type(factory).__name__, "module": type(factory).__module__, "source": _source(type(factory), code_dir)}


def validate_processor_instance(factory, processor):
    processor = normalize_processor(processor)
    if type(factory).__name__ != processor["name"]:
        raise ValueError("DATA_PROCESSOR_BINDING_MISMATCH: 实际数据处理类与本次选择不一致")
    for path, expected in processor["parameters"].items():
        value = factory
        for key in path.split("."):
            if not dataclasses.is_dataclass(value) or isinstance(value, type) or key not in {field.name for field in dataclasses.fields(value)}:
                raise ValueError(f"DATA_PROCESSOR_BINDING_MISMATCH: 实际处理器缺少参数 {path}")
            value = getattr(value, key)
        actual = _json_default(value)
        if actual is _UNSET or json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
            raise ValueError(f"DATA_PROCESSOR_BINDING_MISMATCH: 实际数据处理参数 {path} 与本次选择不一致")
    return processor


def _catalog(spec):
    if not isinstance(spec, dict) or set(spec) != {"code_dir", "allowed_roots"}:
        raise ValueError("DATA_PROCESSOR_INVALID: 处理器目录查询需要代码目录和受管目录")
    code = Path(spec["code_dir"]).resolve(strict=True)
    roots = spec["allowed_roots"]
    if (not Path(spec["code_dir"]).is_absolute() or not isinstance(roots, list) or not roots
            or not any(code.is_relative_to(Path(root).resolve(strict=True)) for root in roots)):
        raise ValueError("DATA_PROCESSOR_OUTSIDE_CODE_DIR: 代码目录必须位于 Agent 受管目录")
    os.environ.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", WANDB_MODE="disabled", WANDB_DISABLED="true",
                      HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path[:0] = [str(code / "src"), str(code)]
    import openpi.training.config as config_module
    source = Path(config_module.__file__).resolve(strict=True)
    if not source.is_relative_to(code) or not source.is_file() or source.stat().st_size > 1024 * 1024:
        raise ValueError("MODULE_OUTSIDE_CODE_DIR: 配置模块必须来自登记的代码目录")
    return {"entries": catalog_data_processors(config_module, code),
            "source": {"code_dir": str(code), "path": source.relative_to(code).as_posix(),
                       "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-json", required=True)
    args = parser.parse_args()
    try:
        result = _catalog(json.loads(args.catalog_json))
    except Exception as exc:
        message = str(exc)
        code = message.split(":", 1)[0]
        if not code.startswith(("DATA_PROCESSOR_", "MODULE_OUTSIDE_CODE_DIR")):
            code = "DATA_PROCESSOR_CATALOG_FAILED"
        print("VLA_DATA_PROCESSORS:" + json.dumps({"error": {"code": code, "message": message[:2000]}}, ensure_ascii=False), flush=True)
        raise SystemExit(2)
    print("VLA_DATA_PROCESSORS:" + json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
