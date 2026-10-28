"""Validate generated configuration evidence without executing its Python source."""

import ast
import hashlib
import re


MAX_CONFIG_SOURCE_BYTES = 384 * 1024


def validate_generated_config(value: object) -> dict:
    if not isinstance(value, dict):
        raise ValueError("missing generated Python configuration")
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("language") != "python" or value.get("filename") != "training_config.py"
            or not isinstance(value.get("source"), str)
            or not isinstance(value.get("sha256"), str)
            or re.fullmatch(r"[a-f0-9]{64}", value["sha256"]) is None):
        raise ValueError("invalid generated configuration schema")
    try:
        source = value["source"].encode("utf-8")
        if not 1 <= len(source) <= MAX_CONFIG_SOURCE_BYTES:
            raise ValueError("generated configuration exceeds source limit")
        if hashlib.sha256(source).hexdigest() != value["sha256"]:
            raise ValueError("generated configuration digest differs from its source")
        ast.parse(value["source"], filename=value["filename"])
    except (SyntaxError, UnicodeError, RecursionError) as exc:
        raise ValueError("generated configuration is not valid Python source") from exc
    return value
