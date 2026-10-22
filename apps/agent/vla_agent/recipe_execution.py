"""Check a saved recipe's source pin before producing a launch snapshot."""

import hashlib
import os
import stat

from vla_adapters.recipe_definitions import validate_definition

from .safety import AgentError, allowed_path


def verify_recipe_source(profile: dict, roots: list[str], definition: dict, snapshot: dict) -> None:
    definition = validate_definition(definition)
    source = definition["source"]
    checkout = allowed_path(profile["code_dir"], roots)
    pinned_checkout = allowed_path(source["code_dir"], roots)
    path = allowed_path(str(pinned_checkout / source["path"]), roots)
    if (checkout != pinned_checkout or checkout not in path.parents or definition["profile_id"] != profile["id"]):
        raise AgentError("RECIPE_SOURCE_CHANGED", "保存的配方不属于当前代码目录或训练环境，请重新导入")
    if snapshot.get("commit_sha") != source["commit_sha"] or snapshot.get("changes_digest") != source["changes_digest"]:
        raise AgentError("RECIPE_SOURCE_CHANGED", "基础代码版本已变化，请查看差异并另存配方新版本")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 1024 * 1024:
                raise ValueError("Not a bounded regular source file")
            content = stream.read(1024 * 1024 + 1)
        if len(content) > 1024 * 1024 or hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("Source hash changed")
    except (OSError, ValueError) as exc:
        raise AgentError("RECIPE_SOURCE_CHANGED", "基础配方源码已变化，请重新导入配方") from exc
