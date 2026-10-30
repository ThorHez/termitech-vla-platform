"""One read-only output-path contract for preflight and the detached runner."""
from __future__ import annotations

import os
from pathlib import Path
import stat

from .safety import AgentError, allowed_path
from .snapshots import git


def _overlap(left: Path, right: Path) -> bool:
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _directory_prefix(path: Path) -> None:
    """Inspect the nearest existing prefix without creating an attempt folder."""
    prefix = path
    while True:
        try:
            details = prefix.stat()
        except FileNotFoundError:
            if prefix.parent == prefix:
                raise AgentError("OUTPUT_UNAVAILABLE", f"无法检查训练输出路径：{path}")
            prefix = prefix.parent
            continue
        except (OSError, RuntimeError) as exc:
            raise AgentError("OUTPUT_UNAVAILABLE", f"无法检查训练输出路径的目录或访问权限：{path}") from exc
        if not stat.S_ISDIR(details.st_mode):
            raise AgentError("OUTPUT_UNAVAILABLE", f"训练输出路径经过了普通文件，必须使用目录：{prefix}")
        if not os.access(prefix, os.X_OK):
            raise AgentError("OUTPUT_UNAVAILABLE", f"Agent 无权访问训练输出路径的父目录：{prefix}")
        return


def validate_output_isolation(config: dict, code_dir: str, inputs: dict, output_dir: str) -> dict:
    """Reject canonical overlap with source or inputs, without any writes.

    The actual probe/attempt directory is checked. A shared output root may be
    an ancestor of unrelated projects or assets: sibling attempt paths remain
    valid. The runner repeats this exact check for its real attempt directory.
    """
    roots = config["roots"]
    output_root = allowed_path(config["output_root"], roots)
    if not output_root.is_dir():
        raise AgentError("OUTPUT_UNAVAILABLE", f"训练输出根目录不是可访问的目录：{output_root}")
    output = allowed_path(output_dir, [str(output_root)], must_exist=False)
    _directory_prefix(output)
    project = allowed_path(code_dir, roots)
    if not project.is_dir():
        raise AgentError("CODE_VERSION_UNAVAILABLE", f"训练代码目录不是可访问的目录：{project}")
    repository = allowed_path(git(project, "rev-parse", "--show-toplevel").decode().strip(), roots)
    if not project.is_relative_to(repository):
        raise AgentError("CODE_VERSION_UNAVAILABLE", "无法确认训练代码目录所属的 Git 仓库")
    for label, source in (("训练项目", project), ("训练 Git 仓库", repository)):
        if _overlap(output, source):
            raise AgentError("OUTPUT_PATH_CONFLICT",
                f"训练输出目录 {output} 与{label} {source} 重叠；请在机器存储配置中选择代码目录之外的输出根目录（当前为 {output_root}）")
    if not isinstance(inputs, dict):
        raise AgentError("INVALID_PATH", "训练输入资产配置无效，无法核对输出目录隔离")
    for role, asset in inputs.items():
        if not isinstance(asset, dict) or not isinstance(asset.get("path"), str):
            raise AgentError("INVALID_PATH", f"训练输入资产 {role} 缺少可检查的绝对路径")
        source = allowed_path(asset["path"], roots)
        if _overlap(output, source):
            raise AgentError("OUTPUT_PATH_CONFLICT",
                f"训练输出目录 {output} 与输入资产 {role} 的路径 {source} 重叠；请使用与输入资产分开的输出目录")
    return {"output_root": str(output_root), "code_dir": str(project), "repository_root": str(repository)}
