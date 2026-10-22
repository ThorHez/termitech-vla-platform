"""Task entrypoints are relative Python files inside a registered project."""

import unicodedata
import os
import stat


def validate_entrypoint(value: str | None) -> str | None:
    if value is None:
        return None
    if (not isinstance(value, str) or not value or len(value) > 4096 or value.startswith("/")
            or "\\" in value or not value.endswith(".py")
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value)):
        raise ValueError("INVALID_ENTRYPOINT: 训练入口必须是项目内的相对 .py 文件路径，不能包含上级目录或控制字符")
    return value


def read_entrypoint(path) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 2 * 1024 * 1024:
            raise ValueError("ENTRYPOINT_INVALID: 训练入口必须是最多 2 MiB 的普通 Python 文件")
        contents = stream.read(2 * 1024 * 1024 + 1)
    if len(contents) > 2 * 1024 * 1024:
        raise ValueError("ENTRYPOINT_INVALID: 训练入口超过 2 MiB")
    return contents
