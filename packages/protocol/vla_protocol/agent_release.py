"""Read immutable Agent release identities without extracting or executing a bundle."""

import hashlib
from importlib import metadata, resources
import io
import json
from pathlib import Path
import re
import tarfile
import zipfile


RELEASE_MEMBER = "vla_agent/agent-release.json"
MAX_WHEEL_BYTES = 64 * 1024 * 1024


def _release(value):
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or not isinstance(value.get("version"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_-]{0,99}", value["version"])
            or not isinstance(value.get("build_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", value["build_id"])):
        raise ValueError("Agent 安装包缺少有效的构建身份，请重新构建安装包")
    return {"version": value["version"], "build_id": value["build_id"]}


def installed_agent_release():
    """Capture once when an Agent starts, so an old process cannot claim new files."""
    try:
        return _release(json.loads(resources.files("vla_agent").joinpath("agent-release.json").read_text()))
    except (OSError, ValueError, TypeError):
        try:
            version = metadata.version("termitech-vla-platform")
        except metadata.PackageNotFoundError:
            version = None
        # Editable checkouts and legacy wheels have no trustworthy build ID.
        return {"version": version, "build_id": None}


def read_agent_bundle(path: str | Path):
    """Validate a small, flat bundle and the identity embedded in its Agent wheel."""
    files = {}
    try:
        with tarfile.open(path, "r:gz") as archive:
            for index, member in enumerate(archive):
                if (index >= 32 or not member.isfile() or not member.name
                        or Path(member.name).name != member.name or member.name in files
                        or member.size > (MAX_WHEEL_BYTES if member.name.endswith(".whl") else 1024 * 1024)):
                    raise ValueError("Agent 安装包文件清单无效")
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Agent 安装包文件不可读")
                files[member.name] = stream.read(member.size + 1)
                if len(files[member.name]) != member.size:
                    raise ValueError("Agent 安装包文件不完整")
        manifest = json.loads(files.get("manifest.json", b"null"))
        if (not isinstance(manifest, dict) or set(manifest) != set(files) - {"manifest.json"}
                or not {"release.json", "requirements.txt", "install_agent.py", "install-agent.sh",
                        "agent-bootstrap.sh", "agent_ownership.py"}.issubset(manifest)):
            raise ValueError("Agent 安装包缺少构建信息，请重新构建安装包")
        if any(not isinstance(checksum, str) or hashlib.sha256(files[name]).hexdigest() != checksum
               for name, checksum in manifest.items()):
            raise ValueError("Agent 安装包 SHA-256 校验失败")
        release = _release(json.loads(files["release.json"]))
        wheels = [name for name in files if name.endswith(".whl")]
        if len(wheels) != 1:
            raise ValueError("Agent 安装包需要且只能包含一个 wheel")
        with zipfile.ZipFile(io.BytesIO(files[wheels[0]])) as wheel:
            entries = wheel.infolist()
            candidates = [item for item in entries if item.filename == RELEASE_MEMBER]
            if len(entries) > 5000 or len(candidates) != 1 or candidates[0].file_size > 4096:
                raise ValueError("Agent wheel 缺少有效的构建信息，请重新构建安装包")
            embedded = _release(json.loads(wheel.read(candidates[0])))
            if embedded != release:
                raise ValueError("Agent 安装包与 wheel 的构建身份不一致")
        return release
    except (tarfile.TarError, zipfile.BadZipFile, KeyError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Agent 安装包构建信息无效，请重新构建安装包") from exc
