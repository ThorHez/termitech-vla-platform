"""Generate enrollment configuration without mutating or probing remote nodes.

Paths below are POSIX strings for a different machine. Never resolve them against
the control plane filesystem or claim that the target interpreter exists here.
"""

import ipaddress
from pathlib import PurePosixPath
import posixpath
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

from .common import APIError, uid
from .schemas import AgentConfigInput


def _has_controls(value: str) -> bool:
    return any(unicodedata.category(char).startswith("C") for char in value)


def _path(value: str, field: str) -> str:
    if (not value.startswith("/") or ".." in PurePosixPath(value).parts
            or _has_controls(value)):
        raise APIError(422, "INVALID_AGENT_PATH", f"{field} must be an absolute POSIX path without '..' or control characters")
    # Collapse redundant separators and '.' without inspecting this host's disk.
    normalized = "/" + posixpath.normpath(value).lstrip("/")
    if normalized == "/" and field != "roots":
        raise APIError(422, "INVALID_AGENT_PATH", f"{field} must use a dedicated path; '/' is allowed only as a managed browsing root")
    return normalized


def _under(path: str, root: str) -> bool:
    candidate, parent = PurePosixPath(path), PurePosixPath(root)
    return candidate == parent or parent in candidate.parents


def _overlap(left: str, right: str) -> bool:
    return _under(left, right) or _under(right, left)


def _url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise APIError(422, "INVALID_CONTROL_PLANE_URL", "Control plane URL is invalid") from exc
    if (value != value.strip() or _has_controls(value) or any(char.isspace() for char in value)
            or parsed.scheme not in {"http", "https"} or not hostname
            or parsed.username is not None or parsed.password is not None
            or "?" in value or "#" in value or "\\" in value
            or (port is not None and port < 1)):
        raise APIError(422, "INVALID_CONTROL_PLANE_URL", "Use an HTTP(S) control plane URL without credentials, query or fragment")
    if parsed.scheme == "http" and hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise APIError(422, "INSECURE_CONTROL_PLANE_URL", "Remote control planes require HTTPS; HTTP is allowed only for a local endpoint or SSH tunnel")
    if any(segment == ".." for segment in parsed.path.split("/")):
        raise APIError(422, "INVALID_CONTROL_PLANE_URL", "Control plane URL cannot contain path traversal")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _local_hostname(hostname: str) -> bool:
    hostname = hostname.lower().rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def resolve_agent_config_defaults(configured_url: str | None, request_url: str) -> dict:
    """Choose an enrollment endpoint without probing or trusting proxy headers.

    The caller supplies the ASGI request base URL. A local development origin
    cannot describe how another machine reaches this service, so it must never
    silently become that machine's loopback endpoint.
    """
    if configured_url:
        try:
            control_url = _url(configured_url)
        except APIError as exc:
            raise APIError(503, "AGENT_DEFAULT_URL_INVALID",
                           "平台预配置的 Agent 连接入口无效；请管理员检查 VLA_AGENT_CONTROL_PLANE_URL") from exc
        hostname = urlsplit(control_url).hostname or ""
        message = ("使用平台统一配置的接入地址；目标机器需已有本机中台或 SSH 隧道"
                   if _local_hostname(hostname) else "使用平台统一配置的接入地址")
        return {"control_plane_url": control_url, "source": "configured", "message": message}

    try:
        control_url = _url(request_url)
        parsed = urlsplit(control_url)
        hostname = parsed.hostname or ""
        if parsed.scheme != "https" or _local_hostname(hostname):
            raise ValueError("A public HTTPS origin is required")
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            address = None
        if address is not None and not address.is_global:
            raise ValueError("A public HTTPS origin is required")
    except (APIError, ValueError):
        return {"control_plane_url": None, "source": "unavailable",
                "message": "平台尚未配置可供 Agent 访问的地址，可在高级连接设置中指定"}
    return {"control_plane_url": control_url, "source": "request",
            "message": "已使用当前平台的 HTTPS 入口"}


def generate_agent_config(body: AgentConfigInput, enrollment_token: str, default_control_plane_url: str | None = None) -> dict:
    """Return a config document. Actual path/runtime checks occur on the Agent."""
    if not body.name.strip() or _has_controls(body.name):
        raise APIError(422, "INVALID_AGENT_NAME", "Machine name must contain visible text without control characters")
    control_url_value = body.control_plane_url if body.control_plane_url is not None else default_control_plane_url
    if control_url_value is None:
        raise APIError(422, "AGENT_CONTROL_PLANE_URL_REQUIRED",
                       "无法自动确定 Agent 连接入口；请管理员配置统一入口，或在高级设置中填写目标机器可访问的中台地址")
    control_url = _url(control_url_value)
    roots = list(dict.fromkeys(_path(root, "roots") for root in body.roots))
    output = _path(body.output_root, "output_root")
    state = _path(body.state_dir, "state_dir")
    if (state in {"/opt", "/usr", "/root", "/var", "/home", "/etc", "/tmp"}
            or any(_under(root, state) for root in roots)):
        raise APIError(422, "INVALID_AGENT_PATH", "Agent state must use a dedicated subdirectory, not a system directory or managed root")
    if not any(_under(output, root) for root in roots):
        raise APIError(422, "PATH_OUTSIDE_ROOTS", "output_root must be inside a managed root")
    if _overlap(output, state):
        raise APIError(422, "AGENT_PATH_OVERLAP", "Agent state and training outputs must use separate directories")
    if len({profile.id for profile in body.profiles}) != len(body.profiles):
        raise APIError(422, "DUPLICATE_PROFILE", "Profile IDs must be unique on each node")

    onboarding_id = uid()
    profiles = []
    for profile in body.profiles:
        code = _path(profile.code_dir, "profiles.code_dir")
        python = _path(profile.python, "profiles.python")
        if not any(_under(code, root) for root in roots):
            raise APIError(422, "PATH_OUTSIDE_ROOTS", "Training code must be inside a managed root")
        if _overlap(code, output) or _overlap(code, state):
            raise APIError(422, "AGENT_PATH_OVERLAP", "Training code, outputs and Agent state must use separate directories")
        if not profile.name.strip() or _has_controls(profile.name):
            raise APIError(422, "INVALID_PROFILE_NAME", "Runtime name must contain visible text without control characters")
        if profile.recipe is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,255}", profile.recipe):
            raise APIError(422, "INVALID_RECIPE", "Recipe must be a named training configuration containing letters, numbers, underscores or hyphens")
        metadata = {"validated": False, "onboarding_id": onboarding_id}
        if profile.adapter == "smoke":
            metadata["simulation"] = True
        profiles.append({**profile.model_dump(), "name": profile.name.strip(), "code_dir": code,
                         "python": python, "metadata": metadata})

    return {"id": onboarding_id, "filename": f"vla-agent-{onboarding_id}.json", "config": {
        "control_plane_url": control_url, "enrollment_token": enrollment_token,
        "name": body.name.strip(), "roots": roots, "output_root": output, "state_dir": state,
        "poll_interval_seconds": 2, "heartbeat_interval_seconds": 10, "profiles": profiles,
    }}
