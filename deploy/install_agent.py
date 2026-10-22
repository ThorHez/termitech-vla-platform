"""Install the downloaded Agent bundle in an isolated runtime and systemd service."""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import re
import secrets
import select
import shutil
import sqlite3
import stat
import subprocess
import sys
import time
from urllib.parse import quote, quote_plus, urlsplit


BUNDLE = Path(__file__).resolve().parent
SYSTEMD_UNIT_DIRECTORY = Path("/etc/systemd/system")
AGENT_PIP_INDEX_URL = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
TRAINING_PYTHON_PROBE_TIMEOUT_SECONDS = 20
EVENT_PREFIX = "VLA_INSTALL_EVENT "
INSTALL_PHASES = {"CHECKING", "ENVIRONMENT", "DEPENDENCIES", "PACKAGE", "VERIFY", "CONFIG", "SERVICE"}
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\|$)|\x1b\[[0-?]*[ -/]*(?:[@-~]|$)|\x1b[@-_]")
_SECRET_KEY = r"(?:[a-z0-9_-]*(?:password|passwd|token|secret|key|credential)[a-z0-9_-]*|authorization)"


def _plain_output(message):
    message = _ANSI.sub("", str(message))
    return "".join(c for c in message if (ord(c) >= 32 and ord(c) != 127 and not 0x80 <= ord(c) <= 0x9f)
                   and c not in "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def sanitize_install_output(message: str, protected=()) -> str:
    """Keep local private logs as safe as the centrally published event stream."""
    message = _plain_output(message)
    if (re.match(r"^\s*(?:\{.*|[}\]][,;]?\s*|\"[^\"]+\"\s*:)", message)
            or re.search(r"[\"'](?:profiles|control_plane_url|enrollment_token|state_dir|output_root)[\"']\s*:", message)
            or re.search(r"(?:^|\s)(?:sudo\s|(?:\+\s*)?(?:sh|bash)\s+-[a-z]*c\s)", message)
            or message.lstrip().startswith("+ ")):
        return "[已隐藏配置或命令输出]"
    variants = set()
    for secret in protected:
        if isinstance(secret, str) and secret:
            variants.update((secret, quote(secret, safe=""), quote_plus(secret, safe=""), json.dumps(secret)[1:-1]))
            variants.update(_plain_output(part) for part in secret.splitlines())
    variants.discard("")
    for secret in sorted(variants, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]")
    message = re.sub(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s?#<>\"'`]*@", r"\1[REDACTED]@", message)
    message = re.sub(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s?#<>\"'`]+)[?#][^\s<>\"'`]*", r"\1?[REDACTED]", message)
    message = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[^\s,;\"']+", "[REDACTED authorization]", message)
    message = re.sub(r"(?i)(\b" + _SECRET_KEY + r"\b[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^,;}]+)",
                     r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(--" + _SECRET_KEY + r"\s+)(?:\"[^\"]*\"|'[^']*'|[^\s]+)", r"\1[REDACTED]", message)
    message = message.strip()
    return message if len(message) <= 2000 else message[:1976] + " …[该行已截断]"


class InstallEvents:
    """Publish redacted, durable installer events without exposing command arguments."""

    def __init__(self):
        self.phase = "CHECKING"
        self.secrets: set[str] = set()
        self.pending: list[dict] = []
        self.stream = None
        for name, value in os.environ.items():
            if re.search(r"token|password|passwd|secret|credential|(?:^|_)key(?:_|$)|(?:^|_)pat(?:_|$)|(?:^|_)jwt(?:_|$)", name, re.I) and value:
                self.secrets.add(value)

    def protect_configuration(self, value) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if re.search(r"token|password|passwd|secret|credential|api_key", str(key), re.I):
                    if isinstance(item, str) and item:
                        self.secrets.add(item)
                else:
                    self.protect_configuration(item)
        elif isinstance(value, list):
            for item in value:
                self.protect_configuration(item)

    def redact(self, message: str) -> str:
        return sanitize_install_output(message, self.secrets)

    def emit(self, phase: str, message: str, *, level: str = "info", elapsed_seconds: float | None = None) -> None:
        if phase not in INSTALL_PHASES or level not in {"info", "warning", "error"}:
            raise ValueError("无效的安装日志事件")
        self.phase = phase
        event = {"phase": phase, "message": self.redact(message), "level": level}
        if elapsed_seconds is not None:
            event["elapsed_seconds"] = round(elapsed_seconds, 1)
        line = json.dumps(event, ensure_ascii=False)
        print(EVENT_PREFIX + line, flush=True)
        if self.stream is None:
            self.pending.append(event)
        else:
            self.stream.write(line + "\n")
            self.stream.flush()

    def open_log(self, prefix: Path) -> Path:
        directory = prefix / "logs"
        if directory.is_symlink():
            raise ValueError("安装日志目录不能是符号链接")
        directory.mkdir(mode=0o700, exist_ok=True)
        if directory.stat().st_uid != os.geteuid():
            raise ValueError("安装日志目录必须属于安装账户")
        directory.chmod(0o700)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = directory / f"install-{stamp}-{secrets.token_hex(4)}.jsonl"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(descriptor, "w", encoding="utf-8")
        for event in self.pending:
            self.stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self.pending.clear()
        self.emit("CHECKING", f"安装日志保存在 {path}（仅管理员可读，失败后保留）")
        return path

    def close(self) -> None:
        if self.stream is not None:
            self.stream.close()


class InstallCommandError(RuntimeError):
    pass


def absolute_path(value: str, *, allow_root: bool = False) -> Path:
    """Root is valid only as an explicit managed-access boundary, never a target."""
    if not isinstance(value, str) or not value.startswith("/") or any(ord(c) < 32 for c in value):
        raise ValueError("配置路径必须是无控制字符的绝对路径")
    path = Path(value)
    if ".." in path.parts or not allow_root and path.resolve() == Path("/"):
        raise ValueError("不能使用根目录或含有 .. 的路径")
    return path


def dedicated_directory(value: str) -> Path:
    path = absolute_path(value).resolve()
    if path.is_mount():
        raise ValueError("Agent 安装、状态及输出目录必须使用独立子目录，不能使用文件系统挂载点")
    return path


def overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def read_configuration(source: Path, prefix: Path) -> dict:
    config = json.loads(source.read_text())
    if not isinstance(config, dict) or not isinstance(config.get("name"), str) or not config["name"].strip():
        raise ValueError("配置缺少机器名称")
    values = config.get("roots")
    if not isinstance(values, list) or not values:
        raise ValueError("至少需要一个现有受管根目录")
    # These boundaries authorize access only. They are never passed to the
    # ownership migration, chmod, or runtime cleanup paths below.
    roots = [absolute_path(value, allow_root=True).resolve(strict=True) for value in values]
    if any(not root.is_dir() for root in roots):
        raise ValueError("受管根目录必须是现有目录")
    output = dedicated_directory(config["output_root"])
    state = dedicated_directory(config["state_dir"])
    if (state in roots or any(state in root.parents for root in roots)
            or state == prefix or state in prefix.parents
            or state in {Path(value) for value in ("/opt", "/usr", "/root", "/var", "/home", "/etc", "/tmp")}):
        raise ValueError("Agent 状态目录需要使用独立子目录，不能使用系统目录、受管根目录或安装根目录")
    if not any(output == root or root in output.parents for root in roots):
        raise ValueError("输出目录必须位于受管根目录下")
    if overlap(output, state):
        raise ValueError("状态目录与输出目录必须分开")
    if not config.get("profiles"):
        raise ValueError("至少需要配置一个训练 Python 环境")
    for profile in config["profiles"]:
        code = absolute_path(profile["code_dir"]).resolve(strict=True)
        interpreter = absolute_path(profile["python"])
        if not code.is_dir() or not interpreter.is_file() or not os.access(interpreter, os.X_OK):
            raise ValueError(f"训练代码目录或 Python 解释器不可用：{profile['id']}")
        if not any(code == root or root in code.parents for root in roots):
            raise ValueError("训练代码目录必须位于受管根目录下")
        if overlap(code, output) or overlap(code, state) or (profile.get("adapter") != "smoke" and overlap(code, prefix)):
            raise ValueError("Agent 安装目录、训练代码、任务输出与状态目录需要分开")
    url = urlsplit(config.get("control_plane_url", ""))
    if (url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password
            or url.query or url.fragment
            or (url.scheme == "http" and url.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ValueError("中台地址需使用 HTTPS，或 SSH 隧道的 loopback HTTP 地址")
    return config


def verify_bundle() -> Path:
    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    required = {"requirements.txt", "install_agent.py", "install-agent.sh", "agent-bootstrap.sh", "agent_ownership.py"}
    wheels = [name for name in manifest if name.endswith(".whl")]
    if not required.issubset(manifest) or len(wheels) != 1:
        raise ValueError("安装包文件清单不完整")
    for name, checksum in manifest.items():
        if Path(name).name != name:
            raise ValueError("安装包文件名无效")
        if hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"安装包校验失败：{name}")
    return BUNDLE / wheels[0]


def require_no_active_attempts(state_dir: str, runtime_python: Path | None = None) -> None:
    database = absolute_path(state_dir) / "agent.sqlite3"
    if not database.exists():
        return
    query = "SELECT COUNT(*) FROM attempts WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')"
    owner = database.stat().st_uid
    if os.geteuid() == 0 and owner != 0:
        # SQLite's read-only WAL reader can still create -wal/-shm files.
        # Keep these files owned by the existing Agent rather than root.
        reader = ("import pathlib,sqlite3,sys; p=pathlib.Path(sys.argv[1]).resolve(); "
                  "db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True); "
                  f"print(db.execute({query!r}).fetchone()[0]); db.close()")
        interpreter = str(runtime_python) if runtime_python is not None and runtime_python.is_file() else (
            shutil.which("python3") or sys.executable
        )
        count = int(subprocess.check_output(
            ["runuser", "-u", pwd.getpwuid(owner).pw_name, "--", interpreter,
             "-I", "-c", reader, str(database)], text=True
        ))
    else:
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as db:
            count = db.execute(query).fetchone()[0]
    if count:
        raise RuntimeError("该 Agent 仍有活动或待对账任务；请先在中台确认任务结束再升级")


def run(*args: str, events: InstallEvents, phase: str, label: str,
        heartbeat_seconds: float = 10, **kwargs):
    """Drain both pipes as one UTF-8 stream, including CR and unterminated lines."""
    started = time.monotonic()
    events.emit(phase, f"开始：{label}", elapsed_seconds=0)
    process = subprocess.Popen(list(args), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    dropping_line = False
    pipe_open = True
    last_event = started

    def consume(text: str, *, final: bool = False) -> None:
        nonlocal pending, dropping_line, last_event
        pending += text
        while True:
            boundary = re.search(r"[\r\n]", pending)
            if boundary is None:
                if final and pending:
                    line, pending = pending, ""
                elif final and dropping_line:
                    line = ""
                else:
                    if len(pending) > 16384:
                        pending = ""
                        dropping_line = True
                    break
            else:
                line, pending = pending[:boundary.start()], pending[boundary.end():]
            if dropping_line or len(line) > 16384:
                events.emit(phase, "子进程输出行过长，已省略此行", level="warning",
                            elapsed_seconds=time.monotonic() - started)
                dropping_line = False
                last_event = time.monotonic()
            elif line.strip():
                events.emit(phase, line, elapsed_seconds=time.monotonic() - started)
                last_event = time.monotonic()

    try:
        while True:
            ready, _, _ = select.select([process.stdout] if pipe_open else [], [], [], min(1, heartbeat_seconds))
            if ready:
                data = os.read(process.stdout.fileno(), 65536)
                if not data:
                    consume(decoder.decode(b"", final=True), final=True)
                    pipe_open = False
                else:
                    consume(decoder.decode(data))
            if not pipe_open and process.poll() is not None:
                break
            elapsed = time.monotonic() - started
            if time.monotonic() - last_event >= heartbeat_seconds:
                events.emit(phase, f"仍在等待：{label}（命令已运行 {elapsed:.1f} 秒，尚未返回）",
                            elapsed_seconds=elapsed)
                last_event = time.monotonic()
        returncode = process.wait()
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        process.stdout.close()
    elapsed = time.monotonic() - started
    if returncode:
        events.emit(phase, f"失败：{label}（退出码 {returncode}，耗时 {elapsed:.1f} 秒）",
                    level="error", elapsed_seconds=elapsed)
        # CalledProcessError includes argv, which may contain credentials.
        raise InstallCommandError(f"{label}失败（退出码 {returncode}）；请查看此前安装日志")
    events.emit(phase, f"完成：{label}（耗时 {elapsed:.1f} 秒）", elapsed_seconds=elapsed)
    return subprocess.CompletedProcess(args, returncode)


def check_service_location(unit: str, prefix: Path) -> bool:
    result = subprocess.run(
        ["systemctl", "show", unit, "--property=LoadState", "--value"], capture_output=True, text=True
    )
    if result.returncode or result.stdout.strip() == "not-found":
        return False
    existing = subprocess.check_output(["systemctl", "show", unit, "--property=ExecStart", "--value"], text=True)
    agent = re.escape(str(prefix / ".venv/bin/vla-agent"))
    if not re.search(r"(?:^|[ {;])path=" + agent + r"\s*(?:;|$)", existing):
        raise RuntimeError("同名 Agent 服务使用另一安装目录；请指定其现有 --prefix 或使用不同 --service-name")
    return subprocess.run(["systemctl", "is-active", "--quiet", unit]).returncode == 0


def own_directory(path: Path, account, mode: int) -> None:
    dedicated_directory(str(path))
    existed = path.exists()
    if existed and path.is_symlink():
        raise ValueError(f"受管写入目录不能是符号链接：{path}")
    if existed and path.stat().st_uid not in {0, account.pw_uid}:
        raise ValueError(f"目录已属于其他账户，不更改其权限：{path}")
    if existed and path.stat().st_uid != account.pw_uid and any(path.iterdir()):
        raise ValueError(f"目录含有其他账户的数据，请指定独立目录：{path}")
    path.mkdir(parents=True, exist_ok=True)
    os.chown(path, account.pw_uid, account.pw_gid)
    path.chmod(mode)


def service_account(name: str | None = None):
    name = name or os.environ.get("SUDO_USER") or pwd.getpwuid(os.getuid()).pw_name
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*\$?", name):
        raise ValueError("Agent 服务账户必须是有效的现有 SSH 登录用户名")
    try:
        return pwd.getpwnam(name)
    except KeyError:
        raise ValueError(f"Agent 服务账户不存在：{name}；请使用 SSH 登录账户") from None


def previous_service_account(unit: str):
    result = subprocess.run(
        ["systemctl", "show", unit, "--property=LoadState", "--value"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode or result.stdout.strip() != "loaded":
        raise RuntimeError("现有 Agent 配置缺少已加载的匹配 systemd 服务，不能迁移文件属主；请先检查现有安装")
    name = subprocess.check_output(["systemctl", "show", unit, "--property=User", "--value"], text=True).strip()
    # An empty system service User means root. Existing service location and
    # config are checked separately before its account is trusted for migration.
    return service_account(name or "root")


def ownership_migration(account, previous_account):
    specification = importlib.util.spec_from_file_location("vla_installer_ownership", BUNDLE / "agent_ownership.py")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module.OwnershipMigration(account, previous_account)


class FileSnapshot:
    """Restore service/config contents and access metadata if account migration fails."""

    def __init__(self, path: Path):
        self.path = path
        if path.is_symlink():
            raise ValueError(f"Agent 配置或服务文件不能是符号链接：{path}")
        self.original = path.stat() if path.exists() else None
        self.contents = path.read_bytes() if self.original is not None else None

    def restore(self) -> None:
        if self.original is None:
            self.path.unlink(missing_ok=True)
            return
        temporary = self.path.with_name(self.path.name + ".rollback-" + secrets.token_hex(6))
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self.contents)
                os.fchown(stream.fileno(), self.original.st_uid, self.original.st_gid)
                os.fchmod(stream.fileno(), stat.S_IMODE(self.original.st_mode))
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def runtime_environment() -> dict[str, str]:
    """Do not let the login/training Python environment affect Agent commands."""
    excluded = {"PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "PYTHONSTARTUP", "PYTHONINSPECT",
                "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"}
    return {key: value for key, value in os.environ.items() if key not in excluded}


def pip_environment(prefix: Path, runtime_kind: str = "venv") -> dict[str, str]:
    # This isolated installer must not inherit extra indexes or trusted-host
    # exemptions from a server's unrelated training/software environment.
    env = {key: value for key, value in runtime_environment().items() if not key.startswith("PIP_")}
    env.update({"TMPDIR": str(prefix / "tmp"), "PIP_CONFIG_FILE": os.devnull,
                "PIP_INDEX_URL": AGENT_PIP_INDEX_URL, "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_REQUIRE_VIRTUALENV": "false" if runtime_kind == "conda" else "true", "PYTHONNOUSERSITE": "1",
                "PIP_NO_INPUT": "1", "PIP_PROGRESS_BAR": "off", "PIP_NO_COLOR": "1",
                "NO_COLOR": "1", "PYTHONUNBUFFERED": "1"})
    return env


def runtime_request(args, prefix: Path) -> tuple[str, str | None]:
    """Validate the runtime choice before stopping any currently running Agent."""
    kind = getattr(args, "runtime_kind", "venv")
    if kind not in {"venv", "conda"}:
        raise ValueError("Agent Python 环境类型必须是 conda 或 venv")
    minimum = (3, 9) if kind == "conda" else (3, 11)
    if sys.version_info < minimum:
        raise RuntimeError("Conda 安装引导需要 Python 3.9+" if kind == "conda"
                           else "无 Conda 时需要 Python 3.11+ 创建 Agent 独立环境")
    conda = getattr(args, "conda_executable", None)
    if kind == "conda":
        if not conda:
            raise ValueError("Conda 安装需要指定已发现的 Conda 可执行文件")
        executable = absolute_path(conda)
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise ValueError("Conda 可执行文件不可用；请修复 Conda 后重试")
    runtime = prefix / ".venv"
    if runtime.is_symlink():
        raise ValueError("Agent Python 环境不能是指向其他环境的符号链接")
    if runtime.exists():
        conda_marker, venv_marker = runtime / "conda-meta", runtime / "pyvenv.cfg"
        if (not runtime.is_dir() or conda_marker.is_symlink() or venv_marker.is_symlink()
                or conda_marker.is_dir() == venv_marker.is_file()):
            raise ValueError("Agent Python 环境目录不完整或类型不明；请保留原目录并检查后重试")
    return kind, conda


class AgentRuntime:
    def __init__(self, path: Path, kind: str, backup: Path | None = None):
        self.path = path
        self.kind = kind
        self.backup = backup

    @property
    def python(self) -> Path:
        return self.path / "bin/python"

    def rollback(self, events: InstallEvents) -> None:
        if self.backup is not None:
            if self.path.is_symlink():
                self.path.unlink()
            elif self.path.exists():
                shutil.rmtree(self.path)
            self.backup.replace(self.path)
            self.backup = None
            events.emit("ENVIRONMENT", "已恢复升级前的 Agent Python 环境", level="warning")


def prepare_runtime(prefix: Path, kind: str, conda: str | None, events: InstallEvents) -> AgentRuntime:
    """Create only an Agent-owned runtime; never install into Conda base or training."""
    path = prefix / ".venv"
    if path.exists() and (path / "conda-meta").is_dir():
        events.emit("ENVIRONMENT", f"复用 Agent 专用 Conda 环境：{path}")
        return AgentRuntime(path, "conda")
    if path.exists() and kind == "venv":
        events.emit("ENVIRONMENT", f"复用 Agent 独立 Python venv：{path}")
        return AgentRuntime(path, "venv")
    runtime = AgentRuntime(path, kind)
    if path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        runtime.backup = prefix / (".venv.before-conda-" + stamp + "-" + secrets.token_hex(4))
        path.replace(runtime.backup)
        events.emit("ENVIRONMENT", "已备份原 Agent venv，开始迁移到专用 Conda 环境")
    try:
        if kind == "conda":
            env = runtime_environment()
            env["CONDA_PKGS_DIRS"] = str(prefix / "conda-pkgs")
            run(str(conda), "create", "--prefix", str(path), "python=3.11", "pip", "--yes",
                env=env, events=events, phase="ENVIRONMENT", label="创建 Agent 专用 Conda 环境（Python 3.11）")
        else:
            run(sys.executable, "-I", "-m", "venv", str(path), env=runtime_environment(),
                events=events, phase="ENVIRONMENT", label="创建 Agent 独立 Python venv")
    except BaseException:
        if runtime.backup is not None:
            runtime.rollback(events)
        elif path.is_symlink():
            path.unlink()
        elif path.exists():
            # Only this invocation created the path. Remove an incomplete Conda
            # transaction so a retry creates it rather than reusing broken files.
            shutil.rmtree(path)
        raise
    return runtime


def verify_runtime(runtime: AgentRuntime, prefix: Path, account, events: InstallEvents) -> None:
    # Running as the service account proves that a root-owned Conda installation
    # did not leave the Agent dependent on /root or another inaccessible base.
    script = (
        "import pathlib,sys; expected=pathlib.Path(sys.argv[1]).resolve(); "
        "assert sys.version_info >= (3,11), 'Agent Python requires 3.11+'; "
        "assert pathlib.Path(sys.prefix).resolve() == expected, 'Agent Python prefix mismatch'; "
        "import pip; assert expected in pathlib.Path(pip.__file__).resolve().parents, 'pip outside Agent environment'; "
        "print('Agent Python: '+sys.executable+' ('+sys.version.split()[0]+')')"
    )
    run("runuser", "-u", account.pw_name, "--", str(runtime.python), "-I", "-c", script, str(runtime.path),
        env={**runtime_environment(), "HOME": str(prefix / "home")},
        events=events, phase="ENVIRONMENT", label="验证 Agent Python、pip 版本及专用账户访问权限")


def verify_training_interpreters(config: dict, account, events: InstallEvents) -> None:
    """Probe training interpreters as the actual service user without importing training code."""
    script = (
        "import json,os,sys; os.chdir(sys.argv[1]); "
        "print(json.dumps({'version': list(sys.version_info[:3]), 'executable': sys.executable}))"
    )
    env = {**runtime_environment(), "HOME": account.pw_dir}
    for profile in config["profiles"]:
        identifier, python, code = profile["id"], profile["python"], profile["code_dir"]
        context = f"训练环境 {identifier}：Python {python}，代码目录 {code}，Agent 服务账户 {account.pw_name}"
        remedy = ("请指定该账户可进入代码目录、可执行的训练 Python；若解释器是符号链接，"
                  "还需检查链接目标及其上级目录权限（例如 /root 下的 uv Python）。")
        events.emit("CHECKING", f"检查训练 Python 的服务账户访问权限：{context}")
        try:
            result = subprocess.run(
                ["runuser", "-u", account.pw_name, "--", python, "-I", "-S", "-B", "-c", script, code],
                cwd=code, env=env, capture_output=True, text=True,
                timeout=TRAINING_PYTHON_PROBE_TIMEOUT_SECONDS, check=False,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"训练 Python 检查超时（{TRAINING_PYTHON_PROBE_TIMEOUT_SECONDS} 秒）：{context}。{remedy}"
            ) from None
        except OSError:
            raise RuntimeError(f"训练 Python 或代码目录无法访问：{context}。{remedy}") from None
        if result.returncode:
            raise RuntimeError(f"训练 Python 无法由服务账户执行（退出码 {result.returncode}）：{context}。{remedy}")
        try:
            metadata = json.loads(result.stdout)
            version = metadata["version"]
            if (not isinstance(version, list) or len(version) != 3
                    or any(type(item) is not int or item < 0 for item in version)
                    or not isinstance(metadata["executable"], str) or not metadata["executable"]):
                raise ValueError("invalid interpreter metadata")
        except (ValueError, KeyError, TypeError):
            raise RuntimeError(f"训练 Python 未返回有效的版本信息：{context}。{remedy}") from None
        events.emit("CHECKING", f"训练环境 {identifier} 的 Python {'.'.join(map(str, version))} 可由 {account.pw_name} 执行")


def _install(args, events: InstallEvents) -> None:
    events.emit("CHECKING", "检查目标系统、配置及安装包完整性")
    if sys.platform != "linux" or os.geteuid() != 0:
        raise RuntimeError("请在目标 Linux 服务器使用 sudo bash ./install-agent.sh 执行")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.service_name):
        raise ValueError("服务名仅允许字母、数字、下划线和连字符")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", args.prefix):
        raise ValueError("Agent 安装目录需使用简单的 Linux 绝对路径")
    prefix = dedicated_directory(args.prefix)
    if prefix in {Path("/opt"), Path("/data"), Path("/data2"), Path("/usr"), Path("/root"), Path("/var")}:
        raise ValueError("请使用专用 Agent 子目录作为安装目录")
    account = service_account(getattr(args, "service_user", None))
    kind, conda = runtime_request(args, prefix)
    source = Path(args.config).resolve(strict=True)
    events.protect_configuration(json.loads(source.read_text()))
    config = read_configuration(source, prefix)
    events.emit("CHECKING", "配置检查通过；训练 Python 环境保持独立")
    wheel = verify_bundle()
    events.emit("CHECKING", "安装包 manifest 与所有文件 SHA256 校验通过")
    for command in ("systemctl", "systemd-analyze", "git", "runuser"):
        if not shutil.which(command):
            raise RuntimeError(f"目标系统缺少命令：{command}")
    unit = args.service_name + ".service"
    was_active = check_service_location(unit, prefix)
    old_config = prefix / "config/agent.json"
    previous_account = None
    previous_config = None
    if old_config.is_file():
        old = json.loads(old_config.read_text())
        previous_config = old
        if old.get("state_dir") != config["state_dir"]:
            raise RuntimeError("升级不能更换 Agent state_dir；请保留已有节点凭据和执行状态")
        require_no_active_attempts(old["state_dir"], prefix / ".venv/bin/python")
        previous_account = previous_service_account(unit)
    require_no_active_attempts(config["state_dir"], prefix / ".venv/bin/python")
    unit_path = SYSTEMD_UNIT_DIRECTORY / unit
    config_snapshot, unit_snapshot = FileSnapshot(old_config), FileSnapshot(unit_path)
    prefix.mkdir(parents=True, exist_ok=True)
    prefix.chmod(0o755)
    events.open_log(prefix)
    events.emit("CHECKING", "现有服务目录及活动训练任务检查通过")
    events.emit("CHECKING", f"Agent 使用 SSH 登录账户运行：{account.pw_name}（UID {account.pw_uid}）")
    verify_training_interpreters(config, account, events)
    if was_active:
        run("systemctl", "stop", unit, events=events, phase="ENVIRONMENT", label="暂停现有 Agent 服务")
    runtime = None
    ownership = None
    service_start_attempted = False
    files_changed = False
    try:
        require_no_active_attempts(config["state_dir"], prefix / ".venv/bin/python")
        ownership = ownership_migration(account, previous_account) if previous_account is not None else None
        for path, mode, recursive in [(prefix / "home", 0o700, True), (prefix / "tmp", 0o700, True),
                                      (absolute_path(config["state_dir"]), 0o700, True),
                                      (absolute_path(config["output_root"]), 0o755, False)]:
            dedicated_directory(str(path))
            if ownership is not None and (recursive or previous_config.get("output_root") == config["output_root"]):
                ownership.migrate(path, recursive=recursive)
            own_directory(path, account, mode)
        runtime = prepare_runtime(prefix, kind, conda, events)
        verify_runtime(runtime, prefix, account, events)
        env = pip_environment(prefix, runtime.kind)
        events.emit("DEPENDENCIES", f"Agent 依赖使用清华 PyPI 镜像：{AGENT_PIP_INDEX_URL}；按锁定版本及 SHA-256 校验")
        run(str(runtime.python), "-I", "-m", "pip", "install", "--require-hashes", "--only-binary=:all:",
            "--no-cache-dir", "-r", str(BUNDLE / "requirements.txt"), env=env,
            events=events, phase="DEPENDENCIES", label="下载并安装 Agent Python 依赖")
        run(str(runtime.python), "-I", "-m", "pip", "install", "--no-deps", "--no-cache-dir",
            "--force-reinstall", str(wheel), env=env,
            events=events, phase="PACKAGE", label="安装 Agent 平台程序")
        run(str(runtime.python), "-I", "-m", "pip", "check", env=env,
            events=events, phase="VERIFY", label="检查 Agent 依赖完整性")
        events.emit("CONFIG", "写入受限访问的 Agent 配置，保留原配置备份")
        config_dir = prefix / "config"
        config_dir.mkdir(exist_ok=True)
        config_dir.chmod(0o755)
        if old_config.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            backup = config_dir / ("agent.before-" + stamp + ".json")
            backup.write_bytes(old_config.read_bytes())
            os.chown(backup, 0, account.pw_gid)
            backup.chmod(0o640)
        staged = config_dir / "agent.installing.json"
        descriptor = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2)
        os.chown(staged, 0, account.pw_gid)
        staged.chmod(0o640)
        files_changed = True
        os.replace(staged, old_config)
        events.emit("CONFIG", f"Agent 配置已写入 {old_config}（不输出配置内容）")
        git_env = {**env, "HOME": str(prefix / "home")}
        for profile in config["profiles"]:
            run("runuser", "-u", account.pw_name, "--", "env", "HOME=" + str(prefix / "home"), "git", "config", "--global", "--add",
                "safe.directory", profile["code_dir"], env=git_env,
                events=events, phase="CONFIG", label="配置 Agent 代码目录访问")
        unit_path.write_text(f"""[Unit]
Description=Termitech VLA bare-metal Agent
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User={account.pw_name}
Group={account.pw_gid}
WorkingDirectory={prefix}
ExecStart={prefix}/.venv/bin/vla-agent --config {prefix}/config/agent.json
Environment=HOME={prefix}/home
Environment=TMPDIR={prefix}/tmp
Environment=PYTHONUNBUFFERED=1
Restart=on-failure
RestartSec=5
UMask=0077
KillMode=process
TimeoutStopSec=20
NoNewPrivileges=true

[Install]
WantedBy=multi-user.target
""")
        unit_path.chmod(0o644)
        run("systemd-analyze", "verify", str(unit_path), events=events, phase="SERVICE", label="校验 systemd 服务配置")
        run("systemctl", "daemon-reload", events=events, phase="SERVICE", label="重新加载 systemd 服务配置")
        service_start_attempted = True
        run("systemctl", "enable", "--now", unit, events=events, phase="SERVICE", label="启用并启动 Agent 服务")
        run("systemctl", "is-active", "--quiet", unit, events=events, phase="SERVICE", label="确认 Agent 服务正在运行")
        if runtime.backup is not None:
            events.emit("ENVIRONMENT", f"原 Agent venv 备份保留在 {runtime.backup}")
        events.emit("SERVICE", f"Agent 已安装并启动：{unit}；等待中台确认节点上线和训练环境探测结果")
    except BaseException:
        restored = True
        try:
            if service_start_attempted:
                run("systemctl", "stop", unit, events=events, phase="SERVICE", label="停止未完成升级的 Agent 服务")
            if runtime is not None:
                runtime.rollback(events)
            if files_changed:
                config_snapshot.restore()
                unit_snapshot.restore()
            if ownership is not None:
                ownership.rollback()
            if files_changed:
                run("systemctl", "daemon-reload", events=events, phase="SERVICE", label="恢复升级前的服务配置")
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            restored = False
            events.emit("SERVICE", "恢复原 Agent 环境、配置或文件属主失败；为避免账户不一致，请检查安装日志后手动恢复服务", level="error")
        if was_active and restored:
            try:
                run("systemctl", "start", unit, events=events, phase="SERVICE", label="恢复原 Agent 服务")
            except (OSError, InstallCommandError):
                events.emit("SERVICE", "恢复原 Agent 服务失败；请检查 systemd 运行日志", level="error")
        raise


def install(args) -> None:
    events = InstallEvents()
    try:
        _install(args, events)
    except BaseException as exc:
        events.emit(events.phase, f"Agent 安装失败：{exc}", level="error")
        raise
    finally:
        events.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="网页下载的 Agent JSON 配置")
    parser.add_argument("--prefix", default="/data2/termitech-vla-agent", help="Agent 独立安装目录")
    parser.add_argument("--service-name", default="termitech-vla-agent", help="systemd 服务名，不含 .service")
    parser.add_argument("--runtime-kind", choices=("conda", "venv"), default="venv", help="安装引导发现的 Agent 环境类型")
    parser.add_argument("--conda-executable", help="安装引导发现的 Conda 绝对路径")
    parser.add_argument("--service-user", help="Agent 运行账户；默认使用 sudo 前的用户或当前用户，应与 SSH 登录账户一致")
    args = parser.parse_args()
    previous_umask = os.umask(0o022)
    try:
        install(args)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError, sqlite3.Error):
        return 1
    finally:
        os.umask(previous_umask)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
