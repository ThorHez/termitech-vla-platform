"""Password SSH deployment and application-owned reverse tunnels.

Remote command output is private diagnostic data unless explicitly streamed by
the installer. Installation lines are bounded and sanitized before publication.
"""

from __future__ import annotations

import base64
import codecs
import errno
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import select
import shlex
import socket
import stat
import threading
import time
from typing import Callable
from urllib.parse import quote, quote_plus

import paramiko


class RemoteError(Exception):
    def __init__(self, code: str, message: str):
        self.code, self.message = code, message
        super().__init__(message)


_INSTALL_PHASES = frozenset({"CHECKING", "ENVIRONMENT", "DEPENDENCIES", "PACKAGE", "VERIFY", "CONFIG", "SERVICE", "STARTING"})
_OUTPUT_LIMIT = 256 * 1024
_STREAM_LIMIT = 2 * 1024 * 1024
_LINE_LIMIT = 16 * 1024
_ANSI = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\|$)|\x1b\[[0-?]*[ -/]*(?:[@-~]|$)|\x1b[@-_]")
_SECRET_KEY = r"(?:[a-z0-9_-]*(?:password|passwd|token|secret|api[_-]?key|credential)[a-z0-9_-]*|(?:private|access|client)[_-]?key|key|(?:proxy[_-]?)?authorization)"


def _plain_output(text):
    text = _ANSI.sub("", str(text))
    return "".join(c for c in text if (ord(c) >= 32 and ord(c) != 127 and not 0x80 <= ord(c) <= 0x9f)
                   and c not in "\u200b\u200c\u200d\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def sanitize_install_output(text: str, secrets=()) -> str:
    """Return a short displayable line, never a remote command or config dump.

    Call only after a complete line has been assembled. Overlong source lines
    are discarded by the stream framer, so truncation cannot publish a secret
    whose remaining characters arrive in the next SSH chunk.
    """
    text = _plain_output(text)
    # Config JSON and shell tracing add no useful installation diagnostics.
    if (re.match(r"^\s*(?:\{.*|[}\]][,;]?\s*|\"[^\"]+\"\s*:)", text)
            or re.search(r"[\"'](?:profiles|control_plane_url|enrollment_token|state_dir|output_root)[\"']\s*:", text)
            or re.search(r"(?:^|\s)(?:sudo\s|(?:\+\s*)?(?:sh|bash)\s+-[a-z]*c\s)", text)
            or text.lstrip().startswith("+ ")):
        return "[已隐藏配置或命令输出]"
    variants = set()
    for secret in secrets:
        if isinstance(secret, str) and secret:
            variants.update((secret, quote(secret, safe=""), quote_plus(secret, safe=""), json.dumps(secret)[1:-1]))
            variants.update(_plain_output(part) for part in secret.splitlines())
    variants.discard("")
    text = re.sub(r"(?i)\b([a-z][a-z0-9+.-]*://)[^/\s?#<>\"'`]*@", r"\1[REDACTED]@", text)
    text = re.sub(r"(?i)(\b[a-z][a-z0-9+.-]*://[^\s?#<>\"'`]+)[?#][^\s<>\"'`]*", r"\1?[REDACTED]", text)
    text = re.sub(r"(?i)\b(?:Bearer|Basic)\s+[^\s,;\"']+", "[REDACTED authorization]", text)
    text = re.sub(r"(?i)(\b" + _SECRET_KEY + r"\b[\"']?\s*[:=]\s*)(?:\"[^\"]*\"|'[^']*'|[^,;}]+)",
                  r"\1[REDACTED]", text)
    text = re.sub(r"(?i)(--" + _SECRET_KEY + r"\s+)(?:\"[^\"]*\"|'[^']*'|[^\s]+)", r"\1[REDACTED]", text)
    # Preserve URL/key syntax until structural redaction has completed. A short
    # literal password such as '/' must not disable the URL credential rules.
    for secret in sorted(variants, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    text = text.strip()
    return text if len(text) <= 2000 else text[:1976] + " …[该行已截断]"


class _OutputLines:
    """Incremental, bounded UTF-8 framing for one SSH output stream."""

    def __init__(self, emit: Callable[[str], None]):
        self.decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.emit = emit
        self.pending = ""
        self.discarding = False

    def _append(self, value):
        if self.discarding:
            return
        if len(self.pending) + len(value) > _LINE_LIMIT:
            self.pending = ""
            self.discarding = True
            self.emit("[安装输出过长，已省略该行]")
        else:
            self.pending += value

    def feed(self, chunk: bytes, *, final=False):
        parts = re.split(r"[\r\n]", self.decoder.decode(chunk, final=final))
        for part in parts[:-1]:
            self._append(part)
            if not self.discarding and self.pending:
                self.emit(self.pending)
            self.pending, self.discarding = "", False
        self._append(parts[-1])
        if final:
            if not self.discarding and self.pending:
                self.emit(self.pending)
            self.pending, self.discarding = "", False


def _fingerprint(key) -> str:
    return "SHA256:" + base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip("=")


_HOST_LOCK = threading.Lock()


class _HostPolicy(paramiko.MissingHostKeyPolicy):
    def __init__(self, path: Path, expected: str | None):
        self.path, self.expected = path, expected

    def missing_host_key(self, client, hostname, key):
        if self.expected and _fingerprint(key) != self.expected:
            raise RemoteError("SSH_HOST_KEY_MISMATCH", "SSH 主机指纹与已验证的机器不一致，请检查目标机器")
        # Another connection may have saved the host while this one negotiated.
        with _HOST_LOCK:
            keys = paramiko.HostKeys()
            if self.path.exists():
                keys.load(str(self.path))
            known = keys.lookup(hostname)
            if known and (key.get_name() not in known or known[key.get_name()] != key):
                raise RemoteError("SSH_HOST_KEY_MISMATCH", "SSH 主机密钥已变化，已停止连接；请核实机器身份")
            keys.add(hostname, key.get_name(), key)
            temporary = self.path.with_name(self.path.name + "." + os.urandom(8).hex())
            try:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(descriptor)
                keys.save(str(temporary))
                os.chmod(temporary, 0o600)
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
            client.get_host_keys().add(hostname, key.get_name(), key)


def _install_path(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", value):
        raise RemoteError("SSH_INSTALL_PATH_INVALID", "Agent 安装目录需要使用简单的 Linux 绝对路径")
    path = PurePosixPath(value)
    normalized = "/" + str(path).lstrip("/")
    if ".." in path.parts or normalized in {"/", "/opt", "/data", "/data2", "/usr", "/root", "/var"}:
        raise RemoteError("SSH_INSTALL_PATH_INVALID", "请使用独立的 Agent 子目录作为安装目录")
    return normalized


def _runtime_selection_command() -> str:
    # Ship the same source in the Agent bundle for manual installations. Do not
    # depend on interactive shell startup files or a sudo-inherited PATH.
    source = Path(__file__).with_name("agent_bootstrap.sh").read_text()
    source += r'''
if vla_select_runtime >/dev/null 2>&1; then
    printf '%s\n' "$VLA_RUNTIME_KIND" "$VLA_BOOTSTRAP_PYTHON" "$VLA_CONDA_EXE"
else
    printf '%s\n' ERROR "${VLA_RUNTIME_ERROR:-无法选择 Agent Python 环境，请检查 Conda 或 Python 3.11+}"
fi
'''
    return "bash -c " + shlex.quote(source)


_INSTALLATION_IDLE_SCRIPT = r'''
def installation_idle(prefix, proc_root=pathlib.Path('/proc')):
    target = os.path.normpath(str(prefix))
    try:
        processes = proc_root.iterdir()
        for process in processes:
            if not process.name.isdecimal():
                continue
            try:
                with (process / 'cmdline').open('rb') as reader:
                    encoded = reader.read(65537)
                if len(encoded) > 65536:
                    continue
                args = [os.fsdecode(arg) for arg in encoded.rstrip(b'\0').split(b'\0')]
                if not any(arg.endswith('/install_agent.py') for arg in args):
                    continue
                for index, arg in enumerate(args[:-1]):
                    if arg == '--prefix' and os.path.normpath(args[index + 1]) == target:
                        return False
            except (OSError, ValueError):
                # Processes may exit while /proc is being enumerated.
                continue
    except OSError:
        pass
    return True
'''


_PROBE_SCRIPT = r'''
import json, os, pathlib, platform, pwd, shutil, sqlite3, subprocess, sys
''' + _INSTALLATION_IDLE_SCRIPT + r'''
prefix = pathlib.Path(sys.argv[1])
checks = []
def check(name, ok, message):
    checks.append({"name": name, "ok": bool(ok), "message": message})
check("linux", sys.platform == "linux", "目标系统需要 Linux")
runtime_kind = sys.argv[2] if len(sys.argv) > 2 else "venv"
conda = sys.argv[3] if len(sys.argv) > 3 else ""
service_user = sys.argv[4] if len(sys.argv) > 4 else pwd.getpwuid(os.getuid()).pw_name
try:
    pwd.getpwnam(service_user)
    check("agent_user", True, "Agent 将使用 SSH 登录用户 " + service_user + " 运行")
except KeyError:
    check("agent_user", False, "Agent 运行账户不存在：" + service_user)
version = platform.python_version()
if runtime_kind == "conda":
    check("agent_runtime", pathlib.Path(conda).is_file() and os.access(conda, os.X_OK),
          "发现 Conda：" + conda + "；将创建 Agent 专用 Conda 环境 " + str(prefix / ".venv") + "（Python 3.11）")
    check("python", sys.version_info >= (3, 9),
          "Conda 引导解释器：" + sys.executable + "（Python " + version + "）；Agent 使用独立 Python 3.11")
elif runtime_kind == "existing":
    check("python", sys.version_info >= (3, 11),
          "使用已有 Agent 解释器进行维护检查：" + sys.executable + "（Python " + version + "）")
else:
    check("python", sys.version_info >= (3, 11),
          "未发现 Conda，使用默认 Python：" + sys.executable + "（Python " + version + "，要求 3.11+）；将创建 Agent 专用 venv")
for command in ("systemctl", "systemd-analyze", "git", "runuser", "tar", "sha256sum"):
    check(command, shutil.which(command) is not None, "检查系统命令 " + command)
check("systemd", pathlib.Path("/run/systemd/system").is_dir(), "Agent 使用 systemd 常驻运行")
check("installation_idle", installation_idle(prefix), "同一安装目录不能存在仍在执行的 Agent 安装进程；请等待完成后重试")
result = {"hostname": platform.node(), "checks": checks}
unit = "termitech-vla-agent.service"
service = subprocess.run(["systemctl", "show", unit, "--property=ExecStart", "--value"], capture_output=True, text=True) if shutil.which("systemctl") else None
has_service = service is not None and service.returncode == 0 and bool(service.stdout.strip())
if has_service:
    expected = str(prefix / ".venv/bin/vla-agent")
    # Match the executable as a token, never a substring of another prefix.
    import re
    matches = bool(re.search(r"(?:^|[\s=;{])" + re.escape(expected) + r"(?:[\s;}\x00]|$)", service.stdout))
    check("service_location", matches, "已有 Agent 服务必须使用同一安装目录")
old = prefix / "config/agent.json"
if old.is_file():
    try:
        config = json.loads(old.read_text())
        existing = {key: config[key] for key in ("name", "roots", "output_root", "state_dir") if key in config}
        existing["profiles"] = []
        for profile in config.get("profiles", []):
            safe = {key: profile[key] for key in ("id", "name", "adapter", "code_dir", "python", "recipe") if key in profile}
            safe["metadata"] = {key: profile.get("metadata", {})[key] for key in ("validated", "simulation", "onboarding_id") if key in profile.get("metadata", {})}
            existing["profiles"].append(safe)
        state = pathlib.Path(config["state_dir"]) / "agent.sqlite3"
        if state.is_file():
            reader = ("import json,pathlib,sqlite3,sys; p=pathlib.Path(sys.argv[1]).resolve(); "
                "d=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True); "
                "r=d.execute('SELECT value FROM credentials WHERE id=1').fetchone(); "
                "v=json.loads(r[0]) if r else {}; "
                "n=d.execute(\"SELECT COUNT(*) FROM attempts WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')\").fetchone()[0]; "
                "print(json.dumps({'node_id':v.get('node_id'),'active_attempts':n}));d.close()")
            agent_python = prefix / ".venv/bin/python"
            reader_python = str(agent_python) if agent_python.is_file() else sys.executable
            args = [reader_python, "-I", "-c", reader, str(state)]
            owner = state.stat().st_uid
            if os.geteuid() == 0 and owner != 0:
                args = ["runuser", "-u", pwd.getpwuid(owner).pw_name, "--"] + args
            identity = subprocess.run(args, capture_output=True, text=True, timeout=10)
            if identity.returncode:
                raise ValueError("state_read_failed")
            existing.update(json.loads(identity.stdout))
            check("idle", existing.get("active_attempts", 0) == 0, "升级前需要先结束已有训练任务")
        result["existing_agent"] = existing
        check("existing_config", True, "已发现 Agent 配置，将保留机器身份和状态目录")
    except Exception:
        check("existing_config", False, "已有 Agent 配置或状态无法读取，请先检查当前安装")
ancestor = prefix
while not ancestor.exists() and ancestor != ancestor.parent:
    ancestor = ancestor.parent
check("install_directory", ancestor.is_dir() and os.access(ancestor, os.W_OK), "安装目录的父目录需要可写")
print(json.dumps(result, ensure_ascii=False))
'''


_UNINSTALL_SCRIPT = r'''
import datetime, json, os, pathlib, pwd, re, shutil, sqlite3, subprocess, sys, uuid

class UninstallError(Exception):
    pass

def fail(code):
    raise UninstallError(code)

def overlap(a, b):
    return a == b or a in b.parents or b in a.parents

def safe_path(value):
    if not isinstance(value, str) or not value.startswith('/') or any(ord(c) < 32 for c in value):
        fail('AGENT_UNINSTALL_UNSAFE_PATH')
    path = pathlib.Path(value)
    if '..' in path.parts or len(path.parts) < 3:
        fail('AGENT_UNINSTALL_UNSAFE_PATH')
    # Move the named directory itself, never follow an alias into training data.
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        fail('AGENT_UNINSTALL_UNSAFE_PATH')
    return path

def command(*args, required=True):
    result = subprocess.run(list(args), capture_output=True, text=True, timeout=40)
    if required and result.returncode:
        fail('AGENT_UNINSTALL_SERVICE_FAILED')
    return result

def require_idle(state, runtime_python=None):
    database = state / 'agent.sqlite3'
    if not database.is_file() or database.is_symlink():
        fail('AGENT_UNINSTALL_STATE_INVALID')
    query = "SELECT COUNT(*) FROM attempts WHERE state NOT IN ('SUCCEEDED','FAILED','CANCELLED')"
    # A read-only WAL reader may create -shm; keep ownership with the Agent.
    reader = ("import pathlib,sqlite3,sys; p=pathlib.Path(sys.argv[1]).resolve(); "
              "d=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True); "
              "tables={r[0] for r in d.execute(\"SELECT name FROM sqlite_master WHERE type='table'\")}; "
              "assert {'credentials','attempts'}.issubset(tables); "
              "n=d.execute(" + repr(query) + ").fetchone()[0]; "
              "n+=d.execute('SELECT COUNT(*) FROM events').fetchone()[0] if 'events' in tables else 0; "
              "print(n);d.close()")
    args = [runtime_python or sys.executable, '-I', '-c', reader, str(database)]
    owner = database.stat().st_uid
    if os.geteuid() == 0 and owner != 0:
        args = ['runuser', '-u', pwd.getpwuid(owner).pw_name, '--'] + args
    result = command(*args, required=False)
    try:
        if result.returncode:
            fail('AGENT_UNINSTALL_STATE_INVALID')
        count = int(result.stdout.strip())
    except ValueError:
        fail('AGENT_UNINSTALL_STATE_INVALID')
    if count:
        fail('AGENT_ACTIVE_TASKS')

def uninstall_agent(value, unit_directory=pathlib.Path('/etc/systemd/system')):
    prefix = safe_path(value)
    unit = 'termitech-vla-agent.service'
    config_path = prefix / 'config/agent.json'
    loaded = command('systemctl', 'show', unit, '--property=LoadState', '--value', required=False)
    if loaded.stdout.strip() == 'not-found':
        if config_path.exists() or config_path.is_symlink():
            fail('AGENT_UNINSTALL_UNVERIFIED')
        return {'backup_dir': None, 'removed_paths': [], 'service_removed': True}
    if loaded.returncode or loaded.stdout.strip() != 'loaded':
        fail('AGENT_UNINSTALL_SERVICE_FAILED')
    executable = str(prefix / '.venv/bin/vla-agent')
    startup = command('systemctl', 'show', unit, '--property=ExecStart', '--value').stdout
    def token(value):
        return bool(re.search(r'(?:^|[\s=;{])' + re.escape(value) + r'(?:[\s;}\x00]|$)', startup))
    if not token(executable) or not token(str(config_path)):
        fail('AGENT_INSTALL_PATH_CONFLICT')
    unit_path = unit_directory / unit
    fragment = command('systemctl', 'show', unit, '--property=FragmentPath', '--value').stdout.strip()
    if fragment != str(unit_path) or unit_path.is_symlink() or not unit_path.is_file():
        fail('AGENT_UNINSTALL_UNVERIFIED')
    if config_path.is_symlink() or not config_path.is_file():
        fail('AGENT_UNINSTALL_UNVERIFIED')
    # The installer writes both files as root, with no group/other write bit.
    # A writable configuration cannot authorize moving an external directory.
    if any(path.stat().st_uid != os.geteuid() or path.stat().st_mode & 0o022 for path in (unit_path, config_path)):
        fail('AGENT_UNINSTALL_UNVERIFIED')
    safe_path(str(config_path.parent))
    try:
        config = json.loads(config_path.read_text())
        state = safe_path(config['state_dir'])
        output = safe_path(config['output_root'])
        profiles = config['profiles']
        if not isinstance(profiles, list) or not profiles:
            fail('AGENT_UNINSTALL_UNVERIFIED')
        codes = [safe_path(p['code_dir']) for p in profiles]
    except (ValueError, TypeError, KeyError):
        fail('AGENT_UNINSTALL_UNVERIFIED')
    components = [prefix / name for name in ('.venv', 'config', 'home', 'tmp')]
    if not any(overlap(state, path) for path in components):
        components.append(state)
    else:
        fail('AGENT_UNINSTALL_UNSAFE_PATH')
    protected = [output, *codes]
    for path in components:
        safe_path(str(path))
        if any(overlap(path, protected_path) for protected_path in protected):
            fail('AGENT_UNINSTALL_PROTECTED_PATH')
    for profile in profiles:
        interpreter = pathlib.Path(profile['python'])
        if not interpreter.is_absolute() or '..' in interpreter.parts:
            fail('AGENT_UNINSTALL_UNVERIFIED')
        # Smoke fixtures intentionally use the Agent venv. Real training does not.
        if profile.get('adapter') != 'smoke' and any(overlap(path, interpreter) for path in components):
            fail('AGENT_UNINSTALL_TRAINING_ENV_SHARED')
    # Only an Agent-owned state directory can be moved, even outside the prefix.
    allowed_state = {'agent.sqlite3', 'agent.sqlite3-wal', 'agent.sqlite3-shm',
                     'instance-id', 'agent.lock', 'locks', 'runner-logs'}
    if not state.is_dir() or any(p.name not in allowed_state or p.is_symlink() for p in state.iterdir()):
        fail('AGENT_UNINSTALL_STATE_INVALID')
    agent_python = prefix / '.venv/bin/python'
    reader_python = str(agent_python) if agent_python.is_file() else sys.executable
    require_idle(state, reader_python)
    was_active = command('systemctl', 'is-active', '--quiet', unit, required=False).returncode == 0
    was_enabled = command('systemctl', 'is-enabled', '--quiet', unit, required=False).returncode == 0
    command('systemctl', 'stop', unit)
    command('systemctl', 'disable', unit)
    try:
        # No Runner may have accepted a final command while stop was in progress.
        require_idle(state, reader_python)
    except UninstallError:
        if was_enabled:
            command('systemctl', 'enable', unit, required=False)
        if was_active:
            command('systemctl', 'start', unit, required=False)
        raise
    if command('systemctl', 'is-active', '--quiet', unit, required=False).returncode == 0:
        fail('AGENT_UNINSTALL_SERVICE_FAILED')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = prefix.with_name(prefix.name + '.uninstalled.' + stamp + '.' + uuid.uuid4().hex[:12])
    backup.mkdir(mode=0o700)
    backup.chmod(0o700)
    removed = []
    for path in components:
        if path.exists():
            destination = backup / ('state' if path == state else path.name)
            shutil.move(str(path), str(destination))
            removed.append(str(path))
    shutil.move(str(unit_path), str(backup / unit))
    removed.append(str(unit_path))
    command('systemctl', 'daemon-reload')
    command('systemctl', 'reset-failed', unit, required=False)
    verified = command('systemctl', 'show', unit, '--property=LoadState', '--value', required=False)
    if verified.stdout.strip() != 'not-found' or config_path.exists():
        fail('AGENT_UNINSTALL_VERIFY_FAILED')
    return {'backup_dir': str(backup), 'removed_paths': removed, 'service_removed': True}

if __name__ == '__main__':
    try:
        print(json.dumps(uninstall_agent(sys.argv[1])))
    except UninstallError as error:
        print(json.dumps({'error': str(error)}))
    except Exception:
        print(json.dumps({'error': 'AGENT_UNINSTALL_FAILED'}))
'''


class SSHRemote:
    def __init__(self, host: str, port: int, username: str, password: str,
                 state_dir: str | Path, expected_fingerprint: str | None = None):
        if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]+", host) or len(host) > 253:
            raise RemoteError("SSH_HOST_INVALID", "请输入有效的裸金属 IP 或主机名")
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise RemoteError("SSH_PORT_INVALID", "SSH 端口需要在 1 到 65535 之间")
        if not isinstance(username, str) or not username or len(username) > 128 or any(ord(c) < 32 for c in username):
            raise RemoteError("SSH_USERNAME_INVALID", "SSH 用户名无效")
        if not isinstance(password, str) or not password or len(password.encode()) > 4096 or "\n" in password or "\r" in password or "\x00" in password:
            raise RemoteError("SSH_PASSWORD_INVALID", "请输入有效的 SSH 密码")
        self.host, self.port, self.username = host, port, username
        self._password, self.expected_fingerprint = password, expected_fingerprint
        self.state_dir = Path(state_dir)
        self.client = None
        self.fingerprint: str | None = None
        self._is_root: bool | None = None
        self._runtime_selection: dict | None = None
        self._closed = threading.Event()
        self._lifecycle_lock = threading.Lock()

    def connect(self):
        self._closed.clear()
        known_hosts = self.state_dir / "ssh_known_hosts"
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if known_hosts.is_symlink() or (known_hosts.exists() and not known_hosts.is_file()):
                raise RemoteError("SSH_KNOWN_HOSTS_INVALID", "平台 SSH 主机记录文件不可用")
            if known_hosts.exists():
                known_hosts.chmod(0o600)
        except OSError:
            raise RemoteError("SSH_STATE_UNAVAILABLE", "平台无法保存 SSH 主机记录，请检查部署状态目录权限") from None
        client = paramiko.SSHClient()
        try:
            # Expose the connecting client to close(), so a connection-test
            # deadline can interrupt a slow handshake or password exchange.
            with self._lifecycle_lock:
                if self._closed.is_set():
                    raise RemoteError("SSH_CANCELLED", "SSH 连接检查已取消，请重新测试连接")
                self.client = client
            if known_hosts.exists():
                client.load_host_keys(str(known_hosts))
            client.set_missing_host_key_policy(_HostPolicy(known_hosts, self.expected_fingerprint))
            client.connect(self.host, port=self.port, username=self.username, password=self._password,
                           allow_agent=False, look_for_keys=False, timeout=15, auth_timeout=15,
                           banner_timeout=15, channel_timeout=20)
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise RemoteError("SSH_CONNECTION_FAILED", "SSH 连接未建立，请检查目标机器网络")
            fingerprint = _fingerprint(transport.get_remote_server_key())
            if self.expected_fingerprint and fingerprint != self.expected_fingerprint:
                raise RemoteError("SSH_HOST_KEY_MISMATCH", "SSH 主机指纹与已验证的机器不一致，请检查目标机器")
            transport.set_keepalive(15)
            with self._lifecycle_lock:
                if self._closed.is_set():
                    raise RemoteError("SSH_CANCELLED", "SSH 连接检查已取消，请重新测试连接")
                self.fingerprint = fingerprint
            return self
        except RemoteError:
            client.close()
            raise
        except paramiko.BadHostKeyException:
            client.close()
            raise RemoteError("SSH_HOST_KEY_MISMATCH", "SSH 主机密钥已变化，已停止连接；请核实机器身份") from None
        except paramiko.AuthenticationException:
            client.close()
            raise RemoteError("SSH_AUTH_FAILED", "SSH 认证失败，请检查用户名、密码以及服务器是否允许密码登录") from None
        except (OSError, paramiko.SSHException, ValueError):
            client.close()
            raise RemoteError("SSH_CONNECTION_FAILED", "SSH 连接失败，请检查 IP、端口、网络和服务器 SSH 服务") from None

    def path_suggestions(self, prefix: str, *, limit: int = 30, directory_only: bool = True) -> dict:
        """List names through SFTP as the SSH user, without shell or file reads."""
        if (not isinstance(prefix, str) or len(prefix) > 4096
                or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in prefix)
                or (prefix and not prefix.startswith("/")) or ".." in PurePosixPath(prefix).parts):
            raise RemoteError("INVALID_PATH_PREFIX", "路径前缀必须是绝对路径，且不能包含控制字符或上级目录")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 50 or not isinstance(directory_only, bool):
            raise RemoteError("INVALID_PATH_QUERY", "路径联想参数无效")
        if self.client is None:
            raise RemoteError("SSH_NOT_CONNECTED", "SSH 尚未连接，请重新连接")
        normalized = "/" + posixpath.normpath(prefix or "/").lstrip("/")
        parent = normalized if not prefix or prefix.endswith("/") else posixpath.dirname(normalized)
        name_prefix = "" if not prefix or prefix.endswith("/") else posixpath.basename(normalized)
        result = {"prefix": prefix, "parent": parent, "entries": [], "truncated": False}
        sftp = None
        try:
            sftp = self.client.open_sftp()
            sftp.get_channel().settimeout(8)
            # A trailing slash explicitly opens a directory; without one, offer
            # that directory itself along with other matching siblings.
            if not stat.S_ISDIR(sftp.stat(parent).st_mode):
                raise RemoteError("PATH_NOT_DIRECTORY", "该路径不是目录，可继续手动填写路径")
            started = time.monotonic()
            scanned, entries = 0, []
            for entry in sftp.listdir_iter(parent, read_aheads=1):
                if scanned >= 5000 or time.monotonic() - started >= 3:
                    result["truncated"] = True
                    break
                scanned += 1
                name = entry.filename
                if (not isinstance(name, str) or name in {".", ".."} or "/" in name
                        or not name.startswith(name_prefix)
                        or any(ord(char) < 32 or 127 <= ord(char) < 160 for char in name)):
                    continue
                path = posixpath.join(parent, name)
                if len(path) + 1 > 4096:
                    continue
                mode = entry.st_mode
                if mode is None or stat.S_ISLNK(mode):
                    try:
                        mode = sftp.stat(path).st_mode
                    except OSError:
                        continue
                kind = "directory" if stat.S_ISDIR(mode) else "file" if stat.S_ISREG(mode) and mode & 0o111 else None
                if kind is None or (directory_only and kind != "directory"):
                    continue
                entries.append({"name": name, "path": path + "/" if kind == "directory" else path, "kind": kind})
            entries.sort(key=lambda item: (item["kind"] != "directory", item["name"]))
            result.update(entries=entries[:limit], truncated=result["truncated"] or len(entries) > limit)
            return result
        except RemoteError:
            raise
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EPERM}:
                raise RemoteError("PATH_PERMISSION_DENIED", "SSH 用户无权列出该目录，可继续手动填写路径") from None
            if exc.errno == errno.ENOENT:
                raise RemoteError("PATH_NOT_FOUND", "该目录不存在，请检查路径后重试") from None
            raise RemoteError("PATH_UNAVAILABLE", "该目录暂不可访问，可继续手动填写路径") from None
        finally:
            if sftp is not None:
                sftp.close()

    def _execute(self, command: str, *, sudo: bool = False, timeout: float = 30,
                 on_output: Callable[[str], None] | None = None,
                 on_idle: Callable[[], None] | None = None) -> tuple[int, str]:
        if self.client is None:
            raise RemoteError("SSH_NOT_CONNECTED", "SSH 尚未连接，请重新测试连接")
        channel = None
        try:
            client = self.client
            if client is None:
                raise RemoteError("SSH_DISCONNECTED", "SSH 连接已断开，请重新连接")
            transport = client.get_transport()
            if transport is None or not transport.is_active():
                raise RemoteError("SSH_DISCONNECTED", "SSH 连接已断开，请重新连接")
            channel = transport.open_session(timeout=15)
            channel.settimeout(15)
            if sudo:
                command = "sudo -S -k -p '' -- sh -c " + shlex.quote(command)
            channel.exec_command(command)
            if sudo:
                channel.sendall((self._password + "\n").encode())
            channel.shutdown_write()
            started = time.monotonic()
            last_activity = started
            output = bytearray()
            streamed, line_count, stream_limited = 0, 0, False

            def output_line(line):
                nonlocal streamed, line_count, stream_limited
                if on_output is None or stream_limited:
                    return
                if streamed + len(line) > _STREAM_LIMIT or line_count >= 10000:
                    on_output("[安装输出较多，后续原始输出已省略；仍在等待安装结果]")
                    stream_limited = True
                    return
                streamed += len(line)
                line_count += 1
                on_output(line)

            streams = (_OutputLines(output_line), _OutputLines(output_line))

            def check_deadline():
                now = time.monotonic()
                if now - started > timeout:
                    raise RemoteError("SSH_COMMAND_TIMEOUT", "远程操作超时；安装可能仍在执行，请检查机器状态后重试")
                return now

            while True:
                # Drain both streams on every iteration so pip cannot fill stderr
                # while stdout waits, or block before the exit status is sent.
                for stream, (ready, receive) in zip(streams, ((channel.recv_ready, channel.recv), (channel.recv_stderr_ready, channel.recv_stderr))):
                    # Bound a drain pass, so continuous output cannot starve the
                    # deadline or the other stream.
                    for _ in range(8):
                        if not ready():
                            break
                        check_deadline()
                        chunk = receive(32768)
                        if not chunk:
                            break
                        if len(output) < _OUTPUT_LIMIT:
                            output.extend(chunk[:_OUTPUT_LIMIT - len(output)])
                        if on_output is not None:
                            stream.feed(chunk)
                        last_activity = check_deadline()
                if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                    if on_output is not None:
                        for stream in streams:
                            stream.feed(b"", final=True)
                    return channel.recv_exit_status(), output.decode("utf-8", errors="replace")
                if not transport.is_active():
                    raise RemoteError("SSH_DISCONNECTED", "SSH 连接已断开；远程安装可能仍在执行，请检查后重新连接")
                now = check_deadline()
                if on_idle is not None and now - last_activity >= 15:
                    on_idle()
                    last_activity = now
                time.sleep(0.03)
        except RemoteError:
            raise
        except (OSError, EOFError, paramiko.SSHException):
            raise RemoteError("SSH_COMMAND_FAILED", "远程命令执行失败，请检查 SSH 连接和目标机器状态") from None
        finally:
            if channel is not None:
                channel.close()

    def run(self, command: str, *, sudo: bool = False, timeout: float = 30) -> str:
        status, output = self._execute(command, sudo=sudo, timeout=timeout)
        if status:
            raise RemoteError("SSH_COMMAND_FAILED", "远程命令执行失败，请检查目标机器配置")
        return output

    def _select_runtime(self, *, refresh: bool = False) -> dict:
        if self._runtime_selection is not None and not refresh:
            return self._runtime_selection
        self._runtime_selection = None
        parts = self.run(_runtime_selection_command()).splitlines()
        if parts and parts[0] == "ERROR":
            message = sanitize_install_output(parts[1] if len(parts) == 2 else "无法选择 Agent Python 环境", (self._password,))
            raise RemoteError("SSH_PYTHON_REQUIRED", message)
        if len(parts) != 3 or parts[0] not in {"conda", "venv"}:
            raise RemoteError("SSH_PYTHON_REQUIRED", "无法选择 Agent Python 环境，请检查 Conda 或默认 Python 3.11+")
        kind, interpreter, conda = parts
        def executable_path(value):
            return value.startswith("/") and not any(ord(c) < 32 or ord(c) == 127 for c in value)
        if not executable_path(interpreter) or (kind == "conda" and not executable_path(conda)) or (kind == "venv" and conda):
            raise RemoteError("SSH_PROBE_FAILED", "机器返回了无效的 Agent Python 路径")
        self._runtime_selection = {"kind": kind, "python": interpreter, "conda": conda}
        return self._runtime_selection

    def probe(self, install_dir: str) -> dict:
        prefix = _install_path(install_dir)
        self._is_root = self.run("id -u").strip() == "0"
        privileged = self._is_root
        if not privileged:
            try:
                privileged = self.run("id -u", sudo=True).strip() == "0"
            except RemoteError:
                privileged = False
        root_check = {"name": "privileges", "ok": privileged, "message": "安装 Agent 需要 root 或可使用密码的 sudo 权限"}
        selection_error = None
        try:
            runtime = self._select_runtime(refresh=True)
        except RemoteError as error:
            if error.code not in {"SSH_PYTHON_REQUIRED", "SSH_PROBE_FAILED"}:
                raise
            selection_error = {"name": "python_installation", "ok": False, "message": error.message}
            # A removed/broken bootstrap Conda must not prevent reading the
            # existing identity and safety checks or uninstalling the Agent.
            # This fallback never authorizes a new installation or upgrade.
            existing_python = prefix + "/.venv/bin/python"
            quoted = shlex.quote(existing_python)
            found = self.run("if [ -x " + quoted + " ]; then printf '%s\\n' " + quoted + "; fi").strip()
            if found != existing_python:
                return {"hostname": self.host, "checks": [root_check, selection_error], "can_install": False}
            runtime = {"kind": "existing", "python": existing_python, "conda": ""}
        command = "env -u PYTHONHOME -u PYTHONPATH " + " ".join(shlex.quote(part) for part in (
            runtime["python"], "-I", "-c", _PROBE_SCRIPT, prefix, runtime["kind"], runtime["conda"], self.username))
        try:
            result = json.loads(self.run(command, sudo=privileged and not self._is_root))
            if not isinstance(result, dict) or not isinstance(result.get("checks"), list):
                raise ValueError()
        except (ValueError, TypeError):
            raise RemoteError("SSH_PROBE_FAILED", "机器检查返回无效结果，请确认远程 Python 和系统环境") from None
        result["checks"].insert(0, root_check)
        if selection_error is not None:
            result["checks"].insert(1, selection_error)
            result["_maintenance_python"] = runtime["python"]
        result["can_install"] = all(check.get("ok") is True for check in result["checks"])
        return result

    def install(self, bundle_path: str | Path, config: dict, install_dir: str,
                emit: Callable[[str, str], None]) -> None:
        prefix = _install_path(install_dir)
        bundle = Path(bundle_path)
        if not bundle.is_file():
            raise RemoteError("AGENT_BUNDLE_MISSING", "平台尚未准备好 Agent 安装包")
        if self._is_root is None:
            self._is_root = self.run("id -u").strip() == "0"
        runtime = self._select_runtime()
        sudo = not self._is_root
        temporary = None
        sftp = None
        try:
            emit("uploading", "正在上传 Agent 安装包和配置")
            temporary = self.run("umask 077; mktemp -d /tmp/vla-agent-install.XXXXXXXXXX").strip()
            if not re.fullmatch(r"/tmp/vla-agent-install\.[A-Za-z0-9]+", temporary):
                raise RemoteError("SSH_TEMP_DIR_FAILED", "无法在远程机器创建私有安装目录")
            remote_bundle = temporary + "/vla-agent-bundle.tar.gz"
            remote_config = temporary + "/agent.json"
            sftp = self.client.open_sftp()
            sftp.put(str(bundle), remote_bundle)
            sftp.chmod(remote_bundle, 0o600)
            # Do not stage private configuration on the control plane's disk.
            payload = json.dumps(config, ensure_ascii=False).encode()
            sftp.putfo(io.BytesIO(payload), remote_config)
            sftp.chmod(remote_config, 0o600)
            sftp.close()
            sftp = None
            emit("verifying", "正在校验安装包完整性")
            local_hash = hashlib.sha256(bundle.read_bytes()).hexdigest()
            remote_hash = self.run("sha256sum -- " + shlex.quote(remote_bundle)).split()[0]
            if remote_hash != local_hash:
                raise RemoteError("AGENT_BUNDLE_CHECKSUM_FAILED", "Agent 安装包传输校验失败，已停止安装")
            emit("installing", "正在安装 Agent 独立环境和系统服务，首次安装可能需要数分钟")
            secrets = [self._password]

            def collect_secrets(value):
                if isinstance(value, dict):
                    for key, item in value.items():
                        if isinstance(item, str) and re.fullmatch(_SECRET_KEY, str(key), re.IGNORECASE):
                            secrets.append(item)
                        elif isinstance(item, (dict, list)):
                            collect_secrets(item)
                elif isinstance(value, list):
                    for item in value:
                        collect_secrets(item)

            collect_secrets(config)
            current_stage = "installing"

            def publish_output(line):
                nonlocal current_stage
                if line.startswith("VLA_INSTALL_EVENT "):
                    try:
                        event = json.loads(line[len("VLA_INSTALL_EVENT "):])
                        if (isinstance(event, dict) and event.get("phase") in _INSTALL_PHASES
                                and isinstance(event.get("message"), str)
                                and event.get("level", "info") in {"info", "warning", "error"}):
                            phase = event["phase"]
                            current_stage = "STARTING" if phase == "STARTING" else "INSTALL_" + phase
                            text = sanitize_install_output(event["message"], secrets)
                            level = event.get("level", "info")
                            if level != "info":
                                text = ("[错误] " if level == "error" else "[警告] ") + text
                            if text:
                                emit(current_stage, text)
                            return
                    except (ValueError, TypeError):
                        pass
                    # A malformed event is diagnostic text, never a stage change.
                    line = line[len("VLA_INSTALL_EVENT "):]
                text = sanitize_install_output(line, secrets)
                if text:
                    emit(current_stage, text)

            def publish_waiting():
                emit(current_stage, "远程安装仍在执行，最近 15 秒未收到新输出；正在等待下一条日志")

            install_argv = (runtime["python"], "-I", "-u", temporary + "/install_agent.py",
                            "--runtime-kind", runtime["kind"], "--conda-executable", runtime["conda"],
                            "--service-user", self.username,
                            "--config", remote_config, "--prefix", prefix)
            command = ("umask 022; tar -xzf " + shlex.quote(remote_bundle) + " -C " + shlex.quote(temporary)
                       + " && env -u PYTHONHOME -u PYTHONPATH " + " ".join(shlex.quote(part) for part in install_argv))
            status, private_output = self._execute(command, sudo=sudo, timeout=900,
                                                   on_output=publish_output, on_idle=publish_waiting)
            if status:
                guards = (("仍有活动", "AGENT_ACTIVE_TASKS", "该 Agent 仍有活动任务，请等待任务结束后再升级"),
                          ("活动或待对账", "AGENT_ACTIVE_TASKS", "该 Agent 仍有活动或待对账任务，请先完成任务对账"),
                          ("另一安装目录", "AGENT_INSTALL_PATH_CONFLICT", "已有 Agent 服务使用其他安装目录，请保留原安装路径"),
                          ("不能更换 Agent state_dir", "AGENT_STATE_DIR_CONFLICT", "升级必须保留已有 Agent 状态目录"),
                          ("训练代码目录或 Python 解释器不可用", "AGENT_TRAINING_ENV_INVALID", "训练代码目录或 Python 解释器不可用，请检查训练环境路径"))
                for marker, code, message in guards:
                    if marker in private_output:
                        raise RemoteError(code, message)
                raise RemoteError("AGENT_INSTALL_FAILED", f"Agent 安装失败（退出码 {status}），请查看部署日志中的具体错误；已有服务可能需要检查恢复")
            emit("starting", "正在确认 Agent 系统服务已启动")
            status, _ = self._execute("systemctl is-active --quiet termitech-vla-agent.service", sudo=sudo)
            if status:
                raise RemoteError("AGENT_SERVICE_NOT_ACTIVE", "Agent 已安装，但系统服务未处于运行状态，请检查远程服务日志")
            emit("started", "Agent 系统服务已启动，等待机器向中台登记")
        except RemoteError:
            raise
        except (OSError, EOFError, paramiko.SSHException, IndexError, TypeError):
            raise RemoteError("AGENT_UPLOAD_FAILED", "Agent 安装文件上传或校验失败，请检查磁盘空间及 SSH 连接") from None
        finally:
            if sftp is not None:
                try:
                    sftp.close()
                except (OSError, paramiko.SSHException):
                    pass
            if temporary is not None and re.fullmatch(r"/tmp/vla-agent-install\.[A-Za-z0-9]+", temporary):
                try:
                    self.run("rm -rf -- " + shlex.quote(temporary), sudo=sudo, timeout=15)
                except RemoteError:
                    pass

    def uninstall(self, install_dir: str, emit: Callable[[str, str], None]) -> dict:
        """Remove this Agent's service/identity, retaining training directories."""
        prefix = _install_path(install_dir)
        emit("CHECKING", "正在检查 Agent 安装位置、权限和任务状态")
        probe = self.probe(prefix)
        checks = {check.get("name"): check.get("ok") for check in probe.get("checks", [])}
        if checks.get("privileges") is not True:
            raise RemoteError("SSH_PRIVILEGES_REQUIRED", "卸载 Agent 需要 root 或可使用密码的 sudo 权限")
        if checks.get("service_location") is False:
            raise RemoteError("AGENT_INSTALL_PATH_CONFLICT", "已有 Agent 服务使用其他安装目录，已停止卸载")
        if checks.get("idle") is False:
            raise RemoteError("AGENT_ACTIVE_TASKS", "该 Agent 仍有活动或待对账任务，请先结束任务并完成对账")
        if checks.get("existing_config") is False:
            raise RemoteError("AGENT_UNINSTALL_UNVERIFIED", "无法确认已有 Agent 配置和状态，已停止卸载")
        interpreter = probe.get("_maintenance_python") or self._select_runtime()["python"]
        emit("UNINSTALLING", "正在停止 Agent、移除系统服务并备份程序与身份状态")
        command = "env -u PYTHONHOME -u PYTHONPATH " + " ".join(shlex.quote(part) for part in (interpreter, "-I", "-c", _UNINSTALL_SCRIPT, prefix))
        status, output = self._execute(command, sudo=not self._is_root, timeout=180)
        errors = {
            "AGENT_ACTIVE_TASKS": "该 Agent 仍有活动或待对账任务，请先结束任务并完成对账",
            "AGENT_INSTALL_PATH_CONFLICT": "已有 Agent 服务使用其他安装目录，已停止卸载",
            "AGENT_UNINSTALL_UNVERIFIED": "无法确认标准 Agent 服务和配置，已停止卸载，请先检查现有安装",
            "AGENT_UNINSTALL_UNSAFE_PATH": "Agent 配置含有不安全或共享目录，已停止卸载并保留现有文件",
            "AGENT_UNINSTALL_PROTECTED_PATH": "Agent 程序或状态目录与训练代码或输出重叠，已停止卸载",
            "AGENT_UNINSTALL_TRAINING_ENV_SHARED": "训练 Python 使用 Agent 程序环境，卸载会影响训练；请先更换独立训练环境",
            "AGENT_UNINSTALL_STATE_INVALID": "Agent 状态目录不是可验证的专属目录，已停止卸载并保留现有文件",
            "AGENT_UNINSTALL_SERVICE_FAILED": "Agent 系统服务操作失败，请检查远程服务状态后重试",
            "AGENT_UNINSTALL_VERIFY_FAILED": "Agent 卸载后的服务或配置检查未通过，请检查远程安装状态",
            "AGENT_UNINSTALL_FAILED": "Agent 卸载未完成，请检查远程安装状态后重试，训练文件已保留",
        }
        try:
            result = json.loads(output) if status == 0 else {}
            if not isinstance(result, dict):
                raise ValueError()
        except (ValueError, TypeError):
            raise RemoteError("AGENT_UNINSTALL_FAILED", errors["AGENT_UNINSTALL_FAILED"]) from None
        if result.get("error"):
            code = result["error"] if result["error"] in errors else "AGENT_UNINSTALL_FAILED"
            raise RemoteError(code, errors[code])
        if result.get("service_removed") is not True or not isinstance(result.get("removed_paths"), list):
            raise RemoteError("AGENT_UNINSTALL_FAILED", errors["AGENT_UNINSTALL_FAILED"])
        emit("UNINSTALLED", "Agent 已卸载，训练代码、数据、模型和输出均已保留")
        return {key: result.get(key) for key in ("backup_dir", "removed_paths", "service_removed")}

    def close(self):
        with self._lifecycle_lock:
            self._closed.set()
            client = self.client
            self.client = None
        if client is not None:
            client.close()


class ManagedTunnel:
    """One reverse loopback listener, with independent socket/channel bridges."""

    def __init__(self, host: str, port: int, username: str, password: str, state_dir: str | Path,
                 remote_port: int, target_host: str, target_port: int,
                 expected_fingerprint: str | None = None):
        if not isinstance(remote_port, int) or isinstance(remote_port, bool) or not 0 <= remote_port <= 65535:
            raise RemoteError("SSH_TUNNEL_PORT_INVALID", "隧道远程端口无效")
        self.remote = SSHRemote(host, port, username, password, state_dir, expected_fingerprint)
        self.remote_port, self.actual_port = remote_port, 0
        self.target_host, self.target_port = target_host, target_port
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bridges: set = set()
        self._lock = threading.Lock()

    @property
    def healthy(self) -> bool:
        client = self.remote.client
        transport = client.get_transport() if client else None
        return bool(not self._stop.is_set() and self.actual_port and transport and transport.is_active()
                    and self._thread and self._thread.is_alive())

    def start(self) -> int:
        if self.healthy:
            return self.actual_port
        self._stop.clear()
        self.remote.connect()
        try:
            transport = self.remote.client.get_transport()
            self.actual_port = transport.request_port_forward("127.0.0.1", self.remote_port)
            if not self.actual_port:
                raise RemoteError("SSH_TUNNEL_FAILED", "SSH 服务器未分配反向隧道端口")
            self._thread = threading.Thread(target=self._accept, name="vla-agent-tunnel", daemon=True)
            self._thread.start()
            return self.actual_port
        except RemoteError:
            self.close()
            raise
        except (OSError, paramiko.SSHException):
            self.close()
            raise RemoteError("SSH_TUNNEL_FAILED", "无法创建 SSH 反向隧道，请检查转发权限或端口占用") from None

    def _accept(self):
        client = self.remote.client
        transport = client.get_transport() if client else None
        while transport and transport.is_active() and not self._stop.is_set():
            try:
                channel = transport.accept(0.5)
            except (OSError, EOFError, paramiko.SSHException):
                break
            if channel is not None:
                threading.Thread(target=self._bridge, args=(channel,), name="vla-agent-bridge", daemon=True).start()

    def _bridge(self, channel):
        target = None
        with self._lock:
            self._bridges.add(channel)
        try:
            target = socket.create_connection((self.target_host, self.target_port), timeout=10)
            target.settimeout(10)
            with self._lock:
                self._bridges.add(target)
            while not self._stop.is_set():
                readable, _, _ = select.select([channel, target], [], [], 0.5)
                if channel in readable:
                    payload = channel.recv(65536)
                    if not payload:
                        break
                    target.sendall(payload)
                if target in readable:
                    payload = target.recv(65536)
                    if not payload:
                        break
                    channel.sendall(payload)
        except (OSError, EOFError, paramiko.SSHException):
            pass
        finally:
            channel.close()
            if target is not None:
                target.close()
            with self._lock:
                self._bridges.discard(channel)
                self._bridges.discard(target)

    def close(self):
        self._stop.set()
        client = self.remote.client
        transport = client.get_transport() if client else None
        if transport and transport.is_active() and self.actual_port:
            try:
                transport.cancel_port_forward("127.0.0.1", self.actual_port)
            except (OSError, EOFError, paramiko.SSHException):
                pass
        with self._lock:
            for bridge in self._bridges:
                try:
                    bridge.close()
                except OSError:
                    pass
        self.remote.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2)
        self.actual_port = 0
