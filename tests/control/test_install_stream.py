"""Exercise real SSH stream framing without opening a network connection."""

import hashlib
import json
from collections import deque
from unittest.mock import Mock

import pytest

from vla_control import ssh_remote as module
from vla_control.ssh_remote import RemoteError, SSHRemote, sanitize_install_output


class StreamingChannel:
    def __init__(self, stdout=(), stderr=(), *, status=0, continuous=False):
        self.stdout, self.stderr = deque(stdout), deque(stderr)
        self.status, self.continuous = status, continuous
        self.closed = self.exited = False
        self.on_exit = lambda: None

    def settimeout(self, value):
        pass

    def exec_command(self, command):
        self.command = command

    def sendall(self, value):
        self.stdin = value

    def shutdown_write(self):
        pass

    def recv_ready(self):
        return bool(self.stdout) or self.continuous

    def recv_stderr_ready(self):
        return bool(self.stderr)

    def recv(self, count):
        return self.stdout.popleft() if self.stdout else b"stream still running\n"

    def recv_stderr(self, count):
        return self.stderr.popleft()

    def exit_status_ready(self):
        return not self.stdout and not self.stderr and not self.continuous

    def recv_exit_status(self):
        self.on_exit()
        self.exited = True
        return self.status

    def close(self):
        self.closed = True


def remote(tmp_path, channels):
    ssh = SSHRemote("192.0.2.7", 22, "root", "ssh-private-password", tmp_path)
    ssh._is_root = True
    ssh.client = Mock()
    transport = ssh.client.get_transport.return_value
    transport.is_active.return_value = True
    transport.open_session.side_effect = channels
    return ssh


def test_streams_are_independent_incremental_and_delivered_before_exit(tmp_path):
    chinese = "正在安装".encode()
    channel = StreamingChannel([b"out\r", b"\n" + chinese[:2], chinese[2:] + b"\ninvalid:\xff", b"\nlast"],
                               [b"err", b"or\r\nlast error"])
    ssh = remote(tmp_path, [channel])
    lines = []

    def append(line):
        assert not channel.exited
        lines.append(line)

    channel.on_exit = lambda: lines.append("EXIT")
    status, _ = ssh._execute("installer", on_output=append)
    assert status == 0 and channel.closed
    assert lines == ["out", "正在安装", "invalid:\ufffd", "error", "last", "last error", "EXIT"]


def test_overlong_line_discards_all_fragments_before_exposure(tmp_path):
    secret = "private-secret-that-crosses-chunks"
    channel = StreamingChannel([b"a" * module._LINE_LIMIT + secret[:15].encode(), secret[15:].encode() + b"\nOK\n"])
    ssh = remote(tmp_path, [channel])
    lines = []
    ssh._execute("installer", on_output=lines.append)
    assert lines == ["[安装输出过长，已省略该行]", "OK"]
    assert not any(fragment in " ".join(lines) for fragment in (secret, secret[:15], secret[15:]))


def test_output_volume_is_bounded_without_blocking_exit(tmp_path, monkeypatch):
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    channel = StreamingChannel([b"x" * 1000 + b"\n"] * 3000)
    ssh = remote(tmp_path, [channel])
    lines = []
    status, captured = ssh._execute("installer", on_output=lines.append)
    assert status == 0 and channel.exited
    assert len(captured) == module._OUTPUT_LIMIT
    assert len(lines) < 2200
    assert "后续原始输出已省略" in lines[-1]


def test_continuous_output_cannot_starve_deadline(tmp_path, monkeypatch):
    channel = StreamingChannel(continuous=True)
    ssh = remote(tmp_path, [channel])
    clock = iter(n / 10 for n in range(100))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    lines = []
    with pytest.raises(RemoteError, match="安装可能仍在执行") as error:
        ssh._execute("installer", timeout=1, on_output=lines.append)
    assert error.value.code == "SSH_COMMAND_TIMEOUT" and channel.closed
    assert 0 < len(lines) < 8


def test_idle_callback_waits_without_claiming_progress(tmp_path, monkeypatch):
    channel = StreamingChannel()
    channel.exit_status_ready = Mock(side_effect=[False, False, True])
    ssh = remote(tmp_path, [channel])
    clock = iter([0, 16, 31])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    idle = Mock()
    ssh._execute("installer", timeout=60, on_idle=idle)
    assert idle.call_count == 2


@pytest.mark.parametrize("line,forbidden", [
    ("Downloading https://alice:topsecret@mirror.example/pkg.whl?credential=hidden&name=ok", ["alice", "topsecret", "hidden", "name=ok"]),
    ("Authorization: Bearer abc.def password='two words' enrollment_token=hidden api-key=xyz", ["abc.def", "two words", "hidden", "xyz"]),
    ("pip --password 'hidden words' --token private", ["hidden words", "private"]),
    ("\x1b[31mssh-private-password\x1b[0m token=private\x00\x1b]0;private-title\x07", ["ssh-private-password", "private-title", "\x1b", "\x00"]),
    ('{"name":"node", "profiles":[], "enrollment_token":"do-not-echo"}', ["node", "profiles", "do-not-echo"]),
    ("+ sudo -S sh -c 'echo deployment-command'", ["sudo", "deployment-command"]),
])
def test_sanitizer_hides_secrets_urls_config_and_commands(line, forbidden):
    safe = sanitize_install_output(line, ("ssh-private-password",))
    assert safe
    assert all(value not in safe for value in forbidden)


def test_sanitizer_redacts_before_truncating_and_bounds_line():
    line = "x" * 1990 + "user-password-that-crosses-limit" + "x" * 1000
    safe = sanitize_install_output(line, ("user-password-that-crosses-limit",))
    assert len(safe) <= 2000 and "user-password" not in safe


def test_literal_replacement_does_not_break_url_or_key_redaction():
    safe = sanitize_install_output("https://user:unknown@mirror.test/pkg?token=private password=secret-value", ("/", "word"))
    assert all(value not in safe for value in ("unknown", "private", "secret-value", "user:"))


def install_stream(tmp_path, stdout, stderr=(), status=0):
    channel = StreamingChannel(stdout, stderr, status=status)
    ssh = remote(tmp_path, [channel, StreamingChannel()])
    ssh._runtime_selection = {"kind": "venv", "python": "/usr/bin/python3.11", "conda": ""}
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle")
    temporary = "/tmp/vla-agent-install.ABC123"
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    ssh.run = Mock(side_effect=[temporary, checksum + " bundle", ""])
    return ssh, bundle, channel


def event(phase, message, **extra):
    return ("VLA_INSTALL_EVENT " + json.dumps({"phase": phase, "message": message, **extra}) + "\n").encode()


def test_installer_protocol_streams_sanitized_details_and_legacy_lines(tmp_path):
    ssh, bundle, channel = install_stream(tmp_path, [
        event("DEPENDENCIES", "安装 Python 依赖"), b"Collecting psutil\nssh-private-", b"password enrollment_",
        b"token=private-config-token\nhttps://u:p@mirror.test/wheel?q=secret\n",
        b"VLA_INSTALL_EVENT broken token=bad-secret\n",
        event("INVALID_PHASE", "must not change phase"),
        event("VERIFY", "import validation", level="warning"), event("STARTING", "starting service"),
    ], [b"pip stderr error: --token private-config-token\n"])
    events = []
    channel.on_exit = lambda: events.append(("EXIT", ""))
    ssh.install(bundle, {"enrollment_token": "private-config-token"}, "/data2/agent", lambda *args: events.append(args))
    encoded = json.dumps(events)
    for secret in ("ssh-private-password", "private-config-token", "bad-secret", "q=secret", "u:p@"):
        assert secret not in encoded
    assert ("INSTALL_DEPENDENCIES", "Collecting psutil") in events
    assert ("INSTALL_VERIFY", "[警告] import validation") in events
    assert ("STARTING", "starting service") in events
    assert events.index(("INSTALL_DEPENDENCIES", "Collecting psutil")) < events.index(("EXIT", ""))
    assert not any(stage == "INSTALL_INVALID_PHASE" for stage, _ in events)


def test_installer_failure_has_safe_exit_status_and_detailed_error(tmp_path):
    ssh, bundle, _ = install_stream(tmp_path, [b"ERROR: No matching distribution found\n"], status=7)
    events = []
    with pytest.raises(RemoteError, match="退出码 7") as error:
        ssh.install(bundle, {"enrollment_token": "private"}, "/data2/agent", lambda *args: events.append(args))
    assert error.value.code == "AGENT_INSTALL_FAILED"
    assert ("installing", "ERROR: No matching distribution found") in events
