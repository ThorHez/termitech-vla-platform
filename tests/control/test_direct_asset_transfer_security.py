"""Independent security boundaries; all peers and SSH homes are synthetic."""

import ast
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from vla_control import direct_asset_transfer as direct


def remote(**changes):
    raw = b"isolated pinned public server key"
    encoded = base64.b64encode(raw).decode()
    pin = "SHA256:" + base64.b64encode(hashlib.sha256(raw).digest()).decode().rstrip("=")
    key = SimpleNamespace(get_name=lambda: "ssh-ed25519", get_base64=lambda: encoded)
    transport = SimpleNamespace(is_active=lambda: True, get_remote_server_key=lambda: key)
    return SimpleNamespace(host="192.0.2.10", port=2222, username="fixture_user", fingerprint=pin,
                           expected_fingerprint=pin, client=SimpleNamespace(get_transport=lambda: transport),
                           **changes)


def test_direct_identity_uses_already_authenticated_pinned_host_key_only():
    source = remote()
    source._password = "must-not-cross-to-target"
    identity = direct._source_identity(source)
    assert set(identity) == {"host", "port", "username", "key_type", "key_base64"}
    assert identity["key_type"] == "ssh-ed25519" and identity["port"] == 2222
    assert "password" not in str(identity) and source._password not in str(identity)


@pytest.mark.parametrize("field,value", [
    ("host", "-oProxyCommand=evil"), ("host", "good;touch owned"), ("host", "user@host"),
    ("host", "host\nkey"), ("username", "root -oProxyCommand=evil"), ("username", "root@other"),
    ("username", "root;id"), ("port", True), ("port", 0), ("port", 65536), ("port", "22"),
])
def test_direct_identity_rejects_remote_shell_option_and_host_injection(field, value):
    source = remote()
    setattr(source, field, value)
    with pytest.raises(direct.TransferError) as error:
        direct._source_identity(source)
    assert error.value.code == "DIRECT_SSH_IDENTITY_INVALID"


@pytest.mark.parametrize("field", ["fingerprint", "expected_fingerprint"])
def test_direct_identity_rejects_any_pinned_key_mismatch(field):
    source = remote()
    setattr(source, field, "SHA256:unexpected")
    with pytest.raises(direct.TransferError) as error:
        direct._source_identity(source)
    assert error.value.code == "DIRECT_HOST_KEY_MISMATCH"


def test_direct_identity_rejects_absent_pin_and_inactive_transport():
    source = remote()
    source.fingerprint = source.expected_fingerprint = None
    with pytest.raises(direct.TransferError):
        direct._source_identity(source)
    source = remote()
    source.client.get_transport().is_active = lambda: False
    with pytest.raises(direct.TransferError):
        direct._source_identity(source)


@pytest.fixture
def authorized(tmp_path):
    # Execute definitions only: no SSH subprocess, pwd lookup, daemon or real
    # ~/.ssh access. Every path passed to these helpers is below tmp_path.
    namespace = {}
    exec(direct._REMOTE_COMMON + direct._AUTHORIZED_KEYS_HELPERS, namespace)
    home = tmp_path.resolve() / "isolated-home"
    home.mkdir()
    fd = namespace["safe_ssh_directory"](str(home))
    yield namespace, home, fd
    os.close(fd)


@pytest.mark.parametrize("tail", [b"", b"\n", b"\r\n"])
def test_temporary_authorization_restores_exact_original_bytes_and_mode(authorized, tail):
    functions, home, fd = authorized
    keys = home / ".ssh" / "authorized_keys"
    original = b"# existing operator key\nssh-ed25519 ORIGINAL no-touch" + tail
    keys.write_bytes(original)
    keys.chmod(0o640)
    line = 'restrict,expiry-time="20990101010101",command="fixed sender" ssh-ed25519 TEMP task-nonce'
    restoration = functions["install_authorized"](fd, line)
    installed = keys.read_bytes()
    assert installed.startswith(original) and installed.endswith(line.encode() + b"\n")
    assert stat.S_IMODE(keys.stat().st_mode) == 0o640
    functions["remove_authorized"](fd, line, restoration)
    assert keys.read_bytes() == original
    assert stat.S_IMODE(keys.stat().st_mode) == 0o640


def test_cleanup_removes_exact_line_and_preserves_other_task_keys_and_edits(authorized):
    functions, home, fd = authorized
    keys = home / ".ssh" / "authorized_keys"
    original = b"# original\nssh-ed25519 ORIGINAL operator\n"
    keys.write_bytes(original)
    keys.chmod(0o600)
    ours = 'restrict,command="fixed" ssh-ed25519 OURKEY termitech-direct-1'
    similar = 'restrict,command="different" ssh-ed25519 OTHERKEY termitech-direct-1-other'
    restore = functions["install_authorized"](fd, ours)
    functions["install_authorized"](fd, similar)
    with keys.open("ab") as stream:
        stream.write(b"# concurrent operator change\nssh-ed25519 MANUAL administrator\n")
    functions["remove_authorized"](fd, ours, restore)
    assert keys.read_bytes() == original + similar.encode() + b"\n# concurrent operator change\nssh-ed25519 MANUAL administrator\n"
    functions["remove_authorized"](fd, ours, restore)
    assert similar.encode() in keys.read_bytes()


def test_cooperating_authorization_writers_share_stable_lock_and_do_not_lose_keys(authorized):
    functions, home, fd = authorized
    keys = home / ".ssh" / "authorized_keys"
    original = b"# operator\nssh-ed25519 ORIGINAL keep\n"
    keys.write_bytes(original)
    keys.chmod(0o600)
    lines = [f'restrict,command="fixed" ssh-ed25519 KEY{i} task-{i}' for i in range(8)]
    barrier = threading.Barrier(len(lines))
    def install(line):
        barrier.wait()
        return functions["install_authorized"](fd, line)
    with ThreadPoolExecutor(max_workers=len(lines)) as workers:
        restorations = list(workers.map(install, lines))
    assert len(keys.read_bytes().splitlines()) == len(original.splitlines()) + len(lines)
    with ThreadPoolExecutor(max_workers=len(lines)) as workers:
        list(workers.map(lambda item: functions["remove_authorized"](fd, *item), zip(lines, restorations)))
    assert keys.read_bytes() == original
    assert (home / ".ssh" / ".termitech-direct.lock").is_file()


def test_external_change_detected_before_replace_is_not_overwritten(authorized):
    functions, home, fd = authorized
    keys = home / ".ssh" / "authorized_keys"
    keys.write_bytes(b"original\n")
    keys.chmod(0o600)
    _, before = functions["read_authorized"](fd)
    changed = b"original\noperator-added-key\n"
    keys.write_bytes(changed)
    with pytest.raises(ValueError, match="DIRECT_AUTH_CHANGED"):
        functions["replace_authorized"](fd, b"would lose operator change\n", before)
    assert keys.read_bytes() == changed
    assert not list(keys.parent.glob(".termitech-authorized-*"))


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "group-writable", "fifo"])
def test_authorized_keys_unsafe_objects_are_rejected_without_touching_target(authorized, kind):
    functions, home, fd = authorized
    keys = home / ".ssh" / "authorized_keys"
    unrelated = home / "unrelated"
    unrelated.write_bytes(b"private unrelated content")
    unrelated.chmod(0o600)
    if kind == "symlink":
        keys.symlink_to(unrelated)
    elif kind == "hardlink":
        os.link(unrelated, keys)
    elif kind == "fifo":
        os.mkfifo(keys, 0o600)
    else:
        keys.write_bytes(b"unsafe permissions")
        keys.chmod(0o660)
    with pytest.raises((ValueError, OSError)):
        functions["install_authorized"](fd, "must-not-be-written")
    assert unrelated.read_bytes() == b"private unrelated content"


def test_symlink_ssh_directory_and_symlink_ancestor_paths_are_rejected(tmp_path):
    namespace = {}
    exec(direct._REMOTE_COMMON + direct._AUTHORIZED_KEYS_HELPERS, namespace)
    root = tmp_path.resolve()
    home, other = root / "home", root / "other"
    home.mkdir()
    other.mkdir()
    (home / ".ssh").symlink_to(other, target_is_directory=True)
    with pytest.raises(OSError):
        namespace["safe_ssh_directory"](str(home))
    assert list(other.iterdir()) == []
    (root / "alias").symlink_to(other, target_is_directory=True)
    (other / "data").mkdir()
    with pytest.raises(OSError):
        namespace["path_fd"](str(root / "alias" / "data"))


def test_target_ssh_argv_disables_ambient_identity_proxy_and_host_trust(tmp_path):
    tree = ast.parse(direct._TARGET_SCRIPT)
    definitions = ast.Module(body=[node for node in tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name == "ssh_arguments"], type_ignores=[])
    assert len(definitions.body) == 1
    namespace = {}
    exec(compile(definitions, "isolated-direct-ssh-argv", "exec"), namespace)
    key, knownhosts = str(tmp_path / "ephemeral-key"), str(tmp_path / "pinned-host")
    arguments = namespace["ssh_arguments"](direct._source_identity(remote()), key, knownhosts, "isolated-alias")
    assert arguments[:3] == ["ssh", "-F", "/dev/null"]
    options = {arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == "-o"}
    assert {
        "HostKeyAlias=isolated-alias", "UserKnownHostsFile=" + knownhosts, "GlobalKnownHostsFile=/dev/null",
        "StrictHostKeyChecking=yes", "BatchMode=yes", "IdentitiesOnly=yes", "IdentityAgent=none",
        "PasswordAuthentication=no", "KbdInteractiveAuthentication=no", "PreferredAuthentications=publickey",
        "ControlMaster=no", "ControlPath=none", "ProxyCommand=none", "ProxyJump=none",
        "PermitLocalCommand=no", "ClearAllForwardings=yes", "RequestTTY=no",
    } <= options
    assert arguments[arguments.index("-i") + 1] == key
    assert "StrictHostKeyChecking=no" not in options and not any("accept-new" in value for value in options)


def watchdog_namespace(script):
    tree = ast.parse(script)
    definitions = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    body = ast.Module(body=[node for node in tree.body if node not in definitions], type_ignores=[])
    namespace = {}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "isolated-watchdog-definitions", "exec"), namespace)
    return namespace, compile(body, "isolated-watchdog-body", "exec")


@pytest.mark.parametrize("parent_alive,pid_matches", [(True, True), (False, True), (False, False)])
def test_target_key_ttl_cleanup_preserves_healthy_transfer_and_guards_pid_reuse(tmp_path, parent_alive, pid_matches):
    namespace, body = watchdog_namespace(direct._TARGET_WATCHDOG_SCRIPT)
    directory = tmp_path.resolve() / "key-material"
    directory.mkdir()
    (directory / "child.json").write_text(json.dumps({"pid": 12345, "identity": "original-start"}))
    spec = {"parent": 67890, "parent_identity": "parent-start", "seconds": 10,
            "temp": str(directory), "temp_identity": [directory.stat().st_dev, directory.stat().st_ino]}
    clock, removed, killed = [0.0], [], []
    namespace.update(
        sys=SimpleNamespace(argv=["watchdog", json.dumps(spec)]),
        time=SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
        alive=lambda *_: parent_alive,
        process_stamp=lambda *_: "original-start" if pid_matches else "reused-pid-start",
        remove_temp=lambda path, identity: removed.append((path, identity)),
        os=SimpleNamespace(path=os.path, killpg=lambda pid, sig: killed.append(pid)),
    )
    exec(body, namespace)
    assert removed == [(str(directory), spec["temp_identity"])]
    assert killed == ([12345] if not parent_alive and pid_matches else [])
    if parent_alive:
        assert clock[0] >= 10


def test_source_authorization_ttl_revokes_exact_key_without_killing_authenticated_sender():
    namespace, body = watchdog_namespace(direct._SOURCE_WATCHDOG_SCRIPT)
    spec = {"parent": 123, "parent_identity": "active-parent", "seconds": 10, "home": "/isolated/home",
            "line": "our exact scoped public key", "temp": "/isolated/temp", "temp_identity": [1, 2],
            "restore": {"length": 12, "sha256": "original-prefix", "separator": False}}
    clock, removed, keys, closed = [0.0], [], [], []
    namespace.update(
        sys=SimpleNamespace(argv=["watchdog", json.dumps(spec)]),
        time=SimpleNamespace(monotonic=lambda: clock[0], sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds)),
        alive=lambda *_: True, safe_ssh_directory=lambda _: 42,
        remove_authorized=lambda fd, line, restore: keys.append((fd, line, restore)),
        remove_temp=lambda path, identity: removed.append((path, identity)),
        os=SimpleNamespace(close=closed.append),
    )
    exec(body, namespace)
    assert clock[0] >= 10 and keys == [(42, spec["line"], spec["restore"])]
    assert closed == [42] and removed == [(spec["temp"], spec["temp_identity"])]


@pytest.mark.parametrize("expired", [False, True])
def test_forced_sender_ignores_original_command_and_numeric_expiry_blocks_new_sessions(tmp_path, expired):
    root = tmp_path.resolve()
    dataset = root / "dataset';touch NOT_ALLOWED;#"
    dataset.mkdir()
    (dataset / "payload").write_text("allowed source bytes")
    wrapper = root / "sender.py"
    wrapper.write_text(direct._FORCED_SENDER_SCRIPT)
    (root / "scope.json").write_text(json.dumps({"path": str(dataset), "kind": "directory",
                                               "deadline": time.time() + (-10 if expired else 600)}))
    # Intercept exec only in a separate Python process; the test never starts
    # sshd, a real SSH command, or a real sender and cannot touch real ~/.ssh.
    prefix = """import os,json,shutil
def capture(binary,arguments):
    print(json.dumps({'binary':binary,'arguments':arguments,'cwd':os.getcwd(),'payload':open('payload').read()}),flush=True)
    os._exit(0)
os.execv=capture
shutil.which=lambda _binary:'/isolated/rsync'
"""
    environment = dict(os.environ, SSH_ORIGINAL_COMMAND=f"touch {root / 'NOT_ALLOWED'}; cat /etc/passwd")
    command = prefix + "exec(" + repr(direct._FORCED_SENDER_SCRIPT) + ",{'__file__':" + repr(str(wrapper)) + "})"
    result = subprocess.run([sys.executable, "-c", command], env=environment,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
    assert not (root / "NOT_ALLOWED").exists()
    if expired:
        assert result.returncode == 126 and result.stdout == b""
    else:
        assert result.returncode == 0, result.stderr
        value = json.loads(result.stdout)
        assert value["arguments"] == ["/isolated/rsync", "--server", "--sender", "-rct", "--partial", ".", "./"]
        assert value["cwd"] == str(dataset) and value["payload"] == "allowed source bytes"


def test_target_prepared_key_is_private_and_eof_cleans_it_before_copy_starts(tmp_path):
    if not shutil.which("ssh-keygen") or not shutil.which("rsync"):
        pytest.skip("Isolated preparation needs ssh-keygen and rsync installed")
    from .test_direct_asset_transfer import direct_local_pair

    root = tmp_path.resolve()
    source, target = direct_local_pair(root / "peers")
    staging = root / "owned.partial"
    staging.mkdir()
    token = "isolated-security-" + os.urandom(12).hex()
    channel = None
    try:
        channel = direct._open_channel(target, shlex.join(["python3", "-u", "-c", direct._TARGET_SCRIPT]))
        direct._send(channel, {"token": token, "transfer_id": "isolated-key-setup", "source": direct._source_identity(source),
                              "staging": str(staging), "timeout": 30, "initial_bytes": 0, "bytes_total": 0, "files_total": 0})
        messages = direct._Messages(channel)
        ready, deadline = [], time.monotonic() + 5
        while not ready and time.monotonic() < deadline:
            ready.extend(messages.poll())
            time.sleep(.01)
        assert len(ready) == 1 and set(ready[0]) == {"type", "public_key"}
        assert ready[0]["type"] == "ready" and ready[0]["public_key"].startswith("ssh-ed25519 ")
        directories = list(Path("/tmp").glob(".termitech-direct-" + token + "-*"))
        assert len(directories) == 1
        directory = directories[0]
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700
        assert stat.S_IMODE((directory / "key").stat().st_mode) == 0o600
        assert stat.S_IMODE((directory / "known_hosts").stat().st_mode) == 0o600
        assert "PRIVATE KEY" not in json.dumps(ready)
        # Closing before the controller's start message exercises the crash
        # window after target keygen but before authorization/copy can begin.
        direct._finish(channel)
        assert not directory.exists() and not list(staging.iterdir())
        assert not (source.home / ".ssh").exists()
    finally:
        if channel is not None:
            channel.close()
        source.close()
        target.close()
