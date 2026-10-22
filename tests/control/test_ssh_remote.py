import hashlib
import json
import os
import pathlib
import shlex
import sqlite3
import subprocess
import sys
from unittest.mock import Mock

import paramiko
import pytest

from vla_control import ssh_remote as module
from vla_control.ssh_remote import ManagedTunnel, RemoteError, SSHRemote


def remote(tmp_path, **kwargs):
    return SSHRemote("192.0.2.7", 22, "operator", "private-password", tmp_path, **kwargs)


@pytest.mark.parametrize("field,value", [("host", "host\nsecret"), ("host", "$(whoami)"), ("port", 0),
                                          ("port", True), ("username", "a\nb"), ("password", "a\nb")])
def test_reject_invalid_connection_inputs(tmp_path, field, value):
    values = dict(host="host", port=22, username="root", password="secret", state_dir=tmp_path)
    values[field] = value
    with pytest.raises(RemoteError):
        SSHRemote(**values)


def test_connect_disables_user_keys_and_pins_application_host(tmp_path, monkeypatch):
    client = Mock()
    key = paramiko.RSAKey.generate(1024)
    transport = client.get_transport.return_value
    transport.get_remote_server_key.return_value = key
    transport.is_active.return_value = True
    monkeypatch.setattr(paramiko, "SSHClient", lambda: client)
    ssh = remote(tmp_path).connect()
    options = client.connect.call_args.kwargs
    assert options["allow_agent"] is False and options["look_for_keys"] is False
    assert options["password"] == "private-password"
    assert ssh.fingerprint == module._fingerprint(key)
    assert not client.load_system_host_keys.called
    assert isinstance(client.set_missing_host_key_policy.call_args.args[0], module._HostPolicy)
    ssh.close()
    client.close.assert_called_once()


@pytest.mark.parametrize("failure,code", [(paramiko.AuthenticationException("private-password"), "SSH_AUTH_FAILED"),
                                         (OSError("private-password"), "SSH_CONNECTION_FAILED")])
def test_connection_errors_never_include_credentials(tmp_path, monkeypatch, failure, code):
    client = Mock()
    client.connect.side_effect = failure
    monkeypatch.setattr(paramiko, "SSHClient", lambda: client)
    with pytest.raises(RemoteError) as error:
        remote(tmp_path).connect()
    assert error.value.code == code
    assert "private-password" not in str(error.value)
    client.close.assert_called_once()


def test_host_keys_saved_private_and_changes_rejected(tmp_path):
    path = tmp_path / "ssh_known_hosts"
    key = paramiko.RSAKey.generate(1024)
    policy = module._HostPolicy(path, None)
    policy.missing_host_key(Mock(), "[host]:2222", key)
    assert path.stat().st_mode & 0o777 == 0o600
    assert paramiko.HostKeys(str(path)).check("[host]:2222", key)
    other = paramiko.RSAKey.generate(1024)
    with pytest.raises(RemoteError, match="主机密钥已变化"):
        policy.missing_host_key(Mock(), "[host]:2222", other)
    assert paramiko.HostKeys(str(path)).check("[host]:2222", key)


def test_unexpected_new_fingerprint_is_not_saved(tmp_path):
    key = paramiko.RSAKey.generate(1024)
    path = tmp_path / "ssh_known_hosts"
    with pytest.raises(RemoteError) as error:
        module._HostPolicy(path, "SHA256:different").missing_host_key(Mock(), "host", key)
    assert error.value.code == "SSH_HOST_KEY_MISMATCH"
    assert not path.exists()


def test_close_during_handshake_cannot_leave_a_connected_client(tmp_path, monkeypatch):
    ssh = remote(tmp_path)
    client = Mock()
    key = paramiko.RSAKey.generate(1024)
    client.get_transport.return_value.get_remote_server_key.return_value = key
    client.get_transport.return_value.is_active.return_value = True
    client.connect.side_effect = lambda *args, **kwargs: ssh.close()
    monkeypatch.setattr(paramiko, "SSHClient", lambda: client)
    with pytest.raises(RemoteError) as error:
        ssh.connect()
    assert error.value.code == "SSH_CANCELLED"
    assert ssh.client is None
    assert client.close.called


class Channel:
    def __init__(self, output=b"ok", stderr=b"", status=0, done=True):
        self.output, self.stderr, self.status, self.done = output, stderr, status, done
        self.closed = False
        self.command = self.stdin = None

    def settimeout(self, value):
        pass

    def exec_command(self, value):
        self.command = value

    def sendall(self, value):
        self.stdin = value

    def shutdown_write(self):
        pass

    def recv_ready(self):
        return bool(self.output)

    def recv_stderr_ready(self):
        return bool(self.stderr)

    def recv(self, count):
        output, self.output = self.output, b""
        return output

    def recv_stderr(self, count):
        output, self.stderr = self.stderr, b""
        return output

    def exit_status_ready(self):
        return self.done

    def recv_exit_status(self):
        return self.status

    def close(self):
        self.closed = True


def attach(ssh, channel):
    ssh.client = Mock()
    transport = ssh.client.get_transport.return_value
    transport.is_active.return_value = True
    transport.open_session.return_value = channel
    return transport


def test_sudo_password_uses_stdin_and_both_streams_are_drained(tmp_path):
    ssh = remote(tmp_path)
    channel = Channel(output=b"stdout", stderr=b"stderr")
    attach(ssh, channel)
    output = ssh.run("printf 'hello'", sudo=True)
    assert "private-password" not in channel.command
    assert channel.stdin == b"private-password\n"
    assert channel.command.startswith("sudo -S -k -p '' -- sh -c ")
    assert output == "stdoutstderr"
    assert channel.closed


def test_timeout_closes_channel_and_does_not_publish_output(tmp_path, monkeypatch):
    ssh = remote(tmp_path)
    channel = Channel(output=b"secret-enrollment-token", done=False)
    attach(ssh, channel)
    clock = iter([0, 2])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    with pytest.raises(RemoteError) as error:
        ssh.run("command", timeout=1)
    assert error.value.code == "SSH_COMMAND_TIMEOUT"
    assert "secret-enrollment-token" not in str(error.value)
    assert channel.closed


def test_failed_run_does_not_publish_remote_output(tmp_path):
    ssh = remote(tmp_path)
    attach(ssh, Channel(output=b"password and token", status=1))
    with pytest.raises(RemoteError) as error:
        ssh.run("command")
    assert error.value.code == "SSH_COMMAND_FAILED"
    assert "password" not in str(error.value)


def test_probe_sudo_and_missing_python(tmp_path):
    ssh = remote(tmp_path)
    ssh.run = Mock(side_effect=["1000\n", "0\n", "", ""])
    result = ssh.probe("/data2/agent")
    assert result["can_install"] is False
    assert result["checks"][0]["ok"] is True
    assert ssh.run.call_args_list[1].kwargs == {"sudo": True}


def test_probe_keeps_discovered_conda_paths_when_using_sudo(tmp_path):
    ssh = remote(tmp_path)
    ssh.run = Mock(side_effect=["1000\n", "0\n",
        "conda\n/home/operator/miniconda3/bin/python\n/home/operator/miniconda3/bin/conda\n",
        json.dumps({"hostname": "gpu", "checks": [{"name": "python", "ok": True}]})])
    result = ssh.probe("/data2/agent")
    assert result["can_install"] is True
    command = shlex.split(ssh.run.call_args.args[0])
    assert command[:6] == ["env", "-u", "PYTHONHOME", "-u", "PYTHONPATH", "/home/operator/miniconda3/bin/python"]
    assert command[-4:] == ["/data2/agent", "conda", "/home/operator/miniconda3/bin/conda", "operator"]
    assert ssh.run.call_args.kwargs == {"sudo": True}


def test_probe_surfaces_runtime_selection_failure_without_running_old_python(tmp_path):
    ssh = remote(tmp_path)
    ssh.run = Mock(side_effect=["0\n", "ERROR\nSSH 默认 Python：/usr/bin/python3（3.10.12）；需要 Python 3.11+\n", ""])
    result = ssh.probe("/data2/agent")
    assert not result["can_install"]
    assert "/usr/bin/python3（3.10.12）" in result["checks"][-1]["message"]
    assert ssh.run.call_count == 3


def test_existing_agent_identity_and_uninstall_survive_missing_bootstrap_conda(tmp_path):
    ssh = remote(tmp_path)
    probe = {"hostname": "gpu", "checks": [{"name": "idle", "ok": True}],
             "existing_agent": {"node_id": "node-original"}}
    ssh.run = Mock(side_effect=["0\n", "ERROR\n已发现 Conda，但不可用\n",
                               "/data2/agent/.venv/bin/python\n", json.dumps(probe)])
    result = ssh.probe("/data2/agent")
    assert result["existing_agent"]["node_id"] == "node-original"
    assert not result["can_install"]
    assert result["_maintenance_python"] == "/data2/agent/.venv/bin/python"
    assert result["checks"][1]["ok"] is False
    ssh.probe = Mock(return_value=result)
    ssh.run.reset_mock()
    ssh._execute = Mock(return_value=(0, json.dumps({"backup_dir": "/data2/backup",
                           "removed_paths": [], "service_removed": True})))
    ssh.uninstall("/data2/agent", Mock())
    ssh.run.assert_not_called()
    assert shlex.split(ssh._execute.call_args.args[0])[5] == "/data2/agent/.venv/bin/python"


@pytest.mark.parametrize("output", [
    "conda\nrelative/python\n/opt/conda/bin/conda\n",
    "conda\n/opt/conda/bin/python\n/opt/conda/bin/conda\nextra\n",
    "venv\n/usr/bin/python3\n/unexpected/conda\n",
    "venv\n/usr/bin/py\x00thon3\n\n",
])
def test_runtime_selection_rejects_malformed_paths(tmp_path, output):
    ssh = remote(tmp_path)
    ssh.run = Mock(return_value=output)
    with pytest.raises(RemoteError):
        ssh._select_runtime()


def test_conda_probe_can_bootstrap_from_python310(tmp_path):
    conda = tmp_path / "conda"
    conda.write_text("#!/bin/sh\n")
    conda.chmod(0o755)
    script = "import sys; sys.version_info=(3,10,12,'final',0)\n" + module._PROBE_SCRIPT
    result = json.loads(subprocess.check_output(
        [sys.executable, "-c", script, str(tmp_path / "agent"), "conda", str(conda)], text=True))
    checks = {item["name"]: item for item in result["checks"]}
    assert checks["python"]["ok"] is True
    assert checks["agent_runtime"]["ok"] is True
    assert "Python 3.11" in checks["agent_runtime"]["message"]


@pytest.mark.parametrize("path", ["/", "/data2", "/data2/../agent", "/data2/a;touch-hack", "/data2/a\n", "agent"])
def test_probe_rejects_unsafe_install_paths_before_remote_command(tmp_path, path):
    ssh = remote(tmp_path)
    ssh.run = Mock()
    with pytest.raises(RemoteError):
        ssh.probe(path)
    ssh.run.assert_not_called()


def test_probe_reads_only_whitelisted_config_and_node_identity(tmp_path):
    prefix = tmp_path / "agent"
    (prefix / "config").mkdir(parents=True)
    state = tmp_path / "state"
    state.mkdir()
    config = {"name": "existing", "roots": [str(tmp_path)], "output_root": str(tmp_path / "runs"),
              "state_dir": str(state), "enrollment_token": "private-enrollment-token", "control_plane_url": "https://example.com",
              "profiles": [{"id": "smoke", "python": sys.executable, "code_dir": str(tmp_path), "recipe": "test",
                            "private": "private-profile-secret", "metadata": {"validated": False, "token": "private-metadata-secret"}}]}
    (prefix / "config/agent.json").write_text(json.dumps(config))
    with sqlite3.connect(state / "agent.sqlite3") as database:
        database.executescript("CREATE TABLE credentials(id INTEGER,value TEXT);CREATE TABLE attempts(state TEXT);")
        database.execute("INSERT INTO credentials VALUES(1,?)", (json.dumps({"node_id": "node-123", "token": "private-node-token"}),))
        database.execute("INSERT INTO attempts VALUES('SUCCEEDED')")
    output = subprocess.check_output([sys.executable, "-c", module._PROBE_SCRIPT, str(prefix)], text=True)
    result = json.loads(output)
    existing = result["existing_agent"]
    assert existing["node_id"] == "node-123" and existing["active_attempts"] == 0
    assert existing["profiles"][0]["metadata"] == {"validated": False}
    for forbidden in ("private-enrollment-token", "private-node-token", "private-metadata-secret", "private-profile-secret"):
        assert forbidden not in output


def test_probe_detects_live_installer_only_for_exact_normalized_prefix(tmp_path):
    namespace = {"pathlib": pathlib, "os": os}
    exec(compile(module._INSTALLATION_IDLE_SCRIPT, "installation_idle", "exec"), namespace)
    process = tmp_path / "123"
    process.mkdir()
    command = ["/usr/bin/python3.11", "/tmp/stage/install_agent.py", "--prefix", "/data2/agent/./"]
    (process / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in command) + b"\0")
    assert namespace["installation_idle"]("/data2/agent", tmp_path) is False
    assert namespace["installation_idle"]("/data2/agent_test", tmp_path) is True


def test_probe_ignores_shell_text_unrelated_processes_and_disappearing_pids(tmp_path):
    namespace = {"pathlib": pathlib, "os": os}
    exec(compile(module._INSTALLATION_IDLE_SCRIPT, "installation_idle", "exec"), namespace)
    for pid, args in {
        "123": ["bash", "-c", "python /tmp/install_agent.py --prefix /data2/agent"],
        "124": ["/tmp/not_install_agent.py", "--prefix", "/data2/agent"],
        "125": ["/tmp/install_agent.py", "--prefix-other", "/data2/agent"],
    }.items():
        process = tmp_path / pid
        process.mkdir()
        (process / "cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in args) + b"\0")
    (tmp_path / "126").mkdir()  # /proc entry vanished before cmdline was read.
    assert namespace["installation_idle"]("/data2/agent", tmp_path) is True


def install_fixture(tmp_path):
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"bundle-data")
    ssh = remote(tmp_path)
    ssh._is_root = True
    ssh._runtime_selection = {"kind": "venv", "python": "/usr/bin/python3.11", "conda": ""}
    ssh.client = Mock()
    temporary = "/tmp/vla-agent-install.ABC123"
    checksum = hashlib.sha256(bundle.read_bytes()).hexdigest()
    ssh.run = Mock(side_effect=[temporary, checksum + "  " + temporary + "/vla-agent-bundle.tar.gz", ""])
    ssh._execute = Mock(side_effect=[(0, "private-token output"), (0, "")])
    return ssh, bundle, temporary


def test_install_checksums_stages_and_cleans_up_without_exposing_config(tmp_path):
    ssh, bundle, temporary = install_fixture(tmp_path)
    events = []
    ssh.install(bundle, {"enrollment_token": "private-token", "name": "test"}, "/data2/agent", lambda *args: events.append(args))
    assert [event[0] for event in events] == ["uploading", "verifying", "installing", "starting", "started"]
    assert "private-token" not in json.dumps(events)
    sftp = ssh.client.open_sftp.return_value
    sftp.chmod.assert_any_call(temporary + "/agent.json", 0o600)
    assert "--prefix /data2/agent" in ssh._execute.call_args_list[0].args[0]
    assert "--service-user operator" in ssh._execute.call_args_list[0].args[0]
    assert ssh._execute.call_args_list[0].kwargs["timeout"] == 900
    assert "private-token" not in ssh._execute.call_args_list[0].args[0]
    assert ssh.run.call_args.args[0] == "rm -rf -- " + temporary


def test_install_uses_same_selected_conda_python_across_sudo(tmp_path):
    ssh, bundle, temporary = install_fixture(tmp_path)
    ssh._is_root = False
    ssh._runtime_selection = {"kind": "conda", "python": "/home/operator/conda base/bin/python",
                              "conda": "/home/operator/conda base/bin/conda"}
    ssh.install(bundle, {}, "/data2/agent", Mock())
    call = ssh._execute.call_args_list[0]
    command = shlex.split(call.args[0])
    assert "/home/operator/conda base/bin/python" in command
    assert command[command.index("--runtime-kind") + 1] == "conda"
    assert command[command.index("--conda-executable") + 1] == "/home/operator/conda base/bin/conda"
    assert temporary + "/install_agent.py" in command
    assert call.kwargs["sudo"] is True


def test_install_checksum_failure_never_runs_installer_and_still_cleans(tmp_path):
    ssh, bundle, temporary = install_fixture(tmp_path)
    ssh.run.side_effect = [temporary, "wrong-checksum  bundle", ""]
    with pytest.raises(RemoteError) as error:
        ssh.install(bundle, {"enrollment_token": "private-token"}, "/data2/agent", Mock())
    assert error.value.code == "AGENT_BUNDLE_CHECKSUM_FAILED"
    ssh._execute.assert_not_called()
    assert ssh.run.call_args.args[0] == "rm -rf -- " + temporary


def test_install_guard_failure_is_safe_and_cleans_private_files(tmp_path):
    ssh, bundle, temporary = install_fixture(tmp_path)
    ssh._execute.side_effect = [(1, "secret-token 仍有活动或待对账任务")]
    with pytest.raises(RemoteError) as error:
        ssh.install(bundle, {"enrollment_token": "secret-token"}, "/data2/agent", Mock())
    assert error.value.code == "AGENT_ACTIVE_TASKS"
    assert "secret-token" not in str(error.value)
    assert ssh.run.call_args.args[0] == "rm -rf -- " + temporary


def uninstall_script_fixture(tmp_path, monkeypatch, *, state_outside=False):
    """Exercise the actual remote script against files and SQLite, fake systemd."""
    prefix = tmp_path / "agent"
    unit_dir = tmp_path / "systemd"
    unit_dir.mkdir()
    unit = unit_dir / "termitech-vla-agent.service"
    unit.write_text("[Service]\n")
    for component in ("config", ".venv/bin", "home", "tmp", "runs", "smoke-code"):
        (prefix / component).mkdir(parents=True, exist_ok=True)
    (prefix / ".venv/bin/vla-agent").write_text("agent-program")
    (prefix / "runs/result.txt").write_text("training-output")
    (prefix / "smoke-code/train.py").write_text("training-code")
    state = tmp_path / "external-state" if state_outside else prefix / "state"
    state.mkdir()
    (state / "instance-id").write_text("old-instance")
    with sqlite3.connect(state / "agent.sqlite3") as db:
        db.executescript("CREATE TABLE credentials(id INTEGER,value TEXT);CREATE TABLE attempts(state TEXT);CREATE TABLE events(type TEXT);")
        db.execute("INSERT INTO credentials VALUES(1,?)", (json.dumps({"node_id": "old-node", "token": "private-token"}),))
        db.execute("INSERT INTO attempts VALUES('SUCCEEDED')")
    config = {"state_dir": str(state), "output_root": str(prefix / "runs"),
              "profiles": [{"adapter": "smoke", "python": str(prefix / ".venv/bin/python"),
                            "code_dir": str(prefix / "smoke-code")}], "enrollment_token": "private-token"}
    config_path = prefix / "config/agent.json"
    config_path.write_text(json.dumps(config))
    (prefix / "config/agent.before-test.json").write_text(json.dumps(config))
    original_run = subprocess.run
    system = {"active": True, "enabled": True, "startup": None, "after_stop": None, "calls": []}

    def run(args, **kwargs):
        if args[0] != "systemctl":
            return original_run(args, **kwargs)
        system["calls"].append(args)
        status, output = 0, ""
        if args[1] == "show":
            if "--property=LoadState" in args:
                output = "loaded\n" if unit.exists() else "not-found\n"
            elif "--property=ExecStart" in args:
                output = system["startup"] or f"{{ path={prefix}/.venv/bin/vla-agent ; argv[]={prefix}/.venv/bin/vla-agent --config {config_path} ; }}"
            elif "--property=FragmentPath" in args:
                output = str(unit)
        elif args[1] == "stop":
            system["active"] = False
            if system["after_stop"]:
                system["after_stop"]()
        elif args[1] == "start":
            system["active"] = True
        elif args[1] == "disable":
            system["enabled"] = False
        elif args[1] == "enable":
            system["enabled"] = True
        elif args[1] == "is-active":
            status = 0 if system["active"] else 3
        elif args[1] == "is-enabled":
            status = 0 if system["enabled"] else 1
        return subprocess.CompletedProcess(args, status, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    namespace = {"__name__": "uninstall_test"}
    exec(compile(module._UNINSTALL_SCRIPT, "remote_uninstall", "exec"), namespace)
    return namespace, prefix, unit_dir, state, config_path, system


@pytest.mark.parametrize("state_outside", [False, True])
def test_uninstall_moves_identity_and_program_but_retains_training_files(tmp_path, monkeypatch, state_outside):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch, state_outside=state_outside)
    result = namespace["uninstall_agent"](str(prefix), unit_dir)
    backup = module.Path(result["backup_dir"])
    assert backup.stat().st_mode & 0o777 == 0o700
    assert result["service_removed"] is True
    assert not config_path.exists() and not state.exists()
    assert not (prefix / ".venv").exists()
    assert (backup / "config/agent.json").is_file()
    assert (backup / "config/agent.before-test.json").is_file()
    assert (backup / "state/instance-id").read_text() == "old-instance"
    with sqlite3.connect(backup / "state/agent.sqlite3") as database:
        assert json.loads(database.execute("SELECT value FROM credentials").fetchone()[0])["node_id"] == "old-node"
    assert (prefix / "runs/result.txt").read_text() == "training-output"
    assert (prefix / "smoke-code/train.py").read_text() == "training-code"
    assert not (unit_dir / "termitech-vla-agent.service").exists()
    assert not system["active"] and not system["enabled"]
    assert namespace["uninstall_agent"](str(prefix), unit_dir) == {"backup_dir": None, "removed_paths": [], "service_removed": True}


@pytest.mark.parametrize("state_value,events", [("RUNNING", False), ("UNKNOWN", False), ("SUCCEEDED", True)])
def test_uninstall_refuses_active_unreconciled_and_pending_events(tmp_path, monkeypatch, state_value, events):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)
    with sqlite3.connect(state / "agent.sqlite3") as db:
        db.execute("UPDATE attempts SET state=?", (state_value,))
        if events:
            db.execute("INSERT INTO events VALUES('SUCCEEDED')")
    with pytest.raises(namespace["UninstallError"], match="AGENT_ACTIVE_TASKS"):
        namespace["uninstall_agent"](str(prefix), unit_dir)
    assert config_path.exists() and system["active"]
    assert not any(args[1] == "stop" for args in system["calls"])


def test_uninstall_rechecks_after_stop_and_restores_service_on_race(tmp_path, monkeypatch):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)

    def accept_final_command():
        with sqlite3.connect(state / "agent.sqlite3") as db:
            db.execute("UPDATE attempts SET state='STARTING'")

    system["after_stop"] = accept_final_command
    with pytest.raises(namespace["UninstallError"], match="AGENT_ACTIVE_TASKS"):
        namespace["uninstall_agent"](str(prefix), unit_dir)
    assert system["active"] and system["enabled"] and config_path.exists()


def test_uninstall_refuses_different_service_prefix(tmp_path, monkeypatch):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)
    system["startup"] = f"{{ path={prefix}-other/.venv/bin/vla-agent ; argv[]={prefix}-other/.venv/bin/vla-agent --config {config_path} ; }}"
    with pytest.raises(namespace["UninstallError"], match="AGENT_INSTALL_PATH_CONFLICT"):
        namespace["uninstall_agent"](str(prefix), unit_dir)
    assert system["active"] and config_path.exists()


@pytest.mark.parametrize("kind", ["state-output", "state-code", "shared-python", "unknown-state-file", "state-symlink", "writable-config"])
def test_uninstall_refuses_shared_or_unverified_directories(tmp_path, monkeypatch, kind):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)
    config = json.loads(config_path.read_text())
    if kind == "state-output":
        config["output_root"] = str(state)
    elif kind == "state-code":
        config["profiles"][0]["code_dir"] = str(state)
    elif kind == "shared-python":
        config["profiles"][0]["adapter"] = "openpi"
    elif kind == "unknown-state-file":
        (state / "important-data.txt").write_text("preserve")
    elif kind == "state-symlink":
        (state / "external").symlink_to(prefix / "runs")
    config_path.write_text(json.dumps(config))
    if kind == "writable-config":
        config_path.chmod(0o666)
    with pytest.raises(namespace["UninstallError"]):
        namespace["uninstall_agent"](str(prefix), unit_dir)
    assert config_path.exists() and system["active"]
    assert (prefix / "runs/result.txt").is_file()


def test_uninstall_no_unit_does_not_move_unverified_config(tmp_path, monkeypatch):
    namespace, prefix, unit_dir, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)
    (unit_dir / "termitech-vla-agent.service").unlink()
    with pytest.raises(namespace["UninstallError"], match="AGENT_UNINSTALL_UNVERIFIED"):
        namespace["uninstall_agent"](str(prefix), unit_dir)
    assert config_path.exists() and state.exists()


def test_uninstall_wrapper_uses_sudo_and_fixed_public_messages(tmp_path):
    ssh = remote(tmp_path)
    ssh._is_root = False
    ssh.probe = Mock(return_value={"checks": [{"name": "privileges", "ok": True}]})
    ssh._runtime_selection = {"kind": "venv", "python": "/usr/bin/python3.11", "conda": ""}
    result = {"backup_dir": "/data2/agent.uninstalled.123", "removed_paths": ["/data2/agent/config"], "service_removed": True}
    ssh._execute = Mock(return_value=(0, json.dumps(result)))
    events = []
    assert ssh.uninstall("/data2/agent", lambda *args: events.append(args)) == result
    assert [event[0] for event in events] == ["CHECKING", "UNINSTALLING", "UNINSTALLED"]
    assert ssh._execute.call_args.kwargs == {"sudo": True, "timeout": 180}
    assert "private-password" not in ssh._execute.call_args.args[0]


def test_uninstall_wrapper_refuses_probe_active_before_mutation(tmp_path):
    ssh = remote(tmp_path)
    ssh.probe = Mock(return_value={"checks": [{"name": "privileges", "ok": True}, {"name": "idle", "ok": False}]})
    ssh._execute = Mock()
    with pytest.raises(RemoteError) as error:
        ssh.uninstall("/data2/agent", Mock())
    assert error.value.code == "AGENT_ACTIVE_TASKS"
    ssh._execute.assert_not_called()


def test_uninstall_wrapper_does_not_expose_remote_errors_or_output(tmp_path):
    ssh = remote(tmp_path)
    ssh.probe = Mock(return_value={"checks": [{"name": "privileges", "ok": True}]})
    ssh._runtime_selection = {"kind": "venv", "python": "/usr/bin/python3.11", "conda": ""}
    ssh._execute = Mock(return_value=(0, json.dumps({"error": "private-password-token"})))
    with pytest.raises(RemoteError) as error:
        ssh.uninstall("/data2/agent", Mock())
    assert error.value.code == "AGENT_UNINSTALL_FAILED"
    assert "private-password-token" not in str(error.value)


def test_tunnel_requests_new_loopback_port_and_cancels_own_listener(tmp_path, monkeypatch):
    tunnel = ManagedTunnel("host", 22, "root", "secret", tmp_path, 0, "127.0.0.1", 8000)
    tunnel.remote = Mock()
    transport = tunnel.remote.client.get_transport.return_value
    transport.is_active.return_value = True
    transport.request_port_forward.return_value = 42123
    thread = Mock()
    thread.is_alive.return_value = True
    monkeypatch.setattr(module.threading, "Thread", lambda **kwargs: thread)
    assert tunnel.start() == 42123
    transport.request_port_forward.assert_called_once_with("127.0.0.1", 0)
    assert tunnel.healthy
    tunnel.close()
    transport.cancel_port_forward.assert_called_once_with("127.0.0.1", 42123)
    assert not tunnel.healthy
    tunnel.remote.close.assert_called_once()


def test_tunnel_failure_does_not_cancel_or_take_existing_port(tmp_path):
    tunnel = ManagedTunnel("host", 22, "root", "secret", tmp_path, 18000, "127.0.0.1", 8000)
    tunnel.remote = Mock()
    transport = tunnel.remote.client.get_transport.return_value
    transport.is_active.return_value = True
    transport.request_port_forward.side_effect = paramiko.SSHException("secret-token")
    with pytest.raises(RemoteError) as error:
        tunnel.start()
    assert error.value.code == "SSH_TUNNEL_FAILED"
    assert "secret-token" not in str(error.value)
    transport.cancel_port_forward.assert_not_called()
