"""An independent Agent connection cannot manage the canonical systemd unit."""
import hashlib
import json
import shutil
import shlex
import subprocess
import sys
from unittest.mock import Mock

import pytest

from vla_control import ssh_remote as module
from vla_control.ssh_remote import RemoteError, SSHRemote

CUSTOM = "termitech-vla-agent-mac-test"
CANONICAL = "termitech-vla-agent"


def remote(tmp_path, **kwargs):
    return SSHRemote("192.0.2.7", 22, "root", "private-test-password", tmp_path, **kwargs)


@pytest.mark.parametrize("name", ["", None, 42, True, "x" * 101, "agent.service",
    "../agent", "agent other", "agent\nother", "agent;reboot", "$(reboot)", "代理"])
def test_invalid_service_names_fail_before_connection_or_mutation(tmp_path, name):
    with pytest.raises(RemoteError) as error:
        remote(tmp_path, service_name=name)
    assert error.value.code == "SSH_SERVICE_INVALID"
    assert not (tmp_path / "ssh_known_hosts").exists()


@pytest.mark.parametrize("name", [CANONICAL, CUSTOM, "Agent_test-42", "x" * 100])
def test_service_name_accepts_only_bounded_plain_unit_stem(tmp_path, name):
    assert remote(tmp_path, service_name=name).service_name == name


@pytest.mark.parametrize("name", [None, CUSTOM])
def test_probe_routes_only_the_selected_service_namespace(tmp_path, name):
    ssh = remote(tmp_path, **({"service_name": name} if name else {}))
    ssh.run = Mock(side_effect=["0\n", "venv\n/usr/bin/python3.11\n\n",
        json.dumps({"hostname": "test", "checks": [{"name": "python", "ok": True}]})])
    ssh.probe("/data2/test-agent")
    command = shlex.split(ssh.run.call_args.args[0])
    tail = command[command.index(module._PROBE_SCRIPT) + 1:]
    expected = ["/data2/test-agent", "venv", "", "root"]
    assert tail == expected + ([CUSTOM] if name else [])


@pytest.mark.parametrize("name", [None, CUSTOM])
def test_install_passes_selected_name_and_confirms_only_its_unit(tmp_path, name):
    selected = name or CANONICAL
    ssh = remote(tmp_path, **({"service_name": name} if name else {}))
    ssh._is_root = True
    ssh._runtime_selection = {"kind": "venv", "python": "/usr/bin/python3.11", "conda": ""}
    ssh.client = Mock()
    bundle = tmp_path / "bundle.tar.gz"
    bundle.write_bytes(b"test-bundle")
    temporary = "/tmp/vla-agent-install.ABC123"
    ssh.run = Mock(side_effect=[temporary, hashlib.sha256(bundle.read_bytes()).hexdigest() + " bundle", ""])
    ssh._execute = Mock(side_effect=[(0, ""), (0, "")])
    ssh.install(bundle, {}, "/data2/test-agent", Mock())
    install = shlex.split(ssh._execute.call_args_list[0].args[0])
    assert install[install.index("--service-name") + 1] == selected
    assert shlex.split(ssh._execute.call_args_list[1].args[0]) == [
        "systemctl", "is-active", "--quiet", selected + ".service"]
    if name:
        assert CANONICAL + ".service" not in ssh._execute.call_args_list[1].args[0]


def test_uninstall_wrapper_carries_custom_name_to_remote_script(tmp_path):
    ssh = remote(tmp_path, service_name=CUSTOM)
    ssh._is_root = True
    ssh.probe = Mock(return_value={"_maintenance_python": "/usr/bin/python3.11",
        "checks": [{"name": "privileges", "ok": True}]})
    ssh._execute = Mock(return_value=(0, json.dumps({"backup_dir": None,
        "removed_paths": [], "service_removed": True})))
    ssh.uninstall("/data2/test-agent", Mock())
    command = shlex.split(ssh._execute.call_args.args[0])
    assert command[command.index(module._UNINSTALL_SCRIPT) + 1:] == ["/data2/test-agent", CUSTOM]


def test_actual_custom_uninstall_retains_canonical_unit(tmp_path, monkeypatch):
    # Reuse the existing real-files/SQLite fixture; replace only its fake systemd
    # command layer so the two units have distinct identity and lifecycle state.
    from .test_ssh_remote import uninstall_script_fixture
    namespace, prefix, unit_dir, state, config_path, _ = uninstall_script_fixture(tmp_path, monkeypatch)
    canonical = unit_dir / (CANONICAL + ".service")
    canonical.write_text("production-unit-must-survive\n")
    custom = unit_dir / (CUSTOM + ".service")
    custom.write_text("[Service]\n")
    calls = []
    active = {"value": True}
    original_command = namespace["command"]

    def command(*args, required=True):
        if args[0] != "systemctl":
            return original_command(*args, required=required)
        calls.append(args)
        if args[1] not in {"daemon-reload"}:
            assert CUSTOM + ".service" in args
        status, output = 0, ""
        if args[1] == "show":
            if "--property=LoadState" in args:
                output = "loaded\n" if custom.exists() else "not-found\n"
            elif "--property=ExecStart" in args:
                output = f"{{ path={prefix}/.venv/bin/vla-agent ; argv[]={prefix}/.venv/bin/vla-agent --config {config_path} ; }}"
            elif "--property=FragmentPath" in args:
                output = str(custom)
        elif args[1] == "is-active":
            status = 0 if active["value"] else 3
        elif args[1] == "stop":
            active["value"] = False
        return subprocess.CompletedProcess(args, status, output, "")

    namespace["command"] = command
    result = namespace["uninstall_agent"](str(prefix), unit_dir, service_name=CUSTOM)
    assert result["service_removed"] is True
    assert not custom.exists()
    assert canonical.read_text() == "production-unit-must-survive\n"
    assert (prefix / "runs/result.txt").is_file()
    assert all(CANONICAL + ".service" not in args for args in calls)


def test_actual_custom_probe_rejects_prefix_owned_by_canonical_service(tmp_path, monkeypatch, capsys):
    prefix = tmp_path / "production-prefix"
    (prefix / "config").mkdir(parents=True)
    state = tmp_path / "production-state"
    state.mkdir()
    config = prefix / "config/agent.json"
    content = json.dumps({"state_dir": str(state), "profiles": []})
    config.write_text(content)
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if args[0] == "systemctl":
            assert args[1] == "show"  # Probe must never stop/start either unit.
            if args[2] == "*.service":
                output = (f"Id={CANONICAL}.service\nExecStart={{ path={prefix}/.venv/bin/vla-agent ; "
                          f"argv[]={prefix}/.venv/bin/vla-agent --config {config} ; }}\n")
            else:
                assert args[2] == CUSTOM + ".service"
                output = ""  # The new test unit does not exist yet.
        else:
            output = json.dumps({"node_id": "existing-node", "active_attempts": 0})
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(shutil, "which", lambda command: "/usr/bin/" + command)
    monkeypatch.setattr(sys, "argv", ["probe", str(prefix), "venv", "", "root", CUSTOM])
    exec(compile(module._PROBE_SCRIPT, "remote_probe", "exec"), {"__name__": "probe_test"})
    result = json.loads(capsys.readouterr().out)
    checks = {check["name"]: check["ok"] for check in result["checks"]}
    assert checks["service_ownership"] is False
    assert config.read_text() == content
    assert state.is_dir()
    assert any(args[:3] == ["systemctl", "show", "*.service"] for args in calls)
