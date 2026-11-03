"""Shared GPU-lock mounts must never enter an Agent uninstall backup."""
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from vla_control.ssh_remote import RemoteError, SSHRemote

from .test_ssh_remote import uninstall_script_fixture


def mount_line(path):
    encoded = str(path).replace("\\", r"\134").replace(" ", r"\040").replace("\t", r"\011").replace("\n", r"\012")
    return f"81 42 0:33 / {encoded} rw,relatime - tmpfs tmpfs rw\n"


def test_nested_shared_lock_mount_rejects_before_stop_or_file_move(tmp_path, monkeypatch):
    namespace, prefix, units, state, config_path, system = uninstall_script_fixture(tmp_path, monkeypatch)
    locks = state / "locks" / "nested"
    locks.mkdir(parents=True)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(mount_line(locks))
    namespace["MOUNTINFO_PATH"] = mountinfo
    monkeypatch.setattr(namespace["os"].path, "ismount", lambda path: False)
    before = config_path.read_bytes()
    with pytest.raises(namespace["UninstallError"], match="AGENT_UNINSTALL_SHARED_MOUNT"):
        namespace["uninstall_agent"](str(prefix), units)
    assert system["active"] and system["enabled"]
    assert all(call[1] == "show" for call in system["calls"])
    assert config_path.read_bytes() == before and (state / "agent.sqlite3").is_file()
    assert not list(tmp_path.glob("agent.uninstalled.*"))


@pytest.mark.parametrize("name", ["space directory", r"back\slash", "tab\tname", "line\nname"])
def test_mountinfo_octal_paths_are_decoded_before_descendant_check(tmp_path, monkeypatch, name):
    namespace, *_ = uninstall_script_fixture(tmp_path, monkeypatch)
    root = tmp_path / name
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(mount_line(root / "locks" / "deep"))
    namespace["MOUNTINFO_PATH"] = mountinfo
    monkeypatch.setattr(namespace["os"].path, "ismount", lambda path: False)
    with pytest.raises(namespace["UninstallError"], match="AGENT_UNINSTALL_SHARED_MOUNT"):
        namespace["require_no_mounted_components"]([root])


def test_sibling_mount_is_not_an_agent_descendant(tmp_path, monkeypatch):
    namespace, prefix, units, _, _, _ = uninstall_script_fixture(tmp_path, monkeypatch)
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(mount_line(tmp_path / "agent-other" / "locks"))
    namespace["MOUNTINFO_PATH"] = mountinfo
    namespace["require_no_mounted_components"]([prefix])
    assert namespace["uninstall_agent"](str(prefix), units)["service_removed"]


def test_shared_mount_wrapper_returns_clear_error_without_removing_selected_unit(tmp_path):
    ssh = SSHRemote("192.0.2.1", 22, "root", "private-password", tmp_path,
                    service_name="termitech-vla-agent-mac-test")
    ssh._is_root = True
    ssh.probe = Mock(return_value={"_maintenance_python": "/usr/bin/python3.11",
        "checks": [{"name": "privileges", "ok": True}]})
    ssh._execute = Mock(return_value=(0, json.dumps({"error": "AGENT_UNINSTALL_SHARED_MOUNT"})))
    with pytest.raises(RemoteError, match="请先解除测试实例的共享锁挂载后再卸载") as error:
        ssh.uninstall("/data2/test-agent", Mock())
    assert error.value.code == "AGENT_UNINSTALL_SHARED_MOUNT"
