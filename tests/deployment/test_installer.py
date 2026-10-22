import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pytest


spec = importlib.util.spec_from_file_location(
    "vla_installer", Path(__file__).resolve().parents[2] / "deploy/install_agent.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def config_file(tmp_path, **changes):
    code = tmp_path / "training"
    code.mkdir(exist_ok=True)
    value = {
        "name": "GPU server", "control_plane_url": "http://127.0.0.1:18000",
        "roots": [str(tmp_path)], "output_root": str(tmp_path / "runs"),
        "state_dir": str(tmp_path / "state"), "profiles": [{
            "id": "sft", "python": sys.executable, "code_dir": str(code), "adapter": "openpi-sft",
        }], **changes,
    }
    path = tmp_path / "agent.json"
    path.write_text(json.dumps(value))
    return path


def test_training_environment_is_kept_separate_from_agent_venv(tmp_path):
    source = config_file(tmp_path)
    result = installer.read_configuration(source, tmp_path / "agent")
    assert result["profiles"][0]["python"] == sys.executable
    source.write_text(source.read_text().replace(str(tmp_path / "training"), str(tmp_path / "agent")))
    (tmp_path / "agent").mkdir()
    with pytest.raises(ValueError, match="分开"):
        installer.read_configuration(source, tmp_path / "agent")


def test_explicit_filesystem_access_root_is_accepted_only_for_managed_roots(tmp_path):
    source = config_file(tmp_path, roots=["/"])
    result = installer.read_configuration(source, tmp_path / "agent")
    assert result["roots"] == ["/"]
    assert result["state_dir"] == str(tmp_path / "state")
    assert installer.absolute_path("/", allow_root=True) == Path("/")
    with pytest.raises(ValueError, match="根目录"):
        installer.absolute_path("/")


def test_configuration_validation_subprocess_accepts_root_without_installing_or_touching_training(tmp_path):
    import subprocess

    source = config_file(tmp_path, roots=["/"])
    marker = tmp_path / "training/keep.py"
    marker.write_text("# training source remains unchanged\n")
    driver = (
        "import importlib.util,json,pathlib,sys; "
        "s=importlib.util.spec_from_file_location('installer',sys.argv[1]); "
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m); "
        "c=m.read_configuration(pathlib.Path(sys.argv[2]),m.dedicated_directory(sys.argv[3])); "
        "print(json.dumps({'roots':c['roots'],'profile_count':len(c['profiles'])}))"
    )
    result = subprocess.run([sys.executable, "-I", "-B", "-c", driver, str(Path(installer.__file__)),
                             str(source), str(tmp_path / "agent")], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"roots": ["/"], "profile_count": 1}
    assert marker.read_text() == "# training source remains unchanged\n"
    assert not (tmp_path / "agent").exists()
    assert not (tmp_path / "state").exists() and not (tmp_path / "runs").exists()


@pytest.mark.parametrize("role", ["state_dir", "output_root", "code_dir", "python"])
@pytest.mark.parametrize("bad", ["/", "//", "traversal", "root-symlink"])
def test_root_access_never_allows_root_or_traversal_for_critical_paths(tmp_path, role, bad):
    source = config_file(tmp_path, roots=["/"])
    value = json.loads(source.read_text())
    if bad == "traversal":
        invalid = str(tmp_path) + "/training/../training"
    elif bad == "root-symlink":
        alias = tmp_path / "root-alias"
        alias.symlink_to("/", target_is_directory=True)
        invalid = str(alias)
    else:
        invalid = bad
    if role in {"code_dir", "python"}:
        value["profiles"][0][role] = invalid
    else:
        value[role] = invalid
    source.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="根目录|\\.\\."):
        installer.read_configuration(source, tmp_path / "agent")


@pytest.mark.parametrize("invalid", ["/", "//", "traversal", "root-symlink"])
def test_install_prefix_root_or_traversal_is_rejected_before_any_changes(tmp_path, monkeypatch, invalid):
    if invalid == "traversal":
        path = str(tmp_path) + "/agent/../agent"
    elif invalid == "root-symlink":
        alias = tmp_path / "root-alias"
        alias.symlink_to("/", target_is_directory=True)
        path = str(alias)
    else:
        path = invalid
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer, "service_account", lambda *a: pytest.fail("Prefix validation must precede account/install work"))
    with pytest.raises(ValueError, match="根目录|\\.\\.|绝对路径"):
        installer._install(SimpleNamespace(prefix=path, service_name="test-agent"), installer.InstallEvents())


@pytest.mark.parametrize("role", ["state_dir", "output_root", "prefix"])
def test_root_access_cannot_change_ownership_of_a_filesystem_mount(tmp_path, monkeypatch, role):
    mount = tmp_path / "data-volume"
    mount.mkdir()
    original = installer.Path.is_mount
    monkeypatch.setattr(installer.Path, "is_mount", lambda path: path == mount or original(path))
    monkeypatch.setattr(installer.os, "chown", lambda *a, **kw: pytest.fail("Do not chown mountpoint"))
    source = config_file(tmp_path, roots=["/"], **({role: str(mount)} if role != "prefix" else {}))
    with pytest.raises(ValueError, match="挂载点"):
        if role == "prefix":
            installer.dedicated_directory(str(mount))
        else:
            installer.read_configuration(source, tmp_path / "agent")
    with pytest.raises(ValueError, match="挂载点"):
        installer.own_directory(mount, SimpleNamespace(pw_uid=0, pw_gid=0), 0o700)


def test_managed_roots_can_include_mountpoints_and_multiple_ordinary_paths(tmp_path, monkeypatch):
    code = tmp_path / "training"
    code.mkdir()
    output = tmp_path / "runs"
    output.mkdir()
    source = config_file(tmp_path, roots=[str(code), str(output)])
    monkeypatch.setattr(installer.Path, "is_mount", lambda path: path == code)
    result = installer.read_configuration(source, tmp_path / "agent")
    assert result["roots"] == [str(code), str(output)]
    assert not (tmp_path / "state").exists()


@pytest.mark.parametrize("roots", ["/", None, [], {"path": "/"}])
def test_managed_roots_require_an_explicit_nonempty_list(tmp_path, roots):
    with pytest.raises(ValueError, match="受管根目录"):
        installer.read_configuration(config_file(tmp_path, roots=roots), tmp_path / "agent")


def test_root_access_still_rejects_traversal_managed_boundaries(tmp_path):
    source = config_file(tmp_path, roots=[str(tmp_path) + "/training/../training"])
    with pytest.raises(ValueError, match="\\.\\."):
        installer.read_configuration(source, tmp_path / "agent")


def test_state_and_training_outputs_cannot_overlap(tmp_path):
    with pytest.raises(ValueError, match="分开"):
        installer.read_configuration(
            config_file(tmp_path, state_dir=str(tmp_path / "runs/state")), tmp_path / "agent"
        )


def test_remote_plaintext_control_plane_is_rejected_before_installation(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        installer.read_configuration(
            config_file(tmp_path, control_plane_url="http://public.example.com"), tmp_path / "agent"
        )


def test_upgrade_refuses_live_or_unreconciled_attempts(tmp_path):
    with sqlite3.connect(tmp_path / "agent.sqlite3") as db:
        db.execute("CREATE TABLE attempts (state TEXT)")
        db.execute("INSERT INTO attempts VALUES ('UNKNOWN')")
    with pytest.raises(RuntimeError, match="待对账"):
        installer.require_no_active_attempts(str(tmp_path))
    with sqlite3.connect(tmp_path / "agent.sqlite3") as db:
        db.execute("UPDATE attempts SET state='SUCCEEDED'")
    installer.require_no_active_attempts(str(tmp_path))


def test_installer_checks_bundle_integrity_before_running_package_code(tmp_path, monkeypatch):
    files = {"requirements.txt": b"locked", "install_agent.py": b"installer", "agent-bootstrap.sh": b"bootstrap",
             "agent_ownership.py": b"ownership",
             "install-agent.sh": b"shell", "platform.whl": b"wheel"}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    (tmp_path / "manifest.json").write_text(json.dumps({
        name: hashlib.sha256(data).hexdigest() for name, data in files.items()
    }))
    monkeypatch.setattr(installer, "BUNDLE", tmp_path)
    assert installer.verify_bundle().name == "platform.whl"
    (tmp_path / "platform.whl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="校验失败"):
        installer.verify_bundle()


def test_root_upgrade_reads_state_as_the_existing_owner(tmp_path, monkeypatch):
    database = tmp_path / "agent.sqlite3"
    database.touch()
    original_stat = installer.Path.stat
    monkeypatch.setattr(installer.Path, "stat", lambda path, *a, **kw:
                        SimpleNamespace(st_uid=12345) if path == database
                        else original_stat(path, *a, **kw))
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="vla-agent"))
    calls = []
    def output(argv, **kwargs):
        calls.append(argv)
        return "1\n"
    monkeypatch.setattr(installer.subprocess, "check_output", output)
    with pytest.raises(RuntimeError, match="待对账"):
        installer.require_no_active_attempts(str(tmp_path))
    assert calls[0][:3] == ["runuser", "-u", "vla-agent"]
    assert calls[0][-1] == str(database)


def event_lines(output):
    return [json.loads(line.removeprefix(installer.EVENT_PREFIX))
            for line in output.splitlines() if line.startswith(installer.EVENT_PREFIX)]


def test_subprocess_output_is_streamed_before_exit_with_unicode_cr_and_no_final_newline(capsys, monkeypatch):
    import time

    events = installer.InstallEvents()
    observed = []
    original_emit = events.emit

    def observe(phase, message, **kwargs):
        observed.append((message, time.monotonic()))
        original_emit(phase, message, **kwargs)

    monkeypatch.setattr(events, "emit", observe)
    script = (
        "import os,time; "
        "os.write(1, '下载中文'.encode()[:4]); time.sleep(.05); "
        "os.write(1, '下载中文'.encode()[4:]+b'\\r'); "
        "os.write(2,b'stderr detail\\n'); time.sleep(.25); "
        "os.write(1, '最后无换行'.encode())"
    )
    installer.run(sys.executable, "-u", "-c", script, events=events,
                  phase="DEPENDENCIES", label="安装测试依赖")
    finished = time.monotonic()
    output = event_lines(capsys.readouterr().out)
    messages = [event["message"] for event in output]
    assert "下载中文" in messages
    assert "stderr detail" in messages
    assert "最后无换行" in messages
    assert finished - next(moment for message, moment in observed if message == "下载中文") >= .2
    assert all(event["phase"] == "DEPENDENCIES" for event in output)
    assert all(event["level"] == "info" for event in output)


def test_silent_subprocess_reports_waiting_and_elapsed_time(capsys):
    events = installer.InstallEvents()
    installer.run(sys.executable, "-c", "import time; time.sleep(.22)", events=events,
                  phase="VERIFY", label="校验测试依赖", heartbeat_seconds=.05)
    output = event_lines(capsys.readouterr().out)
    waiting = [event for event in output if event["message"].startswith("仍在等待：")]
    assert len(waiting) >= 2
    assert all("尚未返回" in event["message"] for event in waiting)
    assert all(event["elapsed_seconds"] >= 0 for event in waiting)
    assert "完成：" in output[-1]["message"]


def test_large_output_is_drained_without_unbounded_log_lines(capsys):
    events = installer.InstallEvents()
    installer.run(sys.executable, "-c",
                  "import os; os.write(1,b'x'*250000); os.write(1,b'\\nafter-large-output\\n')",
                  events=events, phase="PACKAGE", label="大输出测试")
    output = event_lines(capsys.readouterr().out)
    assert any(event["level"] == "warning" and "过长" in event["message"] for event in output)
    assert any(event["message"] == "after-large-output" for event in output)
    assert all(len(event["message"]) <= 4000 for event in output)


def test_failure_preserves_private_log_and_redacts_credentials(tmp_path, capsys, monkeypatch):
    import stat

    enrollment = "enrollment-secret-example"
    env_secret = "env-secret-example"
    monkeypatch.setenv("VLA_TEST_PASSWORD", env_secret)

    def fail_install(args, events):
        events.protect_configuration({"enrollment_token": enrollment})
        events.emit("CHECKING", "before durable file")
        events.open_log(tmp_path)
        installer.run(sys.executable, "-c",
                      "import sys; print(sys.argv[1]); sys.exit(17)",
                      f"{enrollment} {env_secret} https://private-user:private-password@example.test/simple "
                      "https://host.test/file?token=query-secret \x1b[31mred\x1b[0m",
                      events=events, phase="DEPENDENCIES", label="安装测试依赖")

    monkeypatch.setattr(installer, "_install", fail_install)
    with pytest.raises(installer.InstallCommandError, match="退出码 17") as failure:
        installer.install(SimpleNamespace())
    paths = list((tmp_path / "logs").glob("install-*.jsonl"))
    assert len(paths) == 1
    assert stat.S_IMODE(paths[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(paths[0].parent.stat().st_mode) == 0o700
    stored = paths[0].read_text()
    output = capsys.readouterr().out
    for secret in (enrollment, env_secret, "private-user", "private-password", "query-secret"):
        assert secret not in output + stored + str(failure.value)
    assert "\x1b" not in output + stored
    records = [json.loads(line) for line in stored.splitlines()]
    assert records[0]["message"] == "before durable file"
    assert any(record["message"].startswith("失败：") for record in records)
    assert records[-1]["level"] == "error"
    assert "Agent 安装失败" in records[-1]["message"]


def test_log_directory_rejects_symlink_without_writing_target(tmp_path):
    target = tmp_path / "other"
    target.mkdir()
    prefix = tmp_path / "agent"
    prefix.mkdir()
    (prefix / "logs").symlink_to(target, target_is_directory=True)
    events = installer.InstallEvents()
    with pytest.raises(ValueError, match="符号链接"):
        events.open_log(prefix)
    assert list(target.iterdir()) == []


def test_silent_command_keeps_heartbeats_after_closing_its_output(capsys):
    events = installer.InstallEvents()
    installer.run(sys.executable, "-c",
                  "import os,time; os.close(1); os.close(2); time.sleep(.2)",
                  events=events, phase="ENVIRONMENT", label="静默环境检查", heartbeat_seconds=.05)
    output = event_lines(capsys.readouterr().out)
    assert sum(event["message"].startswith("仍在等待：") for event in output) >= 2


@pytest.mark.parametrize("line,hidden", [
    ("Authorization: Bearer unknown-credential", "unknown-credential"),
    ("Downloading https://user:pass@host.test/pkg.whl?auth=private#secret", "private"),
    ("Downloading https://host.test/pkg.whl#signature-data", "signature-data"),
    ('detail access_token="unknown token with spaces"', "unknown token with spaces"),
    ("detail storage_key='private key with spaces'", "private key with spaces"),
    ("detail --api-key hidden-cli-value", "hidden-cli-value"),
    ('  "enrollment_token": "json-config-secret",', "json-config-secret"),
    ('+ env SECRET=traced-secret bash -c install', "traced-secret"),
])
def test_durable_output_hides_unregistered_credentials_and_config_traces(line, hidden):
    result = installer.sanitize_install_output(line)
    assert hidden not in result
    if line.startswith("Downloading"):
        assert "https://" in result
        assert "pkg.whl" in result
    assert result


def test_durable_output_redacts_encoded_known_secrets_without_hiding_log_path():
    from urllib.parse import quote, quote_plus

    events = installer.InstallEvents()
    protected = '私密 value/"token'
    events.protect_configuration({"enrollment_token": protected})
    for value in (protected, quote(protected, safe=""), quote_plus(protected, safe=""), json.dumps(protected)[1:-1]):
        result = events.redact("dependency detail " + value)
        assert value not in result
        assert "[REDACTED]" in result
    assert "/data2/agent/logs/install-test.jsonl" in events.redact(
        "安装日志保存在 /data2/agent/logs/install-test.jsonl（仅管理员可读，失败后保留）"
    )


@pytest.mark.parametrize("line,hidden", [
    ("Authorization: Basic unknown-base64", "unknown-base64"),
    ("warning password=two unquoted secret words", "unquoted secret words"),
    ('diagnostic {"profiles": [{"python": "/secret/path"}]}', "/secret/path"),
    ("\x1b]unclosed-terminal-sequence", "terminal-sequence"),
])
def test_durable_output_handles_unquoted_secrets_config_and_unclosed_ansi(line, hidden):
    assert hidden not in installer.sanitize_install_output(line)


def test_durable_output_redacts_secret_fragments_after_control_cleanup():
    events = installer.InstallEvents()
    events.protect_configuration({"enrollment_token": "first-line\nsecond\tpart"})
    assert "first-line" not in events.redact("first-line")
    assert "secondpart" not in events.redact("second\tpart")


def test_agent_mirror_is_isolated_from_host_pip_indexes_and_tls_exemptions(tmp_path, monkeypatch):
    host_values = {"PIP_INDEX_URL": "https://other.example/simple", "PIP_EXTRA_INDEX_URL": "https://extra.example/simple",
                   "PIP_TRUSTED_HOST": "extra.example", "PIP_NO_INDEX": "1", "PIP_FIND_LINKS": "/training/wheels",
                   "PIP_TARGET": "/training/site-packages",
                   "PIP_CONFIG_FILE": "/training/pip.conf", "PYTHONPATH": "/training/code", "HTTPS_PROXY": "http://proxy.example:8080"}
    for key, value in host_values.items():
        monkeypatch.setenv(key, value)
    result = installer.pip_environment(tmp_path)
    assert result["PIP_INDEX_URL"] == "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
    assert result["PIP_CONFIG_FILE"] == installer.os.devnull
    assert result["PIP_REQUIRE_VIRTUALENV"] == "true"
    assert result["HTTPS_PROXY"] == host_values["HTTPS_PROXY"]
    assert not {"PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST", "PIP_NO_INDEX", "PIP_FIND_LINKS", "PIP_TARGET", "PYTHONPATH"} & result.keys()
    assert {key: installer.os.environ[key] for key in host_values} == host_values


def fake_runtime(path, kind="venv"):
    (path / "bin").mkdir(parents=True)
    (path / "bin/python").write_text("Agent interpreter")
    (path / "bin/python").chmod(0o755)
    if kind == "conda":
        (path / "conda-meta").mkdir()
    else:
        (path / "pyvenv.cfg").write_text("home = /usr/bin\n")


def test_conda_creates_dedicated_environment_without_touching_base_or_training(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/training/python")
    monkeypatch.setenv("PYTHONPATH", "/training/code")
    monkeypatch.setenv("CONDA_PREFIX", "/training/conda-env")
    monkeypatch.setenv("CONDA_PKGS_DIRS", "/training/conda-cache")
    calls = []
    monkeypatch.setattr(installer, "run", lambda *argv, **kw: calls.append((argv, kw)))
    runtime = installer.prepare_runtime(tmp_path, "conda", "/opt/conda/bin/conda", installer.InstallEvents())
    argv, options = calls[0]
    assert argv == ("/opt/conda/bin/conda", "create", "--prefix", str(tmp_path / ".venv"),
                    "python=3.11", "pip", "--yes")
    assert options["env"]["CONDA_PKGS_DIRS"] == str(tmp_path / "conda-pkgs")
    assert not {"PYTHONHOME", "PYTHONPATH", "CONDA_PREFIX"} & options["env"].keys()
    assert runtime.python == tmp_path / ".venv/bin/python"
    assert runtime.kind == "conda"
    assert installer.os.environ["CONDA_PREFIX"] == "/training/conda-env"


@pytest.mark.parametrize("selected,existing,expected", [("conda", "conda", "conda"),
                                                       ("venv", "conda", "conda"),
                                                       ("venv", "venv", "venv")])
def test_upgrade_reuses_existing_agent_runtime(tmp_path, monkeypatch, selected, existing, expected):
    path = tmp_path / ".venv"
    fake_runtime(path, existing)
    (path / "installed-package").write_text("keep")
    monkeypatch.setattr(installer, "run", lambda *a, **kw: pytest.fail("Existing runtime must be reused"))
    runtime = installer.prepare_runtime(tmp_path, selected, "/opt/conda/bin/conda", installer.InstallEvents())
    assert runtime.kind == expected
    assert runtime.backup is None
    assert (path / "installed-package").read_text() == "keep"


def test_missing_conda_falls_back_to_isolated_default_python_venv(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(installer, "run", lambda *argv, **kw: calls.append((argv, kw)))
    runtime = installer.prepare_runtime(tmp_path, "venv", None, installer.InstallEvents())
    assert calls[0][0] == (sys.executable, "-I", "-m", "venv", str(tmp_path / ".venv"))
    assert runtime.kind == "venv"


def test_venv_migration_keeps_backup_and_restores_it_on_creation_failure(tmp_path, monkeypatch):
    path = tmp_path / ".venv"
    fake_runtime(path)
    (path / "old-package").write_text("old installation")
    calls = []

    def fail_conda(*argv, **kwargs):
        calls.append(argv)
        assert not path.exists()
        assert len(list(tmp_path.glob(".venv.before-conda-*"))) == 1
        fake_runtime(path, "conda")
        raise installer.InstallCommandError("Conda channel unavailable")

    monkeypatch.setattr(installer, "run", fail_conda)
    with pytest.raises(installer.InstallCommandError, match="channel unavailable"):
        installer.prepare_runtime(tmp_path, "conda", "/opt/conda/bin/conda", installer.InstallEvents())
    assert len(calls) == 1  # A failed Conda creation must never fall back to venv.
    assert (path / "old-package").read_text() == "old installation"
    assert (path / "pyvenv.cfg").is_file()
    assert not (path / "conda-meta").exists()
    assert not list(tmp_path.glob(".venv.before-conda-*"))


def test_failed_fresh_conda_creation_removes_partial_environment_for_retry(tmp_path, monkeypatch):
    calls = []

    def fail_conda(*argv, **kwargs):
        calls.append(argv)
        fake_runtime(tmp_path / ".venv", "conda")
        raise installer.InstallCommandError("Conda channel unavailable")

    monkeypatch.setattr(installer, "run", fail_conda)
    with pytest.raises(installer.InstallCommandError, match="channel unavailable"):
        installer.prepare_runtime(tmp_path, "conda", "/opt/conda/bin/conda", installer.InstallEvents())
    assert len(calls) == 1
    assert not (tmp_path / ".venv").exists()


def test_service_user_verifies_target_interpreter_and_pip_in_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONHOME", "/root/training-python")
    calls = []
    monkeypatch.setattr(installer, "run", lambda *argv, **kw: calls.append((argv, kw)))
    runtime = installer.AgentRuntime(tmp_path / ".venv", "conda")
    account = SimpleNamespace(pw_name="operator")
    installer.verify_runtime(runtime, tmp_path, account, installer.InstallEvents())
    argv, options = calls[0]
    assert argv[:7] == ("runuser", "-u", "operator", "--", str(runtime.python), "-I", "-c")
    assert "sys.version_info >= (3,11)" in argv[7]
    assert "pathlib.Path(sys.prefix).resolve() == expected" in argv[7]
    assert "pip.__file__" in argv[7]
    assert argv[-1] == str(runtime.path)
    assert "PYTHONHOME" not in options["env"]
    assert options["env"]["HOME"] == str(tmp_path / "home")


def test_conda_pip_is_explicitly_isolated_without_virtualenv_only_restriction(tmp_path, monkeypatch):
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV", "PIP_PREFIX", "PIP_TARGET"):
        monkeypatch.setenv(key, "/training/environment")
    result = installer.pip_environment(tmp_path, "conda")
    assert result["PIP_REQUIRE_VIRTUALENV"] == "false"
    assert not {"PYTHONHOME", "PYTHONPATH", "PYTHONUSERBASE", "VIRTUAL_ENV", "PIP_PREFIX", "PIP_TARGET"} & result.keys()
    assert result["PIP_CONFIG_FILE"] == installer.os.devnull


def test_conda_bootstrap_can_be_older_than_target_python(tmp_path, monkeypatch):
    conda = tmp_path / "conda"
    conda.touch(mode=0o755)
    monkeypatch.setattr(installer.sys, "version_info", (3, 10, 12))
    assert installer.runtime_request(SimpleNamespace(runtime_kind="conda", conda_executable=str(conda)), tmp_path) == (
        "conda", str(conda))
    with pytest.raises(RuntimeError, match="3.11"):
        installer.runtime_request(SimpleNamespace(runtime_kind="venv"), tmp_path)


def test_runtime_guard_rejects_link_to_training_environment(tmp_path):
    training = tmp_path / "training"
    fake_runtime(training, "conda")
    (tmp_path / ".venv").symlink_to(training, target_is_directory=True)
    with pytest.raises(ValueError, match="符号链接"):
        installer.runtime_request(SimpleNamespace(runtime_kind="venv"), tmp_path)
    assert (training / "conda-meta").is_dir()


def test_failed_dependency_upgrade_restores_venv_before_restarting_original_service(tmp_path, monkeypatch):
    prefix = tmp_path / "agent"
    fake_runtime(prefix / ".venv")
    (prefix / ".venv/old-package").write_text("old installation")
    conda = tmp_path / "conda"
    conda.touch(mode=0o755)
    source = config_file(tmp_path)
    args = SimpleNamespace(config=str(source), prefix=str(prefix), service_name="test-agent",
                           runtime_kind="conda", conda_executable=str(conda), service_user="operator")
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.shutil, "which", lambda value: "/usr/bin/" + value)
    monkeypatch.setattr(installer, "verify_bundle", lambda: tmp_path / "agent.whl")
    monkeypatch.setattr(installer, "check_service_location", lambda unit, path: True)
    monkeypatch.setattr(installer.pwd, "getpwnam", lambda user: SimpleNamespace(pw_name=user, pw_uid=123, pw_gid=123, pw_dir="/home/operator"))
    monkeypatch.setattr(installer, "own_directory", lambda *a: None)
    monkeypatch.setattr(installer, "ownership_migration", lambda *a: SimpleNamespace(migrate=lambda *a, **kw: None, rollback=lambda: None))
    monkeypatch.setattr(installer, "verify_training_interpreters", lambda *a: None)
    monkeypatch.setattr(installer.InstallEvents, "open_log", lambda self, prefix: prefix / "logs/test.jsonl")
    calls = []

    def run(*argv, **kwargs):
        calls.append((argv, kwargs))
        if argv[0] == str(conda):
            fake_runtime(prefix / ".venv", "conda")
        if argv[:5] == (str(prefix / ".venv/bin/python"), "-I", "-m", "pip", "install"):
            assert kwargs["env"]["PIP_REQUIRE_VIRTUALENV"] == "false"
            raise installer.InstallCommandError("dependency download failed")
        if argv[:2] == ("systemctl", "start"):
            assert (prefix / ".venv/old-package").read_text() == "old installation"
            assert (prefix / ".venv/pyvenv.cfg").is_file()

    monkeypatch.setattr(installer, "run", run)
    events = installer.InstallEvents()
    try:
        with pytest.raises(installer.InstallCommandError, match="dependency download failed"):
            installer._install(args, events)
    finally:
        events.close()
    assert calls[0][0] == ("systemctl", "stop", "test-agent.service")
    assert calls[-1][0] == ("systemctl", "start", "test-agent.service")
    assert not list(prefix.glob(".venv.before-conda-*"))


@pytest.mark.parametrize("guard", ["active-task", "changed-state-dir", "invalid-conda"])
def test_runtime_upgrade_guards_run_before_stopping_service(tmp_path, monkeypatch, guard):
    prefix = tmp_path / "agent"
    fake_runtime(prefix / ".venv")
    conda = tmp_path / "conda"
    conda.touch(mode=0o755)
    source = config_file(tmp_path)
    args = SimpleNamespace(config=str(source), prefix=str(prefix), service_name="test-agent",
                           runtime_kind="conda", conda_executable=str(conda))
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.shutil, "which", lambda value: "/usr/bin/" + value)
    monkeypatch.setattr(installer, "verify_bundle", lambda: tmp_path / "agent.whl")
    monkeypatch.setattr(installer, "check_service_location", lambda unit, path: True)
    monkeypatch.setattr(installer, "run", lambda *a, **kw: pytest.fail("Must not stop or modify existing service"))
    if guard == "invalid-conda":
        conda.unlink()
        error, message = ValueError, "Conda 可执行文件不可用"
    elif guard == "changed-state-dir":
        (prefix / "config").mkdir()
        (prefix / "config/agent.json").write_text(json.dumps({"state_dir": str(tmp_path / "original-state")}))
        error, message = RuntimeError, "state_dir"
    else:
        def reject_active(*args):
            raise RuntimeError("待对账任务")
        monkeypatch.setattr(installer, "require_no_active_attempts", reject_active)
        error, message = RuntimeError, "待对账任务"
    with pytest.raises(error, match=message):
        installer._install(args, installer.InstallEvents())
    assert (prefix / ".venv/pyvenv.cfg").is_file()
    assert not list(prefix.glob(".venv.before-conda-*"))


def test_root_state_reader_prefers_existing_agent_interpreter(tmp_path, monkeypatch):
    database = tmp_path / "agent.sqlite3"
    database.touch()
    runtime_python = tmp_path / "agent/.venv/bin/python"
    runtime_python.parent.mkdir(parents=True)
    runtime_python.touch()
    original_stat = installer.Path.stat
    monkeypatch.setattr(installer.Path, "stat", lambda path, *a, **kw:
                        SimpleNamespace(st_uid=12345) if path == database
                        else original_stat(path, *a, **kw))
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.pwd, "getpwuid", lambda uid: SimpleNamespace(pw_name="vla-agent"))
    calls = []
    monkeypatch.setattr(installer.subprocess, "check_output", lambda argv, **kw: calls.append(argv) or "0\n")
    installer.require_no_active_attempts(str(tmp_path), runtime_python)
    assert calls[0][4:7] == [str(runtime_python), "-I", "-c"]


def account(name="operator", uid=123, gid=456):
    return SimpleNamespace(pw_name=name, pw_uid=uid, pw_gid=gid, pw_dir="/home/" + name)


def test_service_account_uses_explicit_ssh_user_before_sudo_default(monkeypatch):
    monkeypatch.setenv("SUDO_USER", "original-login")
    seen = []
    monkeypatch.setattr(installer.pwd, "getpwnam", lambda name: seen.append(name) or account(name))
    assert installer.service_account("ssh-login").pw_name == "ssh-login"
    assert installer.service_account().pw_name == "original-login"
    assert seen == ["ssh-login", "original-login"]


def test_service_account_without_sudo_uses_current_user_including_root(monkeypatch):
    monkeypatch.delenv("SUDO_USER", raising=False)
    monkeypatch.setattr(installer.os, "getuid", lambda: 0)
    monkeypatch.setattr(installer.pwd, "getpwuid", lambda uid: account("root", 0, 0))
    monkeypatch.setattr(installer.pwd, "getpwnam", lambda name: account(name, 0, 0))
    assert installer.service_account().pw_name == "root"


@pytest.mark.parametrize("name", ["--root", "user;id", "a b", "a\nroot", "../root"])
def test_service_account_rejects_unsafe_names(name):
    with pytest.raises(ValueError, match="用户名"):
        installer.service_account(name)


def test_service_account_must_already_exist(monkeypatch):
    def missing(name):
        raise KeyError(name)
    monkeypatch.setattr(installer.pwd, "getpwnam", missing)
    with pytest.raises(ValueError, match="不存在"):
        installer.service_account("missing-user")


def test_loaded_service_with_default_user_is_verified_as_root(monkeypatch):
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kw:
                        calls.append(argv) or SimpleNamespace(returncode=0, stdout="loaded\n"))
    monkeypatch.setattr(installer.subprocess, "check_output", lambda argv, **kw: calls.append(argv) or "\n")
    monkeypatch.setattr(installer.pwd, "getpwnam", lambda name: account(name, 0, 0))
    assert installer.previous_service_account("test-agent.service").pw_name == "root"
    assert "--property=LoadState" in calls[0]
    assert "--property=User" in calls[1]


def test_training_probe_uses_selected_user_and_no_site_imports_or_bytecode(tmp_path, monkeypatch):
    config = json.loads(config_file(tmp_path).read_text())
    monkeypatch.setenv("PYTHONHOME", "/training/other-python")
    monkeypatch.setenv("PYTHONPATH", "/training/injected")
    calls = []

    def completed(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"version": [3, 11, 15], "executable": sys.executable}))

    monkeypatch.setattr(installer.subprocess, "run", completed)
    installer.verify_training_interpreters(config, account("ssh-user"), installer.InstallEvents())
    argv, options = calls[0]
    assert argv[:9] == ["runuser", "-u", "ssh-user", "--", sys.executable, "-I", "-S", "-B", "-c"]
    assert "os.chdir(sys.argv[1])" in argv[9]
    assert argv[-1] == options["cwd"] == str(tmp_path / "training")
    assert options["timeout"] == 20
    assert options["env"]["HOME"] == "/home/ssh-user"
    assert not {"PYTHONHOME", "PYTHONPATH"} & options["env"].keys()


@pytest.mark.parametrize("failure,expected", [("permission", "无法由服务账户执行"),
                                             ("timeout", "检查超时"), ("cwd", "目录无法访问"),
                                             ("invalid-json", "有效的版本信息")])
def test_training_probe_failures_identify_profile_interpreter_and_actual_account(tmp_path, monkeypatch, failure, expected):
    config = json.loads(config_file(tmp_path).read_text())

    def failed(argv, **kwargs):
        if failure == "timeout":
            raise installer.subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if failure == "cwd":
            raise PermissionError("denied")
        return SimpleNamespace(returncode=1 if failure == "permission" else 0, stdout="invalid")

    monkeypatch.setattr(installer.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match=expected) as result:
        installer.verify_training_interpreters(config, account("actual-ssh-user"), installer.InstallEvents())
    message = str(result.value)
    assert all(part in message for part in ("sft", sys.executable, "actual-ssh-user", str(tmp_path / "training")))
    assert "符号链接" in message


def test_stdlib_training_probe_does_not_execute_checkout_sitecustomize_or_create_bytecode(tmp_path, monkeypatch):
    config = json.loads(config_file(tmp_path).read_text())
    code = tmp_path / "training"
    (code / "sitecustomize.py").write_text("raise RuntimeError('must never import checkout code')\n")
    original_run = installer.subprocess.run
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kw: original_run(argv[4:], **kw))
    installer.verify_training_interpreters(config, account(), installer.InstallEvents())
    assert not (code / "__pycache__").exists()


def upgrade_fixture(tmp_path, monkeypatch):
    prefix = tmp_path / "agent"
    fake_runtime(prefix / ".venv", "conda")
    source = config_file(tmp_path)
    (prefix / "config").mkdir()
    (prefix / "config/agent.json").write_bytes(source.read_bytes())
    unit_directory = tmp_path / "systemd"
    unit_directory.mkdir()
    (unit_directory / "test-agent.service").write_text("[Service]\nUser=legacy-agent\n")
    args = SimpleNamespace(config=str(source), prefix=str(prefix), service_name="test-agent",
                           runtime_kind="venv", service_user="ssh-user")
    selected, previous = account("ssh-user"), account("legacy-agent", 996, 996)
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.os, "geteuid", lambda: 0)
    monkeypatch.setattr(installer.shutil, "which", lambda value: "/usr/bin/" + value)
    monkeypatch.setattr(installer, "verify_bundle", lambda: tmp_path / "agent.whl")
    monkeypatch.setattr(installer, "service_account", lambda name: selected)
    monkeypatch.setattr(installer, "previous_service_account", lambda unit: previous)
    monkeypatch.setattr(installer, "check_service_location", lambda unit, path: True)
    monkeypatch.setattr(installer.InstallEvents, "open_log", lambda self, prefix: prefix / "logs/test.jsonl")
    monkeypatch.setattr(installer, "SYSTEMD_UNIT_DIRECTORY", unit_directory)
    return args, prefix, selected, previous, unit_directory / "test-agent.service"


@pytest.mark.parametrize("failure", ["permission", "timeout"])
def test_training_access_failure_precedes_service_stop_and_all_migrations(tmp_path, monkeypatch, failure):
    args, prefix, selected, previous, unit = upgrade_fixture(tmp_path, monkeypatch)
    before_config, before_unit = (prefix / "config/agent.json").read_bytes(), unit.read_bytes()
    monkeypatch.setattr(installer, "run", lambda *a, **kw: pytest.fail("Service or dependency command ran before access guard"))
    monkeypatch.setattr(installer, "ownership_migration", lambda *a: pytest.fail("Private ownership changed before access guard"))

    def failed(argv, **kwargs):
        assert argv[:4] == ["runuser", "-u", "ssh-user", "--"]
        if failure == "timeout":
            raise installer.subprocess.TimeoutExpired(argv, 20)
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(installer.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="训练 Python"):
        installer._install(args, installer.InstallEvents())
    assert (prefix / "config/agent.json").read_bytes() == before_config
    assert unit.read_bytes() == before_unit
    assert not (prefix / "config/agent.installing.json").exists()


@pytest.mark.parametrize("managed_roots", [None, ["/"]])
def test_ssh_account_upgrade_rolls_back_files_and_ownership_before_old_service_restart(tmp_path, monkeypatch, managed_roots):
    args, prefix, selected, previous, unit = upgrade_fixture(tmp_path, monkeypatch)
    if managed_roots is not None:
        for path in (Path(args.config), prefix / "config/agent.json"):
            config = json.loads(path.read_text())
            config["roots"] = managed_roots
            path.write_text(json.dumps(config))
    old_config, old_unit = (prefix / "config/agent.json").read_bytes(), unit.read_bytes()
    calls, migrations, rollback = [], [], []
    monkeypatch.setattr(installer, "verify_training_interpreters", lambda *a: calls.append("training-access"))
    monkeypatch.setattr(installer, "verify_runtime", lambda runtime, prefix, account, events: calls.append("runtime-check"))
    monkeypatch.setattr(installer, "own_directory", lambda *a: None)
    monkeypatch.setattr(installer.os, "chown", lambda *a, **kw: None)
    monkeypatch.setattr(installer.os, "fchown", lambda *a, **kw: None)

    def migration_factory(new, old):
        assert (new, old) == (selected, previous)
        assert ("systemctl", "stop", "test-agent.service") in calls
        def migrate(path, recursive=False):
            migrations.append((path, recursive))
        def restore():
            assert (prefix / "config/agent.json").read_bytes() == old_config
            assert unit.read_bytes() == old_unit
            rollback.append("restored")
        return SimpleNamespace(migrate=migrate, rollback=restore)

    monkeypatch.setattr(installer, "ownership_migration", migration_factory)
    def run(*argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ("systemctl", "enable", "--now"):
            assert "User=ssh-user\n" in unit.read_text()
            assert "Group=456\n" in unit.read_text()
            raise installer.InstallCommandError("new service failed")
        if argv[:2] == ("systemctl", "start"):
            assert rollback == ["restored"]
            assert unit.read_bytes() == old_unit
        if argv[:2] == ("runuser", "-u"):
            assert argv[2] == "ssh-user"
    monkeypatch.setattr(installer, "run", run)
    with pytest.raises(installer.InstallCommandError, match="new service failed"):
        installer._install(args, installer.InstallEvents())
    assert calls[0] == "training-access"
    assert migrations == [(prefix / "home", True), (prefix / "tmp", True),
                          (tmp_path / "state", True), (tmp_path / "runs", False)]
    assert calls[-3:] == [("systemctl", "stop", "test-agent.service"), ("systemctl", "daemon-reload"),
                          ("systemctl", "start", "test-agent.service")]


def test_fresh_install_keeps_nonempty_foreign_directory_guard_before_ownership_changes(tmp_path, monkeypatch):
    args, prefix, selected, previous, unit = upgrade_fixture(tmp_path, monkeypatch)
    (prefix / "config/agent.json").unlink()
    unit.unlink()
    (prefix / "home").mkdir()
    (prefix / "home/private-existing-data").write_text("keep")
    monkeypatch.setattr(installer, "verify_training_interpreters", lambda *a: None)
    monkeypatch.setattr(installer, "check_service_location", lambda *a: False)
    monkeypatch.setattr(installer, "run", lambda *a, **kw: pytest.fail("No install commands before directory guard"))
    monkeypatch.setattr(installer, "ownership_migration", lambda *a: pytest.fail("Fresh install must not migrate existing owners"))
    original_stat = installer.Path.stat
    monkeypatch.setattr(installer.Path, "stat", lambda path, *a, **kw:
                        SimpleNamespace(st_uid=0, st_mode=original_stat(path, *a, **kw).st_mode,
                                        st_dev=original_stat(path, *a, **kw).st_dev, st_ino=original_stat(path, *a, **kw).st_ino)
                        if path == prefix / "home" else original_stat(path, *a, **kw))
    with pytest.raises(ValueError, match="其他账户的数据"):
        installer._install(args, installer.InstallEvents())
    assert (prefix / "home/private-existing-data").read_text() == "keep"


def test_upgrade_does_not_migrate_new_output_path(tmp_path, monkeypatch):
    args, prefix, selected, previous, unit = upgrade_fixture(tmp_path, monkeypatch)
    source = Path(args.config)
    config = json.loads(source.read_text())
    config["output_root"] = str(tmp_path / "new-runs")
    source.write_text(json.dumps(config))
    calls, migrations = [], []
    monkeypatch.setattr(installer, "verify_training_interpreters", lambda *a: None)
    monkeypatch.setattr(installer, "run", lambda *a, **kw: calls.append(a))
    monkeypatch.setattr(installer, "ownership_migration", lambda *a: SimpleNamespace(
        migrate=lambda path, **kw: migrations.append(path), rollback=lambda: None))

    def guard(path, account, mode):
        if path == tmp_path / "new-runs":
            raise ValueError("new output belongs to another account")
    monkeypatch.setattr(installer, "own_directory", guard)
    with pytest.raises(ValueError, match="new output"):
        installer._install(args, installer.InstallEvents())
    assert migrations == [prefix / "home", prefix / "tmp", tmp_path / "state"]
    assert calls == [("systemctl", "stop", "test-agent.service"), ("systemctl", "start", "test-agent.service")]


def test_leftover_config_without_loaded_unit_cannot_authorize_ownership_migration(tmp_path, monkeypatch):
    previous_service_account = installer.previous_service_account
    args, prefix, selected, previous, unit = upgrade_fixture(tmp_path, monkeypatch)
    before = (prefix / "config/agent.json").read_bytes()
    unit.unlink()
    monkeypatch.setattr(installer, "check_service_location", lambda *a: False)
    monkeypatch.setattr(installer, "previous_service_account", previous_service_account)
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0, stdout="not-found\n"))
    monkeypatch.setattr(installer.subprocess, "check_output", lambda *a, **kw: pytest.fail("Do not trust User from missing unit"))
    monkeypatch.setattr(installer, "run", lambda *a, **kw: pytest.fail("Do not stop or modify service"))
    monkeypatch.setattr(installer, "ownership_migration", lambda *a: pytest.fail("Leftover config cannot authorize migration"))
    with pytest.raises(RuntimeError, match="缺少已加载"):
        installer._install(args, installer.InstallEvents())
    assert (prefix / "config/agent.json").read_bytes() == before
    assert not (prefix / "home").exists()
