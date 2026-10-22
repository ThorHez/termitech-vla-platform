"""Catalog probes run with training Python and keep heartbeat dispatch independent."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
import subprocess
import sys

import pytest

from vla_agent import data_processors as catalog_module
from vla_agent.data_processors import data_processors
from vla_agent.main import _execute_preflight_command
from vla_agent.safety import AgentError
from vla_agent.state import State

def query(**changes):
    return {"profile_id": "openpi-native-1", "expires_at": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(), **changes}


def prepared(tmp_path, monkeypatch, *, probe=None):
    root = tmp_path / "checkout"
    path = root / "src/openpi/training/config.py"
    path.parent.mkdir(parents=True)
    path.write_text("# Catalog source\n")
    for args in (("init", "-b", "fixture"), ("config", "user.name", "Catalog Tests"),
                 ("config", "user.email", "catalog@example.invalid"), ("add", "."), ("commit", "-m", "Fixture")):
        subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
    config = {"roots": [str(tmp_path)], "profiles": [{"id": "openpi-native-1", "adapter": "openpi-sft",
              "execution_mode": "native", "code_dir": str(root), "python": sys.executable}]}
    calls = []

    def inspect(argv, cwd, environment, **options):
        calls.append((argv, cwd, environment, options))
        if probe is not None:
            return probe(path)
        source = {"path": "src/openpi/training/config.py", "code_dir": config["profiles"][0]["code_dir"],
                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        return 0, "import log\nVLA_DATA_PROCESSORS:" + json.dumps({"entries": [], "source": source}) + "\n", ""

    monkeypatch.setattr(catalog_module, "_probe_process", inspect)
    return config, path, calls


def test_catalog_uses_training_interpreter_cpu_offline_and_strips_credentials(tmp_path, monkeypatch):
    config, path, calls = prepared(tmp_path, monkeypatch)
    monkeypatch.setenv("VLA_ADMIN_TOKEN", "must-not-leak")
    monkeypatch.setenv("VLA_AGENT_TOKEN", "must-not-leak")
    monkeypatch.setenv("HF_TOKEN", "must-not-leak")
    monkeypatch.setenv("WANDB_API_KEY", "must-not-leak")
    result = data_processors(config, query())
    argv, cwd, environment, options = calls[0]
    assert argv[0] == sys.executable and argv[1] == "-B" and argv[2].endswith("/vla_adapters/data_processors.py")
    assert argv[3] == "--catalog-json"
    assert json.loads(argv[4]) == {"code_dir": config["profiles"][0]["code_dir"], "allowed_roots": config["roots"]}
    assert cwd == config["profiles"][0]["code_dir"]
    assert not any(key.startswith("VLA_") for key in environment) and "HF_TOKEN" not in environment and "WANDB_API_KEY" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "" and environment["JAX_PLATFORMS"] == "cpu"
    assert environment["HF_HUB_OFFLINE"] == "1" and environment["HF_DATASETS_OFFLINE"] == "1"
    assert options["timeout"] == 45 and options["output_limit"] == 1024 * 1024
    assert result["profile_id"] == "openpi-native-1" and result["entries"] == []
    assert result["source"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(result["source"]["commit_sha"]) == 40 and result["source"]["dirty"] is False


@pytest.mark.parametrize("change,code", [
    ({"expires_at": (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()}, "DATA_PROCESSORS_EXPIRED"),
    ({"expires_at": "bad-time"}, "INVALID_DATA_PROCESSORS"),
    ({"profile_id": "missing"}, "PROFILE_UNAVAILABLE"),
])
def test_catalog_rejects_invalid_scope_or_expired_request_before_running_python(tmp_path, monkeypatch, change, code):
    config, _, calls = prepared(tmp_path, monkeypatch)
    with pytest.raises(AgentError) as error:
        data_processors(config, query(**change))
    assert error.value.code == code and not calls


def test_catalog_rejects_mismatched_or_changed_source(tmp_path, monkeypatch):
    def mutate(path):
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        path.write_text("# Changed during import\n")
        return 0, "VLA_DATA_PROCESSORS:" + json.dumps({"entries": [], "source": {
            "path": "src/openpi/training/config.py", "code_dir": str(path.parents[3]), "sha256": old_hash}}), ""
    config, _, _ = prepared(tmp_path, monkeypatch, probe=mutate)
    with pytest.raises(AgentError) as error:
        data_processors(config, query())
    assert error.value.code == "RECIPE_SOURCE_CHANGED"


@pytest.mark.parametrize("code,expected", [("PREFLIGHT_TIMEOUT", "DATA_PROCESSORS_TIMEOUT"),
                                         ("PREFLIGHT_OUTPUT_LIMIT", "DATA_PROCESSORS_OUTPUT_LIMIT")])
def test_probe_limits_map_to_catalog_errors(tmp_path, monkeypatch, code, expected):
    config, _, _ = prepared(tmp_path, monkeypatch)
    def fail(*_args, **_kwargs):
        raise AgentError(code, "bounded probe")
    monkeypatch.setattr(catalog_module, "_probe_process", fail)
    with pytest.raises(AgentError) as error:
        data_processors(config, query())
    assert error.value.code == expected


def test_catalog_worker_dispatch_is_durable_and_retry_idempotent(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    command = {"id": "processor-catalog", "type": "DATA_PROCESSORS", "payload": {"profile_id": "some-profile"}}
    monkeypatch.setattr("vla_agent.main.os.setsid", lambda: None)
    monkeypatch.setattr("vla_agent.main.signal.signal", lambda *_: None)
    calls = []
    def inspect(_config, payload):
        calls.append(payload)
        return {"profile_id": payload["profile_id"], "entries": []}
    monkeypatch.setattr(catalog_module, "data_processors", inspect)
    _execute_preflight_command(config, command)
    _execute_preflight_command(config, command)
    assert calls == [{"profile_id": "some-profile"}]
    assert State(config["state_dir"]).command(command["id"]) == {
        "status": "APPLIED", "result": {"profile_id": "some-profile", "entries": []}}
