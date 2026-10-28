"""Source export and target verification use no real policy or hardware."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters.portable_policy import _adapter_files, _code_files, portable_digest
from vla_agent.main import Agent
from vla_agent.policy_deployment import _portable, handle_policy_command
from vla_agent.policy_export import execute_export_command, handle_export_command
from vla_agent.safety import AgentError


def _description():
    # Manifest integrity verification does not import any model; the worker
    # separately validates code identities and registered constructors.
    return {"schema_version": 1, "model": {}, "data": {}, "policy_metadata": {},
            "code_files": {}, "adapter_files": {}, "runtime_versions": {}, "model_family": "Pi0Config"}


def _transfer(root):
    checkpoint = root / "checkpoint"
    (checkpoint / "assets").mkdir(parents=True)
    (checkpoint / "assets/norm_stats.json").write_text("{}")
    (checkpoint / "params").mkdir()
    (checkpoint / "params/weights").write_bytes(b"weights\0\1")
    description = _description()
    package = {"config": description, "digest": portable_digest(description)}
    (checkpoint / "termitech-policy.json").write_text(json.dumps(package, ensure_ascii=False))
    files = [{"path": str(path.relative_to(checkpoint)), "size": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(checkpoint.rglob("*")) if path.is_file()]
    manifest = {"schema_version": 1, "files": files, "bytes_total": sum(row["size"] for row in files), "files_total": len(files)}
    return checkpoint, {**package, "manifest": manifest, "checkpoint_manifest_digest": portable_digest(manifest), "transfer_id": "transfer-1"}


def test_verified_target_receipt_binds_full_inventory_and_sidecar(tmp_path):
    checkpoint, package = _transfer(tmp_path)
    config = {"roots": [str(tmp_path)]}
    result = _portable(config, checkpoint, {"portable_policy": package})
    assert result["portable_policy"] == {"config": package["config"], "digest": package["digest"]}
    assert result["checkpoint_manifest_digest"] == package["checkpoint_manifest_digest"]
    assert result["transfer_id"] == "transfer-1"
    assert "manifest" not in result["portable_policy"]


def test_filtered_checkpoint_receipt_ignores_only_root_training_directory(tmp_path, monkeypatch):
    import os
    checkpoint, package = _transfer(tmp_path)
    ignored = checkpoint / "train_state" / "optimizer"
    ignored.parent.mkdir()
    ignored.write_bytes(b"old partial optimizer data")
    original_open = os.open
    def guarded_open(path, *_args, **_kwargs):
        assert Path(path) != ignored, "Excluded optimizer state must not be read or hashed"
        return original_open(path, *_args, **_kwargs)
    monkeypatch.setattr("vla_agent.policy_deployment.os.open", guarded_open)
    result = _portable({"roots": [str(tmp_path)]}, checkpoint, {"portable_policy": package})
    assert result["portable_policy"]["digest"] == package["digest"]
    assert ignored.read_bytes() == b"old partial optimizer data"
    # A same-named nested directory is part of the policy tree, not excluded.
    nested = checkpoint / "params" / "train_state"
    nested.mkdir()
    (nested / "unlisted").write_bytes(b"must verify")
    with pytest.raises(AgentError) as error:
        _portable({"roots": [str(tmp_path)]}, checkpoint, {"portable_policy": package})
    assert error.value.code == "POLICY_TRANSFER_INTEGRITY_INVALID"


@pytest.mark.parametrize("root_entry", ("file", "symlink"))
def test_training_state_exclusion_never_hides_plain_files_or_symlinks(tmp_path, root_entry):
    checkpoint, package = _transfer(tmp_path)
    if root_entry == "file":
        (checkpoint / "train_state").write_bytes(b"ordinary unlisted file")
    else:
        (checkpoint / "train_state").symlink_to(checkpoint / "params", target_is_directory=True)
    with pytest.raises(AgentError) as error:
        _portable({"roots": [str(tmp_path)]}, checkpoint, {"portable_policy": package})
    assert error.value.code == "POLICY_TRANSFER_INTEGRITY_INVALID"


def test_legacy_published_receipt_still_verifies_explicit_optimizer_files(tmp_path):
    checkpoint, package = _transfer(tmp_path)
    optimizer = checkpoint / "train_state" / "optimizer"
    optimizer.parent.mkdir()
    optimizer.write_bytes(b"previously verified optimizer")
    manifest = package["manifest"]
    manifest["files"].append({"path": "train_state/optimizer", "size": optimizer.stat().st_size,
                              "sha256": hashlib.sha256(optimizer.read_bytes()).hexdigest()})
    manifest["bytes_total"] += optimizer.stat().st_size
    manifest["files_total"] += 1
    package["checkpoint_manifest_digest"] = portable_digest(manifest)
    assert _portable({"roots": [str(tmp_path)]}, checkpoint, {"portable_policy": package})["checkpoint_manifest_digest"] == package["checkpoint_manifest_digest"]
    optimizer.write_bytes(b"corrupted prior optimizer")
    with pytest.raises(AgentError) as error:
        _portable({"roots": [str(tmp_path)]}, checkpoint, {"portable_policy": package})
    assert error.value.code == "POLICY_TRANSFER_INTEGRITY_INVALID"


@pytest.mark.parametrize("corruption", ["weight", "extra", "symlink", "manifest", "sidecar"])
def test_changed_or_uncommitted_target_cannot_start(tmp_path, corruption):
    checkpoint, package = _transfer(tmp_path)
    if corruption == "weight":
        (checkpoint / "params/weights").write_bytes(b"corrupt")
    elif corruption == "extra":
        (checkpoint / "unexpected").write_text("extra file")
    elif corruption == "symlink":
        (checkpoint / "unexpected").symlink_to(tmp_path / "checkpoint/params/weights")
    elif corruption == "manifest":
        package["manifest"]["files"][0]["size"] += 1
    else:
        (checkpoint / "termitech-policy.json").write_text("{}")
    with pytest.raises(AgentError) as error:
        _portable({"roots": [str(tmp_path)]}, checkpoint, {"portable_policy": package})
    assert error.value.code == "POLICY_TRANSFER_INTEGRITY_INVALID"


@pytest.mark.parametrize("unrecorded_checkpoint", (False, True))
def test_export_command_reports_durable_async_result_without_loading_training(monkeypatch, agent_fixture, unrecorded_checkpoint):
    import vla_agent.policy_export as exporter
    config, root, code = agent_fixture["config"], agent_fixture["root"], agent_fixture["repo"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    package = code / "src/openpi"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("# fixture OpenPI\n")
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    frozen = {"frozen_request": {}, "generated_sha256": "a" * 64}
    warnings = [{"code": "POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH",
        "message": "权重目录不属于所选训练的已记录产物；将继续使用该训练配置，请确认模型结构和归一化参数一致。"}]
    if unrecorded_checkpoint:
        frozen["source_warnings"] = warnings
    monkeypatch.setattr(exporter, "_frozen", lambda *_: frozen)
    worker = root / "export-worker.py"
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-must-not-be-used')
    monkeypatch.setenv('JAX_PLATFORMS', 'cuda')
    monkeypatch.setattr('vla_agent.metrics.gpu_inventory', lambda: pytest.fail('Export must not require an idle GPU'))
    worker.write_text("import json,sys,time,os\nfrom pathlib import Path\nassert os.environ['CUDA_VISIBLE_DEVICES']==''\nassert os.environ['JAX_PLATFORMS']==os.environ['JAX_PLATFORM_NAME']=='cpu'\np=Path(sys.argv[1]);time.sleep(.15)\n(p.parent/'result.json').write_text(json.dumps({'state':'READY','portable_config':{'fixture':True},'portable_digest':'a'*64}))\n")
    monkeypatch.setattr(exporter, "WORKER", worker)
    payload = {"export_id": "export-1", "profile_id": "smoke-profile", "checkpoint_path": str(checkpoint), "source_training": {"run_id": "run-1"}}
    assert handle_export_command(config, "POLICY_EXPORT", payload) == ({"state": "RUNNING", "warnings": warnings}
            if unrecorded_checkpoint else {"state": "RUNNING"})
    directory = Path(config["state_dir"]) / "policy-exports/export-1"
    identity = json.loads((directory / "identity.json").read_text())
    assert handle_export_command(config, "POLICY_EXPORT", payload)["state"] == "RUNNING"
    assert json.loads((directory / "identity.json").read_text()) == identity
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        result = handle_export_command(config, "POLICY_EXPORT_STATUS", payload)
        if result["state"] == "READY":
            break
        time.sleep(.03)
    assert result["state"] == "READY"
    assert result["portable_config"] == {"fixture": True}
    if unrecorded_checkpoint:
        assert result["warnings"] == warnings
        assert json.loads((directory / "manifest.json").read_text())["source_warnings"] == warnings
    else:
        assert "warnings" not in result
    assert handle_export_command(config, "POLICY_EXPORT", payload) == result
    assert not list(checkpoint.iterdir())  # Source checkpoint is never edited.
    assert not (Path(config["state_dir"]) / "attempts").exists()


def test_export_input_safety_and_missing_process_are_explicit(monkeypatch, agent_fixture):
    config = agent_fixture["config"]
    rejected = execute_export_command(config, "POLICY_EXPORT", {"export_id": "../escape"})
    assert rejected["state"] == "FAILED" and rejected["error"]["code"] == "INVALID_IDENTIFIER"
    directory = Path(config["state_dir"]) / "policy-exports/interrupted"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text("{}")
    result = handle_export_command(config, "POLICY_EXPORT_STATUS", {"export_id": "interrupted"})
    assert result["state"] == "FAILED" and result["error"]["code"] == "POLICY_EXPORT_INTERRUPTED"


def test_agent_dispatches_export_receipts_durably(monkeypatch, agent_fixture):
    import vla_agent.policy_export as exporter
    monkeypatch.setattr(exporter, "execute_export_command", lambda *_: {"state": "RUNNING"})
    agent = Agent(agent_fixture["config"])
    try:
        command = {"id": "export-command", "type": "POLICY_EXPORT", "payload": {"export_id": "export-1"}}
        result = agent.handle_command(command)
        assert result == {"status": "APPLIED", "result": {"state": "RUNNING"}}
        assert agent.handle_command(command) == result
    finally:
        agent.close()


def test_export_worker_detects_source_edit_before_importing_resolver(monkeypatch, agent_fixture):
    import vla_agent.policy_export_worker as worker
    import vla_adapters.openpi_entry
    import vla_adapters.generated_config
    root = agent_fixture["repo"]
    directory = root / "src/openpi/training"
    directory.mkdir(parents=True)
    source = directory / "config.py"
    source.write_text("# stable\n")
    module = ModuleType("openpi.training.config")
    module.__file__ = str(source)
    training = ModuleType("openpi.training")
    training.config = module
    openpi = ModuleType("openpi")
    openpi.training = training
    monkeypatch.setitem(sys.modules, "openpi", openpi)
    monkeypatch.setitem(sys.modules, "openpi.training", training)
    monkeypatch.setitem(sys.modules, "openpi.training.config", module)
    baseline = _code_files(root)
    source.write_text("# changed\n")
    monkeypatch.setattr(vla_adapters.openpi_entry, "prepare_config", lambda *_: pytest.fail("Changed source must be rejected before resolver execution"))
    manifest = {"adapter_root": str(root), "code_dir": str(root), "code_files_baseline": baseline,
                "frozen_request": {}, "output_dir": str(agent_fixture["root"]), "generated_sha256": "a" * 64}
    with pytest.raises(ValueError, match="POLICY_EXPORT_CODE_CHANGED"):
        worker.export(manifest)


@pytest.mark.parametrize("execution_id", (None, "execution-hashing"))
def test_stop_during_large_checkpoint_hash_never_launches_a_process(monkeypatch, agent_fixture, execution_id):
    config = agent_fixture["config"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    checkpoint, package = _transfer(agent_fixture["root"])
    entered, release = threading.Event(), threading.Event()
    results = []
    def hashing(*_args):
        entered.set()
        assert release.wait(5)
        return {"portable_policy": {"config": package["config"], "digest": package["digest"]}}
    monkeypatch.setattr("vla_agent.policy_deployment._portable", hashing)
    monkeypatch.setattr("vla_agent.policy_deployment._launch", lambda *_: pytest.fail("A stopped validation must never spawn a model process"))
    payload = {"deployment_id": "stop-during-hash", "profile_id": "smoke-profile", "checkpoint_path": str(checkpoint),
               "config_name": "pi05", "port": 8000, "portable_policy": package}
    if execution_id:
        payload["execution_id"] = execution_id
    thread = threading.Thread(target=lambda: results.append(handle_policy_command(config, "POLICY_DEPLOY_START", payload)))
    thread.start()
    try:
        assert entered.wait(3)
        started = time.monotonic()
        stopped = handle_policy_command(config, "POLICY_DEPLOY_STOP", {
            key: payload[key] for key in ("deployment_id", "execution_id") if key in payload})
        assert stopped["state"] == "STOPPED" and time.monotonic() - started < 1
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert results[0]["state"] == "STOPPED"
    directory = Path(config["state_dir"]) / "policy-deployments" / (execution_id or payload["deployment_id"])
    assert (directory / "stopped.json").exists() and not (directory / "manifest.json").exists()


def test_cancel_before_export_prevents_late_queued_launch(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    payload = {"export_id": "cancelled-before-start"}
    monkeypatch.setattr("vla_agent.policy_export._frozen", lambda *_: pytest.fail("Cancelled export cannot restore training configuration"))
    result = handle_export_command(config, "POLICY_EXPORT_CANCEL", payload)
    assert result["error"]["code"] == "POLICY_EXPORT_CANCELLED"
    assert handle_export_command(config, "POLICY_EXPORT", payload) == result
    assert handle_export_command(config, "POLICY_EXPORT_STATUS", payload) == result
    directory = Path(config["state_dir"]) / "policy-exports" / payload["export_id"]
    assert not (directory / "manifest.json").exists()


def test_cancel_during_source_snapshot_prevents_export_worker_launch(agent_fixture, monkeypatch):
    import vla_agent.policy_export as exporter
    config = agent_fixture["config"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    code = agent_fixture["repo"]
    (code / "src/openpi").mkdir(parents=True)
    (code / "src/openpi/__init__.py").write_text("# source\n")
    checkpoint = agent_fixture["root"] / "checkpoint"
    checkpoint.mkdir()
    entered, release = threading.Event(), threading.Event()
    def frozen(*_args):
        entered.set()
        assert release.wait(5)
        return {"frozen_request": {}, "generated_sha256": "a" * 64}
    monkeypatch.setattr(exporter, "_frozen", frozen)
    worker = agent_fixture["root"] / "forbidden-worker.py"
    marker = agent_fixture["root"] / "forbidden-launch"
    worker.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    monkeypatch.setattr(exporter, "WORKER", worker)
    payload = {"export_id": "cancel-during-snapshot", "profile_id": "smoke-profile", "checkpoint_path": str(checkpoint), "source_training": {"run_id": "source"}}
    results = []
    thread = threading.Thread(target=lambda: results.append(handle_export_command(config, "POLICY_EXPORT", payload)))
    thread.start()
    try:
        assert entered.wait(3)
        cancelled = handle_export_command(config, "POLICY_EXPORT_CANCEL", payload)
        assert cancelled["error"]["code"] == "POLICY_EXPORT_CANCELLED"
    finally:
        release.set()
        thread.join(5)
    assert results[0]["error"]["code"] == "POLICY_EXPORT_CANCELLED"
    directory = Path(config["state_dir"]) / "policy-exports" / payload["export_id"]
    assert not marker.exists() and not (directory / "manifest.json").exists()


def test_cancel_stops_only_owned_export_process(agent_fixture, monkeypatch):
    import vla_agent.policy_export as exporter
    from vla_agent.safety import identity_matches
    config = agent_fixture["config"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    code = agent_fixture["repo"]
    (code / "src/openpi").mkdir(parents=True)
    (code / "src/openpi/__init__.py").write_text("# source\n")
    checkpoint = agent_fixture["root"] / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setattr(exporter, "_frozen", lambda *_: {"frozen_request": {}, "generated_sha256": "a" * 64})
    worker = agent_fixture["root"] / "slow-export.py"
    worker.write_text("import time\ntime.sleep(30)\n")
    monkeypatch.setattr(exporter, "WORKER", worker)
    payload = {"export_id": "cancel-running", "profile_id": "smoke-profile", "checkpoint_path": str(checkpoint), "source_training": {"run_id": "source"}}
    handle_export_command(config, "POLICY_EXPORT", payload)
    directory = Path(config["state_dir"]) / "policy-exports" / payload["export_id"]
    identity = json.loads((directory / "identity.json").read_text())
    assert identity_matches(identity)
    try:
        cancelled = handle_export_command(config, "POLICY_EXPORT_CANCEL", payload)
        assert cancelled["error"]["code"] == "POLICY_EXPORT_CANCELLED"
        deadline = time.monotonic() + 3
        while identity_matches(identity) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not identity_matches(identity)
        # A fake reused PID must not receive a signal on a subsequent cancel.
        fake = {"pid": __import__("os").getpid(), "created_at": 1}
        (directory / "identity.json").write_text(json.dumps(fake))
        monkeypatch.setattr(exporter, "signal_owned_group", lambda *_: pytest.fail("Reused identity must never be signalled"))
        assert handle_export_command(config, "POLICY_EXPORT_CANCEL", payload) == cancelled
    finally:
        for child in list(exporter.CHILDREN.values()):
            if child.pid == identity["pid"]:
                child.wait(timeout=3)


def test_heartbeat_advertises_versioned_portable_and_cancel_support(agent_fixture, monkeypatch):
    agent = Agent(agent_fixture["config"])
    captured = []
    agent.credentials = {"node_id": "fixture-node", "token": "fixture-token"}
    monkeypatch.setattr("vla_agent.main.gpu_inventory", lambda: ([], {}))
    monkeypatch.setattr("vla_agent.main.system_metrics", lambda *_: {})
    monkeypatch.setattr(agent.runtime_monitor, "snapshot", lambda *_: [])
    monkeypatch.setattr(agent, "api", lambda method, path, body: captured.append(body) or {})
    try:
        agent.heartbeat()
        capabilities = captured[0]["metrics"]["capabilities"]
        assert "POLICY_DEPLOY_PORTABLE_V1" in capabilities
        assert "POLICY_EXPORT_CANCEL" in capabilities
        assert "POLICY_DEPLOY_START" in capabilities
        assert all(isinstance(value, str) for value in capabilities)
    finally:
        agent.close()


@pytest.mark.parametrize("execution_id", (None, "portable-execution"))
def test_portable_lifecycle_receipts_keep_exact_configuration_and_checkpoint_witnesses(agent_fixture, monkeypatch, execution_id):
    import vla_agent.policy_deployment as deployment
    config = agent_fixture["config"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    identifier = "portable-receipts"
    directory = Path(config["state_dir"]) / "policy-deployments" / (execution_id or identifier)
    directory.mkdir(parents=True)
    package = {"config": {"schema_version": 1}, "digest": "a" * 64}
    manifest = {"deployment_id": identifier, "python": sys.executable, "port": 8000,
        "portable_policy": package, "checkpoint_manifest_digest": "b" * 64}
    if execution_id:
        manifest["execution_id"] = execution_id
    deployment._record(directory / "manifest.json", manifest)
    alive = [True]
    identity = {"pid": 12345, "created_at": 1}
    monkeypatch.setattr(deployment, "_identity", lambda _directory, mode="serve": identity if mode == "serve" else None)
    monkeypatch.setattr(deployment, "identity_matches", lambda value: value is not None and alive[0])
    monkeypatch.setattr(deployment, "_healthy", lambda *_: True)
    monkeypatch.setattr(deployment, "_launch", lambda *_: None)
    def stopped(*_args):
        alive[0] = False
        return True
    monkeypatch.setattr(deployment, "signal_owned_group", stopped)
    payload = {"deployment_id": identifier, "profile_id": "smoke-profile"}
    if execution_id:
        payload["execution_id"] = execution_id
    receipts = [deployment._status(directory, health=False)]
    receipts.append(handle_policy_command(config, "POLICY_DEPLOY_START", payload))
    receipts.append(handle_policy_command(config, "POLICY_DEPLOY_STATUS", payload))
    observation = agent_fixture["root"] / "observation.json"
    observation.write_text("{}")
    receipts.append(handle_policy_command(config, "POLICY_DEPLOY_TEST", {**payload, "observation_path": str(observation)}))
    receipts.append(handle_policy_command(config, "POLICY_DEPLOY_STOP", payload))
    assert [value["state"] for value in receipts] == ["STARTING", "RUNNING", "RUNNING", "RUNNING", "STOPPED"]
    (directory / "stopped.json").unlink()
    receipts.append(deployment._status(directory))
    assert receipts[-1]["state"] == "FAILED"
    for receipt in receipts:
        assert receipt["portable_digest"] == package["digest"]
        assert receipt["checkpoint_manifest_digest"] == "b" * 64
        if execution_id:
            assert receipt["execution_id"] == execution_id
