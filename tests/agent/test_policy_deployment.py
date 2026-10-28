"""Policy lifecycle tests use isolated subprocesses with a tiny fake runtime."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

from vla_agent.main import Agent
from vla_agent.policy_deployment import handle_policy_command, _frozen, _healthy, _record
from vla_agent.safety import AgentError, process_identity, signal_owned_group


def _write(root, name, source=""):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


@pytest.fixture
def deployment(agent_fixture):
    config = agent_fixture["config"]
    code = agent_fixture["repo"]
    config["profiles"][0]["adapter"] = "openpi-sft"
    checkpoint = agent_fixture["root"] / "checkpoint"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "params" / "weights").write_text("fixture weights")
    (checkpoint / "assets").mkdir()
    (checkpoint / "assets" / "norm_stats.json").write_text("{}")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    # This wire stub intentionally does not claim real WebSocket/GPU coverage.
    # It tests wrapper process ownership and metadata identity independently.
    for name in ("openpi/__init__.py", "openpi/policies/__init__.py", "openpi/training/__init__.py",
                 "openpi/serving/__init__.py", "openpi_client/__init__.py", "websockets/__init__.py", "websockets/sync/__init__.py"):
        _write(code, name)
    _write(code, "openpi/policies/policy_config.py", """
class Policy:
    metadata = {'fixture': True}
    _is_pytorch_model = False
def create_trained_policy(*args, **kwargs): return Policy()
""")
    # Framework/device availability belongs to the fixture, not the host Mac.
    _write(code, "jax.py", "def devices(backend=None): return [object()]\n")
    _write(code, "torch.py", "class cuda:\n    is_available = staticmethod(lambda: True)\n    device_count = staticmethod(lambda: 1)\n")
    _write(code, "openpi/training/config.py", """
from dataclasses import dataclass, field
class Pi0ValueConfig: pass
@dataclass
class Pi0Config:
    action_dim: int = 32
Pi0Config.__module__ = 'openpi.models.pi0_config'
@dataclass
class Config:
    data: object = field(default_factory=object)
    model: object = field(default_factory=object)
def get_config(name): return Config(model=Pi0ValueConfig()) if name == 'value_fixture' else Config(model=Pi0Config())
""")
    _write(code, "openpi/serving/websocket_policy_server.py", """
import asyncio, json, socket
from types import SimpleNamespace
class HealthCheck:
    request = SimpleNamespace(path='/_termitech_health')
    async def send(self, data):
        assert json.loads(data)['_termitech_deployment']
class WebsocketPolicyServer:
    def __init__(self, policy, host, port, metadata):
        self.host, self.port, self.metadata = host, port, metadata
        self._metadata = metadata
    async def _handler(self, websocket):
        raise AssertionError('Health probe reached shared-policy reset handler')
    def serve_forever(self):
        # Exercise the real deployment subclass before accepting probe data.
        asyncio.run(self._handler(HealthCheck()))
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind((self.host,self.port));sock.listen()
            while True:
                client,_ = sock.accept()
                with client: client.sendall(json.dumps(self.metadata).encode())
""")
    _write(code, "websockets/sync/client.py", """
import socket
from urllib.parse import urlsplit
class Connection:
    def __init__(self, uri, **kwargs):
        address=urlsplit(uri);self.sock=socket.create_connection((address.hostname,address.port),timeout=.4)
    def __enter__(self): return self
    def __exit__(self,*args): self.sock.close()
    def recv(self,timeout): self.sock.settimeout(timeout);return self.sock.recv(1048576)
def connect(uri, **kwargs): return Connection(uri,**kwargs)
""")
    _write(code, "openpi_client/msgpack_numpy.py", "import json\ndef unpackb(value): return json.loads(value)\nclass Packer:\n    def pack(self,value): return json.dumps(value).encode()\n")
    payload = {"deployment_id": "deploy-fixture", "profile_id": "smoke-profile", "checkpoint_path": str(checkpoint),
               "config_name": "pi05_fixture", "port": port}
    # Editable installations normally expose OpenPI, this isolated fixture uses
    # PYTHONPATH so the registered interpreter can find the stub runtime.
    import os
    original = os.environ.get("PYTHONPATH")
    os.environ["PYTHONPATH"] = str(code)
    try:
        yield config, payload, config["profiles"][0]
    finally:
        if original is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original
        root = Path(config["state_dir"]) / "policy-deployments"
        for path in root.glob("*/manifest.json"):
            manifest = json.loads(path.read_text())
            if manifest.get("deployment_id") == payload["deployment_id"]:
                stop = dict(payload)
                if manifest.get("execution_id") is not None:
                    stop["execution_id"] = manifest["execution_id"]
                handle_policy_command(config, "POLICY_DEPLOY_STOP", stop)


def _wait(config, payload, state="RUNNING"):
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        result = handle_policy_command(config, "POLICY_DEPLOY_STATUS", payload)
        if result["state"] == state:
            return result
        time.sleep(.05)
    raise AssertionError(result)


def _shared_worker_inventory(deployment, monkeypatch, *, worker_free_mb=22000):
    """The child exercises real locks/probe IPC with an isolated NVML stand-in."""
    from copy import deepcopy
    import vla_agent.policy_deployment as module
    config, original, profile = deployment
    root = Path(config['state_dir']).parent / 'gpu-probe-runtime'
    _write(root, 'vla_agent/__init__.py')
    _write(root, 'vla_agent/safety.py', '''class AgentError(Exception):
    def __init__(self, code, message): self.code=code;super().__init__(message)
    def as_dict(self): return {'code':self.code,'message':str(self)}
''')
    _write(root, 'vla_agent/metrics.py', '''import json
from pathlib import Path
from .safety import AgentError
from vla_protocol.gpu_policy import gpu_deployment_error, normalize_gpu_policy
def check_deployment_gpu(uuid, mode, memory_mb):
    row=json.loads(Path(__file__).with_name('gpu.json').read_text())
    error=gpu_deployment_error(row,mode,memory_mb)
    if error: raise AgentError(error['code'],error['message'])
    return normalize_gpu_policy(mode,memory_mb)
''')
    row = {'uuid': 'GPU-fixture', 'memory_total_mb': 24564, 'memory_used_mb': 1000,
           'memory_free_mb': 22000, 'utilization_pct': 18,
           'processes': [{'pid': 200, 'kind': 'graphics', 'memory_mb': 300}]}
    observed = deepcopy(row)
    observed['memory_free_mb'] = worker_free_mb
    (root / 'vla_agent/gpu.json').write_text(json.dumps(observed))
    monkeypatch.setattr('vla_agent.metrics.gpu_inventory', lambda: ([row], {'available': True}))
    original_launch = module._launch
    def launch(directory, manifest, mode):
        manifest['agent_root'] = str(root)
        _record(directory / 'manifest.json', manifest)
        return original_launch(directory, manifest, mode)
    monkeypatch.setattr(module, '_launch', launch)
    payload = dict(original, gpu_uuid='GPU-fixture', gpu_mode='shared', gpu_memory_mb=16384)
    return config, payload, profile


@pytest.mark.parametrize("backend", ["jax", "pytorch"])
def test_backend_is_frozen_and_attested_by_the_healthy_worker(deployment, backend):
    config, original, profile = deployment
    checkpoint = Path(original["checkpoint_path"])
    if backend == "pytorch":
        (checkpoint / "params" / "weights").unlink()
        (checkpoint / "params").rmdir()
        (checkpoint / "model.safetensors").write_bytes(b"fixture")
        _write(Path(profile["code_dir"]), "openpi/policies/policy_config.py", """
class Policy:
    metadata = {'fixture': True}
    _is_pytorch_model = True
    _pytorch_device = 'cpu'
def create_trained_policy(*args, **kwargs):
    assert kwargs['pytorch_device'] == 'cpu'
    return Policy()
""")
    payload = dict(original, inference_backend=backend)
    started = handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    assert started["resolved_backend"] == backend
    ready = _wait(config, payload)
    assert ready["resolved_backend"] == backend
    directory = Path(config["state_dir"]) / "policy-deployments" / payload["deployment_id"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["inference_backend"] == manifest["resolved_backend"] == backend
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, "POLICY_DEPLOY_START", dict(payload, inference_backend="auto"))
    assert error.value.code == "POLICY_BACKEND_CONFLICT"


def test_backend_mismatch_rejects_start_without_launching_a_worker(deployment):
    config, original, _ = deployment
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, "POLICY_DEPLOY_START", dict(original, inference_backend="pytorch"))
    assert error.value.code == "POLICY_BACKEND_MISMATCH"
    directory = Path(config["state_dir"]) / "policy-deployments" / original["deployment_id"]
    assert not (directory / "manifest.json").exists()


def test_shared_graphics_policy_is_frozen_and_attested_by_running_worker(deployment, monkeypatch):
    config, payload, _ = _shared_worker_inventory(deployment, monkeypatch)
    started = handle_policy_command(config, 'POLICY_DEPLOY_START', payload)
    expected = {'mode': 'shared', 'memory_mb': 16384, 'reserve_mb': 1024}
    assert started['state'] == 'STARTING' and started['gpu_policy'] == expected
    ready = _wait(config, payload)
    assert ready['gpu_policy'] == expected
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, 'POLICY_DEPLOY_START', dict(payload, gpu_memory_mb=8192))
    assert error.value.code == 'POLICY_GPU_POLICY_CONFLICT'
    path = Path(config['state_dir']) / 'policy-deployments' / payload['deployment_id'] / 'manifest.json'
    manifest = json.loads(path.read_text())
    altered = dict(manifest, gpu_policy={**expected, 'memory_mb': 8192})
    try:
        _record(path, altered)
        assert handle_policy_command(config, 'POLICY_DEPLOY_STATUS', payload)['state'] == 'STARTING'
    finally:
        _record(path, manifest)


def test_worker_rechecks_real_free_memory_after_control_snapshot_before_loading(deployment, monkeypatch):
    config, payload, profile = _shared_worker_inventory(deployment, monkeypatch, worker_free_mb=1000)
    marker = Path(config['state_dir']) / 'model-loaded'
    _write(Path(profile['code_dir']), 'openpi/policies/policy_config.py',
           'from pathlib import Path\ndef create_trained_policy(*args, **kwargs):\n    Path(' + repr(str(marker)) + ').touch()\n    raise RuntimeError("model must not be loaded")\n')
    started = handle_policy_command(config, 'POLICY_DEPLOY_START', payload)
    assert started['state'] == 'STARTING'
    failed = _wait(config, payload, 'FAILED')
    assert failed['error']['code'] == 'GPU_MEMORY_INSUFFICIENT'
    assert not marker.exists()


def test_shared_deployment_still_respects_local_runner_gpu_lock(deployment, monkeypatch):
    import fcntl
    config, payload, _ = _shared_worker_inventory(deployment, monkeypatch)
    directory = Path(config['state_dir']) / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ('gpu-' + hashlib.sha256(payload['gpu_uuid'].encode()).hexdigest() + '.lock')
    with path.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle_policy_command(config, 'POLICY_DEPLOY_START', payload)
        failed = _wait(config, payload, 'FAILED')
        assert failed['error']['code'] == 'GPU_BUSY'


def test_real_detached_lifecycle_restart_duplicate_and_loopback_metadata(deployment):
    config, payload, _ = deployment
    started = handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    assert started["state"] == "STARTING"
    assert started["endpoint"].startswith("ws://127.0.0.1:")
    ready = _wait(config, payload)
    assert ready["checks"] == [{"name": "websocket_metadata", "ok": True, "message": "Policy WebSocket metadata verified"}]
    # All state is on disk; a fresh config object recovers the same process.
    recovered = handle_policy_command(json.loads(json.dumps(config)), "POLICY_DEPLOY_START", payload)
    assert recovered["pid"] == ready["pid"]
    stopped = handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)
    assert stopped["state"] == "STOPPED"
    assert _wait(config, payload, "STOPPED")["pid"] is None


def test_logical_deployment_restarts_in_isolated_executions_and_stale_commands_cannot_stop_it(deployment):
    from vla_agent.policy_deployment import active_policy_checkouts, execute_policy_command
    config, legacy, profile = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", legacy)
    first = _wait(config, legacy)
    execution = dict(legacy, execution_id="execution-two")
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, "POLICY_DEPLOY_START", execution)
    assert error.value.code == "POLICY_DEPLOY_EXECUTION_BUSY"
    rejected = execute_policy_command(config, "POLICY_DEPLOY_START", execution)
    assert rejected["error"]["code"] == "POLICY_DEPLOY_EXECUTION_BUSY"
    assert rejected["execution_id"] == execution["execution_id"] and "state" not in rejected
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", legacy)["state"] == "STOPPED"
    assert handle_policy_command(config, "POLICY_DEPLOY_START", execution)["execution_id"] == "execution-two"
    second = _wait(config, execution)
    assert second["pid"] != first["pid"] and second["execution_id"] == "execution-two"
    directory = Path(config["state_dir"]) / "policy-deployments" / execution["execution_id"]
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["deployment_id"] == legacy["deployment_id"]
    assert manifest["execution_id"] == execution["execution_id"]
    assert json.loads((directory / "execution.json").read_text()) == {
        "deployment_id": legacy["deployment_id"], "execution_id": execution["execution_id"]}
    # Recovery after losing the Agent-side identity still uses this execution's
    # exact worker command and manifest path, not the logical deployment path.
    (directory / "identity.json").unlink()
    recovered = handle_policy_command(json.loads(json.dumps(config)), "POLICY_DEPLOY_START", execution)
    assert recovered["pid"] == second["pid"]
    assert active_policy_checkouts(config) == [Path(profile["code_dir"]).resolve()]
    for command in ("POLICY_DEPLOY_STOP", "POLICY_DEPLOY_START", "POLICY_DEPLOY_STATUS"):
        assert handle_policy_command(config, command, legacy)["state"] == "STOPPED"
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, "POLICY_DEPLOY_TEST", legacy)
    assert error.value.code == "POLICY_SERVICE_NOT_READY"
    assert _wait(config, execution)["pid"] == second["pid"]
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", execution)["state"] == "STOPPED"
    restarted = dict(legacy, execution_id="execution-three")
    handle_policy_command(config, "POLICY_DEPLOY_START", restarted)
    third = _wait(config, restarted)
    assert third["pid"] != second["pid"]
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", execution)["state"] == "STOPPED"
    assert handle_policy_command(config, "POLICY_DEPLOY_START", execution)["state"] == "STOPPED"
    assert _wait(config, restarted)["pid"] == third["pid"]


@pytest.mark.parametrize("command", ("POLICY_DEPLOY_START", "POLICY_DEPLOY_STATUS", "POLICY_DEPLOY_STOP", "POLICY_DEPLOY_TEST"))
def test_execution_identifier_cannot_be_reused_by_another_logical_deployment(deployment, command):
    config, payload, _ = deployment
    owner = dict(payload, execution_id="execution-owned")
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", owner)["state"] == "STOPPED"
    stranger = dict(owner, deployment_id="another-deployment")
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, command, stranger)
    assert error.value.code == "POLICY_DEPLOY_EXECUTION_CONFLICT"
    assert handle_policy_command(config, "POLICY_DEPLOY_START", owner)["state"] == "STOPPED"


def test_failed_restart_preflight_reports_new_execution_without_using_the_legacy_manifest(deployment):
    from vla_agent.policy_deployment import execute_policy_command
    config, payload, _ = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    _wait(config, payload)
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)["state"] == "STOPPED"
    request = dict(payload, execution_id="execution-invalid", config_name="invalid;name")
    result = execute_policy_command(config, "POLICY_DEPLOY_START", request)
    assert result["state"] == "FAILED" and result["pid"] is None
    assert result["error"]["code"] == "POLICY_CONFIG_INVALID"
    assert result["execution_id"] == "execution-invalid"
    assert _wait(config, payload, "STOPPED")["pid"] is None


def test_prelaunch_failure_and_stop_cannot_release_another_live_execution(deployment):
    from vla_agent.policy_deployment import execute_policy_command
    config, original, _ = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", original)
    ready = _wait(config, original)
    newer = dict(original, execution_id="execution-never-started", config_name="invalid;name")
    failed = execute_policy_command(config, "POLICY_DEPLOY_START", newer)
    assert failed["error"]["code"] == "POLICY_CONFIG_INVALID"
    assert failed["execution_id"] == newer["execution_id"] and "state" not in failed
    for command in ("POLICY_DEPLOY_STOP", "POLICY_DEPLOY_STATUS", "POLICY_DEPLOY_START"):
        result = execute_policy_command(config, command, newer)
        assert result["error"]["code"] == "POLICY_STOP_UNCONFIRMED"
        assert result["execution_id"] == newer["execution_id"] and "state" not in result
        assert _wait(config, original)["pid"] == ready["pid"]
    directory = Path(config["state_dir"]) / "policy-deployments" / newer["execution_id"]
    assert (directory / "stopped.json").exists() and not (directory / "manifest.json").exists()
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", original)["state"] == "STOPPED"
    assert handle_policy_command(config, "POLICY_DEPLOY_STATUS", newer)["state"] == "STOPPED"
    assert handle_policy_command(config, "POLICY_DEPLOY_START", newer)["state"] == "STOPPED"


def test_wrong_metadata_nonce_and_frozen_service_do_not_report_running(deployment):
    config, payload, _ = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    ready = _wait(config, payload)
    directory = Path(config["state_dir"]) / "policy-deployments" / payload["deployment_id"]
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["nonce"] = "not-the-running-service"
    _record(manifest_path, manifest)
    assert handle_policy_command(config, "POLICY_DEPLOY_STATUS", payload)["state"] == "STARTING"
    identity = process_identity(ready["pid"])
    signal_owned_group(identity, signal.SIGSTOP)
    try:
        started = time.monotonic()
        assert handle_policy_command(config, "POLICY_DEPLOY_STATUS", payload)["state"] != "RUNNING"
        assert time.monotonic() - started < 3
    finally:
        signal_owned_group(identity, signal.SIGCONT)


def test_reused_pid_is_never_signalled_or_treated_as_service(deployment):
    config, payload, _ = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    ready = _wait(config, payload)
    directory = Path(config["state_dir"]) / "policy-deployments" / payload["deployment_id"]
    real_identity = process_identity(ready["pid"])
    stranger = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(30)"], start_new_session=True)
    try:
        foreign = process_identity(stranger.pid)
        foreign["created_at"] -= 100
        _record(directory / "identity.json", foreign)
        assert handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)["state"] == "STOPPED"
        assert stranger.poll() is None
    finally:
        _record(directory / "identity.json", real_identity)
        stranger.terminate();stranger.wait(timeout=3)


@pytest.mark.parametrize("field,value,code", [
    ("deployment_id", "../escape", "INVALID_IDENTIFIER"),
    ("execution_id", "../escape", "INVALID_IDENTIFIER"),
    ("execution_id", "", "INVALID_IDENTIFIER"),
    ("checkpoint_path", "/etc", "PATH_OUTSIDE_ROOTS"),
    ("port", 22, "POLICY_PORT_INVALID"),
    ("config_name", "pi0; touch /tmp/oops", "POLICY_CONFIG_INVALID"),
    ("gpu_uuid", "0;bad", "POLICY_GPU_INVALID"),
])
def test_rejects_untrusted_identifiers_paths_and_execution_arguments(deployment, field, value, code):
    config, payload, _ = deployment
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, "POLICY_DEPLOY_START", dict(payload, **{field: value}))
    assert error.value.code == code


def test_rejects_outside_root_norm_link_and_incomplete_checkpoints(deployment, tmp_path):
    config, payload, _ = deployment
    checkpoint = Path(payload["checkpoint_path"])
    norm = checkpoint / "assets" / "norm_stats.json"
    norm.unlink();norm.symlink_to("/etc/hosts")
    with pytest.raises(AgentError, match="outside configured roots"):
        handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    norm.unlink();norm.write_text("{}")
    (checkpoint / "params" / "weights").unlink();(checkpoint / "params").rmdir()
    with pytest.raises(AgentError) as error:
        handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    assert error.value.code == "POLICY_CHECKPOINT_INVALID"


def test_source_training_cannot_fall_back_to_named_recipe(deployment):
    config, payload, profile = deployment
    with pytest.raises(AgentError) as error:
        _frozen(config, profile, dict(payload, source_training={"run_id": "run", "code_expectation": {}}))
    assert error.value.code == "POLICY_TRAINING_CONFIG_INVALID"


def test_manual_value_model_is_not_served_as_action_policy(deployment):
    config, payload, _ = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", dict(payload, config_name="value_fixture"))
    result = _wait(config, payload, "FAILED")
    assert result["error"]["code"] == "POLICY_VALUE_UNSUPPORTED"


def test_multi_dataset_and_value_only_training_are_rejected(deployment):
    from vla_protocol import canonical_digest
    config, payload, profile = deployment
    for plan, stage in [({"datasets": [{}, {}]}, None), ({"datasets": [{}]}, "value")]:
        expected = {"training_stage": stage, "automatic_preflight": {"plan": plan, "plan_digest": canonical_digest(plan)}}
        with pytest.raises(AgentError) as error:
            _frozen(config, profile, dict(payload, source_training={"code_expectation": expected, "checkpoint_roots": [payload["checkpoint_path"]]}))
        assert error.value.code == "POLICY_CONFIG_UNSUPPORTED"


def test_command_dispatch_is_durable_and_does_not_start_training(deployment, monkeypatch):
    config, payload, _ = deployment
    agent = Agent(config, client=object())
    monkeypatch.setattr("vla_agent.main.run_preflight", lambda *_: pytest.fail("Training preflight invoked"))
    try:
        command = {"id": "policy-start", "type": "POLICY_DEPLOY_START", "payload": payload}
        first = agent.handle_command(command)
        assert first["status"] == "APPLIED"
        assert agent.handle_command(command) == first
        ready = _wait(config, payload)
        assert ready["state"] == "RUNNING"
    finally:
        agent.close()


def test_prelaunch_rejection_is_terminal_and_absent_deployment_stop_is_proven(deployment):
    from vla_agent.policy_deployment import execute_policy_command
    config, payload, _ = deployment
    rejected = execute_policy_command(config, "POLICY_DEPLOY_START", dict(payload, config_name="bad;name"))
    assert rejected["state"] == "FAILED"
    assert rejected["error"]["code"] == "POLICY_CONFIG_INVALID"
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)["state"] == "STOPPED"


def test_expired_start_never_creates_a_process_and_checkpoint_symlink_outside_machine_roots_is_rejected(deployment):
    from vla_agent.policy_deployment import execute_policy_command
    config, payload, profile = deployment
    result = execute_policy_command(config, "POLICY_DEPLOY_START", dict(payload, expires_at="2020-01-01T00:00:00+00:00"))
    assert result["state"] == "FAILED" and result["error"]["code"] == "POLICY_DEPLOY_EXPIRED"
    original_run = Path(payload["checkpoint_path"]).parent / "original-run"
    original_run.mkdir()
    (original_run / "step").symlink_to("/etc", target_is_directory=True)
    with pytest.raises(AgentError) as error:
        _frozen(config, profile, dict(payload, checkpoint_path=str(original_run / "step"),
                source_training={"code_expectation": {}, "checkpoint_roots": [str(original_run)]}))
    assert error.value.code == "PATH_OUTSIDE_ROOTS"


def test_stop_permission_failure_is_explicit_and_keeps_process_ownership(deployment, monkeypatch):
    config, payload, _ = deployment
    handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    ready = _wait(config, payload)
    with monkeypatch.context() as patch:
        patch.setattr("vla_agent.policy_deployment.signal_owned_group", lambda *_: False)
        result = handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)
    assert result["state"] == "STARTING"
    assert result["pid"] == ready["pid"]
    assert result["error"]["code"] == "POLICY_STOP_UNCONFIRMED"
    directory = Path(config["state_dir"]) / "policy-deployments" / payload["deployment_id"]
    assert not (directory / "stopped.json").exists()
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)["state"] == "STOPPED"


@pytest.mark.parametrize("execution_id", (None, "execution-cancelled"))
def test_stop_before_start_persists_tombstone_and_cannot_resurrect_service(deployment, execution_id):
    config, payload, _ = deployment
    if execution_id:
        payload = dict(payload, execution_id=execution_id)
    assert handle_policy_command(config, "POLICY_DEPLOY_STOP", payload)["state"] == "STOPPED"
    assert handle_policy_command(config, "POLICY_DEPLOY_START", payload)["state"] == "STOPPED"
    assert handle_policy_command(config, "POLICY_DEPLOY_STATUS", payload)["state"] == "STOPPED"
    directory = Path(config["state_dir"]) / "policy-deployments" / (execution_id or payload["deployment_id"])
    assert (directory / "stopped.json").exists()
    assert not (directory / "manifest.json").exists()
    assert not (directory / "identity.json").exists()


def test_configuration_digest_ignores_runtime_output_but_detects_model_change(deployment):
    from dataclasses import dataclass
    from vla_adapters.generated_config import generated_training_config
    config, payload, profile = deployment
    @dataclass
    class Configuration:
        checkpoint_base_dir: str
        action_dim: int = 14
    spec = {"code_dir": profile["code_dir"], "recipe": "pi05_fixture", "inputs": {},
            "parameters": {}, "gpu_count": 0, "output_dir": "/old/training-output"}
    first = generated_training_config(Configuration("/old/training-output/checkpoints"), spec)
    second_spec = dict(spec, output_dir="/new/deployment/configuration")
    second = generated_training_config(Configuration("/new/deployment/configuration/checkpoints"), second_spec)
    assert first["sha256"] == second["sha256"]
    changed = generated_training_config(Configuration("/new/deployment/configuration/checkpoints", action_dim=32), second_spec)
    assert changed["sha256"] != first["sha256"]


def test_frozen_request_is_restored_without_exec_and_snapshot_drift_blocks(deployment, monkeypatch):
    from vla_protocol import canonical_digest
    config, payload, profile = deployment
    frozen = {"code_dir": profile["code_dir"], "recipe": "pi05_fixture", "inputs": {},
              "parameters": {}, "gpu_count": 0}
    # Trailing top-level Python is never executed by the AST literal reader.
    source = "FROZEN_REQUEST = " + repr(frozen) + "\nraise RuntimeError('must not execute')\n"
    generated = {"source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
    plan = {"datasets": [{}], "generated_config": generated}
    expected = {"complete": True, "commit_sha": "a" * 40,
                "automatic_preflight": {"plan": plan, "plan_digest": canonical_digest(plan)}}
    monkeypatch.setattr("vla_agent.snapshots.capture_code", lambda *args, **kwargs: dict(expected))
    request = dict(payload, source_training={"code_expectation": expected, "checkpoint_roots": [payload["checkpoint_path"]],
                   "warnings": [{"code": "POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH", "message": "untrusted warning"}]})
    result = _frozen(config, profile, request)
    assert result == {"frozen_request": frozen, "generated_sha256": generated["sha256"]}
    monkeypatch.setattr("vla_agent.snapshots.capture_code", lambda *args, **kwargs: dict(expected, commit_sha="b" * 40))
    with pytest.raises(AgentError) as error:
        _frozen(config, profile, request)
    assert error.value.code == "CODE_CHANGED_SINCE_PREFLIGHT"


@pytest.mark.parametrize("recorded_roots", ("empty", "unrelated", "symlink_to_unrecorded"))
def test_unrecorded_checkpoint_only_warns_and_never_uses_controller_warning_as_ownership(deployment, monkeypatch, recorded_roots):
    from vla_protocol import canonical_digest
    config, payload, profile = deployment
    frozen = {"code_dir": profile["code_dir"], "recipe": "pi05_fixture", "inputs": {}, "parameters": {}, "gpu_count": 0}
    source = "FROZEN_REQUEST = " + repr(frozen) + "\n"
    generated = {"source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
    plan = {"datasets": [{}], "generated_config": generated}
    expected = {"complete": True, "commit_sha": "a" * 40,
                "automatic_preflight": {"plan": plan, "plan_digest": canonical_digest(plan)}}
    monkeypatch.setattr("vla_agent.snapshots.capture_code", lambda *_args, **_kwargs: dict(expected))
    unrelated = Path(payload["checkpoint_path"]).parent / "recorded-run"
    unrelated.mkdir()
    roots = [] if recorded_roots == "empty" else [str(unrelated)]
    if recorded_roots == "symlink_to_unrecorded":
        alias = unrelated / "step"
        alias.symlink_to(payload["checkpoint_path"], target_is_directory=True)
        payload = dict(payload, checkpoint_path=str(alias))
    request = dict(payload, source_training={"run_id": "run-config", "code_expectation": expected,
                   "checkpoint_roots": roots, "warnings": [{"code": "untrusted-controller-value"}]})
    result = _frozen(config, profile, request)
    assert result["frozen_request"] == frozen and result["generated_sha256"] == generated["sha256"]
    assert result["source_run_id"] == "run-config"
    assert result["source_warnings"] == [{"code": "POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH",
        "message": "权重目录不属于所选训练的已记录产物；将继续使用该训练配置，请确认模型结构和归一化参数一致。"}]
    # Advisory ownership cannot bypass the frozen training configuration gates.
    expected["automatic_preflight"]["plan_digest"] = "b" * 64
    with pytest.raises(AgentError) as error:
        _frozen(config, profile, request)
    assert error.value.code == "POLICY_TRAINING_CONFIG_INVALID"


@pytest.mark.parametrize("recorded_roots", (None, "/tmp", ["relative"], [123]))
def test_checkpoint_attribution_requires_well_formed_original_roots(deployment, recorded_roots):
    config, payload, profile = deployment
    with pytest.raises(AgentError) as error:
        _frozen(config, profile, dict(payload, source_training={"code_expectation": {}, "checkpoint_roots": recorded_roots}))
    assert error.value.code == "POLICY_TRAINING_CONFIG_INVALID"


def test_ownership_root_cannot_expand_machine_access_or_trust_controller_warning(deployment):
    config, payload, profile = deployment
    with pytest.raises(AgentError) as error:
        _frozen(config, profile, dict(payload, checkpoint_path="/etc", source_training={"code_expectation": {},
                "checkpoint_roots": ["/"], "warnings": [{"code": "POLICY_DEPLOY_CHECKPOINT_SOURCE_MISMATCH"}]}))
    assert error.value.code == "PATH_OUTSIDE_ROOTS"


def test_single_inference_is_asynchronous_and_writes_finite_action_evidence(deployment):
    config, payload, profile = deployment
    numpy_path = os.environ.get("VLA_TEST_NUMPY_PATH")
    if numpy_path:
        os.environ["PYTHONPATH"] += os.pathsep + numpy_path
    elif importlib.util.find_spec("numpy") is None:
        pytest.skip("Optional inference integration test requires NumPy in the policy interpreter")
    code = Path(profile["code_dir"])
    _write(code, "openpi_client/websocket_client_policy.py", """
import numpy as np
class WebsocketClientPolicy:
    def __init__(self, host, port): pass
    def infer(self, observation):
        assert observation['image'].dtype == np.uint8
        assert observation['state'].dtype == np.float32
        return {'actions': np.zeros((2,3),dtype=np.float32)}
""")
    observation = Path(payload["checkpoint_path"]).parent / "observation.json"
    observation.write_text(json.dumps({"image": {"__array__": [[[0, 1, 255]]], "dtype": "uint8"},
                                       "state": {"__array__": [0, 1, 2], "dtype": "float32"}}))
    handle_policy_command(config, "POLICY_DEPLOY_START", payload)
    _wait(config, payload)
    result = handle_policy_command(config, "POLICY_DEPLOY_TEST", dict(payload, observation_path=str(observation)))
    assert result["test_result"]["status"] == "RUNNING"
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        result = handle_policy_command(config, "POLICY_DEPLOY_STATUS", payload)
        if result["test_result"]["status"] != "RUNNING":
            break
        time.sleep(.05)
    test = result["test_result"]
    assert test["status"] == "SUCCEEDED", test
    assert test["action_shape"] == [2, 3] and test["finite"] is True
    assert test["latency_ms"] >= 0
    assert json.loads(Path(test["output_path"]).read_text())["actions"] == [[0, 0, 0], [0, 0, 0]]
