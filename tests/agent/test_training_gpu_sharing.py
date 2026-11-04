"""Training coexistence checks without CUDA, model loading or training steps."""
from copy import deepcopy
from decimal import Decimal
import fcntl
import hashlib
import json
from pathlib import Path
import sys

import pytest

from vla_agent import metrics, output_isolation, preflight, runner
from vla_agent.metrics import check_gpus, check_training_gpus
from vla_agent.safety import AgentError
from vla_agent.snapshots import capture_code, compare_code
from vla_agent.state import State
from vla_agent.training_gpu import shared_training_environment, validate_training_gpu_policy
from tests.agent.conftest import bind_config_confirmation
from tests.agent.test_automatic_preflight import probe_fixture, write_probe_response


POLICY = {"mode": "shared", "memory_mb": 40 * 1024}


def gpu(uuid="GPU-selected", **changes):
    return {"uuid": uuid, "memory_total_mb": 81920, "memory_used_mb": 19430,
            "memory_free_mb": 62490, "utilization_pct": 0,
            "processes": [{"pid": 551281, "kind": "compute", "ownership": "external", "memory_mb": 19430}],
            **changes}


def test_explicit_training_share_allows_external_compute_without_changing_legacy_admission():
    row = gpu()
    observed = ([row], {"available": True})
    with pytest.raises(AgentError, match="existing process"):
        check_gpus([row["uuid"]], inventory=observed)
    assert check_training_gpus([row["uuid"]], POLICY, inventory=observed) == [row]
    returned = check_training_gpus([row["uuid"]], POLICY, inventory=observed)
    returned[0]["processes"].clear()
    assert row["processes"]  # Snapshot evidence must not alias live telemetry.


@pytest.mark.parametrize("changes,code", [
    ({"memory_free_mb": 41983}, "GPU_MEMORY_INSUFFICIENT"),
    ({"memory_free_mb": None}, "GPU_TELEMETRY_UNAVAILABLE"),
    ({"processes": None}, "GPU_TELEMETRY_UNAVAILABLE"),
    ({"processes": [{"pid": 4, "kind": "unknown"}]}, "GPU_TELEMETRY_UNAVAILABLE"),
    ({"processes": [{"pid": 4, "kind": "compute", "ownership": "platform"}]}, "GPU_BUSY"),
])
def test_shared_training_rejects_insufficient_unknown_or_platform_occupied_devices(changes, code):
    with pytest.raises(AgentError) as exc:
        check_training_gpus(["GPU-selected"], POLICY, inventory=([gpu(**changes)], {"available": True}))
    assert exc.value.code == code
    boundary = gpu(memory_free_mb=41984)
    assert check_training_gpus(["GPU-selected"], POLICY, inventory=([boundary], {"available": True})) == [boundary]


def test_exact_assigned_order_is_preserved_and_no_fallback_is_allowed():
    a, b = gpu("GPU-a"), gpu("GPU-b")
    observed = ([a, b], {"available": True})
    assert [row["uuid"] for row in check_training_gpus(["GPU-b", "GPU-a"], POLICY, inventory=observed)] == ["GPU-b", "GPU-a"]
    with pytest.raises(AgentError) as exc:
        check_training_gpus(["GPU-absent"], POLICY, inventory=observed)
    assert exc.value.code == "GPU_UNAVAILABLE"


def test_shared_child_environment_uses_a_conservative_fraction_and_removes_conflicting_overrides():
    inherited = {"CUDA_VISIBLE_DEVICES": "GPU-selected,GPU-other", "CUSTOM_NON_MEMORY_SETTING": "keep",
                 "XLA_CLIENT_MEM_FRACTION": ".95", "XLA_PYTHON_CLIENT_MEM_FRACTION": ".99",
                 "XLA_PYTHON_CLIENT_ALLOCATOR": "platform", "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
                 "TF_FORCE_UNIFIED_MEMORY": "true", "TF_GPU_ALLOCATOR": "cuda_malloc_async",
                 "JAX_PJRT_CLIENT_CREATE_OPTIONS": "memory_fraction:1.2"}
    devices = [gpu(), gpu("GPU-other", memory_total_mb=65536, memory_free_mb=46106)]
    actual = shared_training_environment(inherited, POLICY, devices)
    assert actual["XLA_CLIENT_MEM_FRACTION"] == "0.500000"
    assert actual["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc" and actual["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert actual["CUSTOM_NON_MEMORY_SETTING"] == "keep"
    assert not {"XLA_PYTHON_CLIENT_MEM_FRACTION", "TF_FORCE_UNIFIED_MEMORY", "TF_GPU_ALLOCATOR", "JAX_PJRT_CLIENT_CREATE_OPTIONS"} & actual.keys()
    assert inherited["XLA_PYTHON_CLIENT_MEM_FRACTION"] == ".99"
    awkward = shared_training_environment({}, {"mode": "shared", "memory_mb": 40000}, [gpu(memory_total_mb=81921.33)])
    assert Decimal(awkward["XLA_CLIENT_MEM_FRACTION"]) * Decimal("81921.33") <= 40000
    assert shared_training_environment(inherited, None, None) is inherited


@pytest.mark.parametrize("changes", [
    {"template_id": "openpi-norm", "gpu_count": 0},
    {"template_id": "smoke"}, {"requested_gpu_uuids": None},
    {"requested_gpu_uuids": ["GPU-selected", "GPU-selected"], "gpu_count": 2},
    {"gpu_count": 2},
])
def test_shared_request_requires_openpi_gpus_and_an_explicit_consistent_selection(changes):
    payload = {"template_id": "openpi-sft", "gpu_count": 1, "requested_gpu_uuids": ["GPU-selected"],
               "training_gpu_policy": POLICY, **changes}
    with pytest.raises(AgentError) as exc:
        validate_training_gpu_policy(payload, {"adapter": "openpi-sft"})
    assert exc.value.code == "TRAINING_GPU_POLICY_INVALID"


def test_frozen_start_cannot_drop_budget_or_change_explicit_device_order():
    payload = {"template_id": "openpi-sft", "gpu_count": 2, "gpu_uuids": ["GPU-b", "GPU-a"],
               "requested_gpu_uuids": ["GPU-b", "GPU-a"], "training_gpu_policy": POLICY,
               "code_expectation": {"training_gpu_policy": POLICY, "requested_gpu_uuids": ["GPU-b", "GPU-a"]}}
    assert validate_training_gpu_policy(payload, {"adapter": "openpi-sft"}, require_frozen=True) == POLICY
    for changed in ({"training_gpu_policy": None}, {"training_gpu_policy": {**POLICY, "memory_mb": 30000}},
                    {"gpu_uuids": ["GPU-a", "GPU-b"]}):
        with pytest.raises(AgentError) as exc:
            validate_training_gpu_policy({**payload, **changed}, {"adapter": "openpi-sft"}, require_frozen=True)
        assert exc.value.code == "TRAINING_GPU_POLICY_INVALID"


def test_shared_policy_changes_provenance_identity_without_altering_legacy_snapshots(agent_fixture):
    profile, roots = agent_fixture["config"]["profiles"][0], agent_fixture["config"]["roots"]
    old = capture_code(profile, roots)
    same = capture_code(profile, roots, training_gpu_policy=None)
    assert "training_gpu_policy" not in same
    compare_code(old, same)
    shared = capture_code(profile, roots, training_gpu_policy=POLICY)
    assert shared["training_gpu_policy"] == POLICY and old["runtime_digest"] != shared["runtime_digest"]
    with pytest.raises(AgentError) as exc:
        compare_code(old, shared)
    assert exc.value.code == "CODE_CHANGED_SINCE_PREFLIGHT"


@pytest.fixture
def shared_runner(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    baseline = {**payload["code_expectation"], "training_gpu_policy": POLICY}
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {"training_gpu_policy": POLICY}}
    payload.update(template_id="openpi-sft", recipe="fixture_recipe", gpu_count=1, gpu_uuids=["GPU-selected"],
                   requested_gpu_uuids=["GPU-selected"], training_gpu_policy=POLICY,
                   code_expectation={**baseline, "requested_gpu_uuids": ["GPU-selected"]})
    bind_config_confirmation(payload, automatic)
    state = State(config["state_dir"]);output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *_: (profile, {}))
    monkeypatch.setattr(output_isolation, "git", lambda *_: str(agent_fixture["repo"]).encode())
    def capture(*args, **kwargs):
        if len(args) > 2:
            args[2].mkdir(parents=True, exist_ok=True)
        return dict(baseline)
    monkeypatch.setattr(runner, "capture_code", capture)
    monkeypatch.setattr(runner, "prepare_launch", lambda launch, *args, **kwargs: launch)
    monkeypatch.setattr(runner, "validate_training", lambda *args, **kwargs: automatic)
    monkeypatch.setattr(runner, "check_gpus", lambda *_: pytest.fail("Shared launch must use its own admission, not legacy exclusivity"))
    code = ("import json,os;from pathlib import Path;"
            f"Path({str(output / 'observed-child-env.json')!r}).write_text(json.dumps(dict(os.environ)));"
            "raise SystemExit(7)")
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: {
        "cwd": str(agent_fixture["repo"]), "argv": [sys.executable, "-c", code],
        "env": {"CUDA_VISIBLE_DEVICES": "GPU-selected"}})
    return {**agent_fixture, "state": state, "output": output}


def test_real_cpu_child_receives_last_checked_shared_budget_without_initializing_cuda(shared_runner, monkeypatch):
    monkeypatch.setattr(metrics, "gpu_inventory", lambda: ([gpu()], {"available": True}))
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")
    monkeypatch.setenv("TF_FORCE_UNIFIED_MEMORY", "true")
    fixture = shared_runner
    assert runner.execute(fixture["config"], fixture["payload"]["attempt_id"]) == 0
    env = json.loads((fixture["output"] / "observed-child-env.json").read_text())
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-selected"
    assert env["XLA_CLIENT_MEM_FRACTION"] == "0.500000" and env["XLA_PYTHON_CLIENT_ALLOCATOR"] == "bfc"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert "XLA_PYTHON_CLIENT_MEM_FRACTION" not in env and "TF_FORCE_UNIFIED_MEMORY" not in env
    proof = json.loads((fixture["output"] / "provenance/code-snapshot.json").read_text())["training_gpu_check"]
    assert proof["policy"] == POLICY and proof["gpu_uuids"] == ["GPU-selected"]
    assert fixture["state"].pending_events()[-1]["data"]["error"]["code"] == "PROCESS_EXITED"


def test_shared_runner_rechecks_decreasing_free_memory_just_before_spawn(shared_runner, monkeypatch):
    snapshots = iter([gpu(), gpu(memory_free_mb=30000)])
    monkeypatch.setattr(metrics, "gpu_inventory", lambda: ([next(snapshots)], {"available": True}))
    fixture = shared_runner
    assert runner.execute(fixture["config"], fixture["payload"]["attempt_id"]) == 1
    assert not (fixture["output"] / "observed-child-env.json").exists()
    assert fixture["state"].pending_events()[-1]["data"]["error"]["code"] == "GPU_MEMORY_INSUFFICIENT"


def test_shared_mode_never_bypasses_another_platform_runner_gpu_lock(shared_runner, monkeypatch):
    monkeypatch.setattr(metrics, "gpu_inventory", lambda: pytest.fail("Must stop on the existing GPU file lock first"))
    fixture = shared_runner;locks = Path(fixture["config"]["state_dir"]) / "locks";locks.mkdir(exist_ok=True)
    name = "gpu-" + hashlib.sha256(b"GPU-selected").hexdigest() + ".lock"
    with (locks / name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runner.execute(fixture["config"], fixture["payload"]["attempt_id"]) == 1
    assert not (fixture["output"] / "observed-child-env.json").exists()
    assert fixture["state"].pending_events()[-1]["data"]["error"]["code"] == "GPU_BUSY"


def test_preflight_collects_fresh_shared_memory_evidence_without_a_gpu_child(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    payload.update(template_id="openpi-sft", gpu_count=1, requested_gpu_uuids=["GPU-selected"], training_gpu_policy=POLICY)
    snapshot = {**payload["code_expectation"], "training_gpu_policy": POLICY}
    monkeypatch.setattr(preflight, "validate_local_inputs", lambda *_: (profile, {}))
    monkeypatch.setattr(preflight, "capture_code", lambda *args, **kwargs: dict(snapshot))
    monkeypatch.setattr(preflight, "compare_code", lambda *_: None)
    monkeypatch.setattr(preflight, "build_launch", lambda *args, **kwargs: {"argv": [], "cwd": profile["code_dir"], "env": {}})
    monkeypatch.setattr(preflight, "prepare_launch", lambda launch, *args, **kwargs: launch)
    monkeypatch.setattr(preflight, "validate_training", lambda *args, **kwargs: {"plan": {"training_gpu_policy": POLICY}})
    readings = iter([gpu(memory_free_mb=61000), gpu(memory_free_mb=60000)])
    monkeypatch.setattr(metrics, "gpu_inventory", lambda: ([next(readings)], {"available": True}))
    result = preflight.run_preflight(config, payload)
    assert "error" not in result
    proof = result["code_snapshot"]["training_gpu_check"]
    assert proof["policy"] == POLICY and proof["devices"][0]["memory_free_mb"] == 60000
    assert proof["gpu_uuids"] == ["GPU-selected"]


@pytest.mark.parametrize("returned", [None, {"mode": "shared", "memory_mb": 30000}, POLICY])
def test_registered_cpu_probe_must_return_the_exact_confirmed_training_gpu_policy(probe_fixture, returned):
    from vla_agent import training_validation
    fixture = probe_fixture
    spec = json.loads(fixture["launch"]["argv"][-1]);spec["training_gpu_policy"] = POLICY
    fixture["launch"]["argv"][-1] = json.dumps(spec)
    plan = deepcopy(fixture["plan"])
    if returned is not None:
        plan["training_gpu_policy"] = returned
    checks = [{"name": name, "ok": True, "message": "CPU-only fixture"} for name in
              ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
               "model_checkpoint", "training_entrypoint", "training_cache")]
    write_probe_response(fixture, checks, plan)
    if returned == POLICY:
        result = training_validation.validate_training(fixture["config"], fixture["launch"], [])
        assert result["plan"]["training_gpu_policy"] == POLICY
    else:
        with pytest.raises(AgentError) as exc:
            training_validation.validate_training(fixture["config"], fixture["launch"], [])
        assert exc.value.code == "PREFLIGHT_INVALID_RESPONSE"
