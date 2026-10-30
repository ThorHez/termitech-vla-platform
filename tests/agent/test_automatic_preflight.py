import json
import hashlib
import copy
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from vla_protocol import canonical_digest
from vla_agent import output_isolation, preflight, runner, training_validation
from vla_agent.safety import AgentError
from vla_agent.snapshots import capture_code, compare_code
from vla_agent.state import State

from tests.agent.conftest import bind_config_confirmation


@pytest.fixture
def probe_fixture(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    root = agent_fixture["root"]
    inputs = {}
    for role in ("dataset", "model", "norm"):
        path = root / role
        path.mkdir()
        inputs[role] = {"path": str(path)}
    params = {"steps": 2, "batch_size": 8, "learning_rate": 0.00005}
    spec = {"code_dir": str(agent_fixture["repo"]), "recipe": "test_recipe", "inputs": inputs,
            "parameters": params, "gpu_count": 1, "output_dir": str(Path(config["output_root"]) / "unused-output")}
    plan = {"recipe": spec["recipe"], "recipe_definition_digest": None, "parameters": params, "gpu_count": 1,
            "requested_inputs": {role: asset["path"] for role, asset in inputs.items()},
            "inputs": {role: asset["path"] for role, asset in inputs.items()},
            "entrypoint": str(agent_fixture["repo"] / "scripts" / "train.py"),
            "model": {"path": inputs["model"]["path"], "parameter_bytes": 16},
            "datasets": [{"path": inputs["dataset"]["path"], "norm_sha256": "a" * 64}],
            "storage": {"required_bytes": 64, "reference_checkpoint_bytes": 32, "model_params_bytes": 16}}
    source = "CONFIG = {\"fixture\": True}\n"
    plan["generated_config"] = {"schema_version": 1, "language": "python", "filename": "training_config.py",
                                "source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}
    cache = str(root / "cache/datasets")
    plan["training_cache"] = {"datasets_cache": cache, "paths": {"HF_DATASETS_CACHE": cache},
                              "builders": [{"path": inputs["dataset"]["path"], "cache_dir": cache + "/parquet/default"}]}
    result = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
              "checks": [{"name": name, "ok": True, "message": "verified"} for name in
                         ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache")]}
    marker = root / "probe-environment.json"
    script = root / "fake-wrapper.py"
    script.write_text(
        "import argparse,json,os\nfrom pathlib import Path\n"
        "parser=argparse.ArgumentParser();parser.add_argument('--spec-json');parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()\n"
        "assert args.validate_only\nspec=json.loads(args.spec_json)\nassert spec['allowed_roots']\n"
        f"Path({str(marker)!r}).write_text(json.dumps(dict(os.environ)))\n"
        f"print('VLA_PREFLIGHT:'+json.dumps({result!r}),flush=True)\n"
    )
    launch = {"argv": [sys.executable, str(script), "--spec-json", json.dumps(spec)],
              "cwd": str(agent_fixture["repo"]), "env": {"CUDA_VISIBLE_DEVICES": "GPU-1", "VLA_ADMIN_TOKEN": "launch-secret"}}
    launch = training_validation.prepare_launch(launch, config, inputs)
    # Busy devices are a scheduler concern; a CPU preflight verifies capacity.
    monkeypatch.setattr(training_validation, "gpu_inventory", lambda: (
        [{"uuid": "GPU-1", "processes": [{"pid": 123}]}], {"available": True}))
    return {**agent_fixture, "inputs": inputs, "plan": plan, "launch": launch, "spec": spec, "marker": marker, "script": script}


def test_cpu_probe_strips_credentials_and_returns_bound_plan_without_outputs(probe_fixture, monkeypatch):
    fixture = probe_fixture
    monkeypatch.setenv("VLA_AGENT_TOKEN", "private-token")
    monkeypatch.setenv("VLA_DATABASE_URL", "private-database")
    checks = []
    evidence = training_validation.validate_training(fixture["config"], fixture["launch"], checks)
    assert evidence["plan_digest"] == canonical_digest(fixture["plan"])
    assert {item["name"] for item in checks} == {
        "recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache",
        "output_isolation", "gpu_capacity", "output_storage"}
    environment = json.loads(fixture["marker"].read_text())
    assert not {"VLA_ADMIN_TOKEN", "VLA_AGENT_TOKEN", "VLA_DATABASE_URL"} & environment.keys()
    assert environment["CUDA_VISIBLE_DEVICES"] == "" and environment["JAX_PLATFORMS"] == "cpu"
    assert environment["HF_HUB_OFFLINE"] == "1" and environment["WANDB_MODE"] == "disabled"
    assert not Path(fixture["spec"]["output_dir"]).exists()
    assert list(Path(fixture["config"]["output_root"]).iterdir()) == []


def test_explicit_cache_root_reaches_cpu_probe_and_is_bound_to_its_evidence(probe_fixture, monkeypatch):
    from vla_agent.storage_configuration import cache_environment
    fixture = probe_fixture
    config = fixture['config']
    config['cache_root'] = str(fixture['root'] / 'new-cache')
    environment = cache_environment(config)
    cache = environment['HF_DATASETS_CACHE']
    fixture['plan']['training_cache'] = {'datasets_cache': cache, 'paths': environment,
                                       'builders': [{'path': fixture['inputs']['dataset']['path'],
                                                     'cache_dir': cache + '/parquet/default'}]}
    response = {'schema_version': 1, 'scope': 'configuration_and_local_assets', 'plan': fixture['plan'],
                'checks': [{'name': name, 'ok': True, 'message': 'verified'} for name in
                           ('recipe_configuration', 'data_contract', 'dataset_timestamps', 'dataset_samples', 'norm_contract', 'model_checkpoint',
                            'training_entrypoint', 'training_cache')]}
    fixture['script'].write_text('import os,json\nfrom pathlib import Path\n'
                                f'Path({str(fixture["marker"])!r}).write_text(json.dumps(dict(os.environ)))\n'
                                f'print("VLA_PREFLIGHT:"+json.dumps({response!r}))\n')
    monkeypatch.setenv('HF_DATASETS_CACHE', '/old/inherited/cache')
    launch = training_validation.prepare_launch(fixture['launch'], config, fixture['inputs'])
    assert all(launch['env'][key] == value for key, value in environment.items())
    evidence = training_validation.validate_training(config, launch, [])
    child = json.loads(fixture['marker'].read_text())
    assert all(child[key] == value for key, value in environment.items())
    assert evidence['storage_configuration'] == {'output_root': config['output_root'], 'cache_root': config['cache_root']}
    # A wrapper cannot claim it adopted the root while reporting the old cache.
    response['plan']['training_cache']['paths'] = {'HF_DATASETS_CACHE': cache}
    fixture['script'].write_text('import json\nprint("VLA_PREFLIGHT:"+json.dumps(' + repr(response) + '))\n')
    with pytest.raises(AgentError, match='training cache evidence differs'):
        training_validation.validate_training(config, launch, [])


def test_output_directory_change_invalidates_the_frozen_preflight_even_with_identical_inputs(agent_fixture):
    baseline = agent_fixture['payload']['code_expectation']
    plan = {'unchanged': True}
    evidence = {'schema_version': 1, 'scope': 'configuration_and_local_assets', 'plan': plan,
                'plan_digest': canonical_digest(plan), 'storage_configuration': {
                    'output_root': agent_fixture['config']['output_root'], 'cache_root': None}}
    expected = {**baseline, 'automatic_preflight': evidence}
    changed = copy.deepcopy(expected)
    changed['automatic_preflight']['storage_configuration']['output_root'] += '-different-disk'
    with pytest.raises(AgentError) as error:
        compare_code(expected, changed)
    assert error.value.code == 'ASSETS_CHANGED_SINCE_PREFLIGHT'


def test_output_space_and_gpu_inventory_fail_with_specific_checks(probe_fixture, monkeypatch):
    fixture = probe_fixture
    monkeypatch.setattr(training_validation.shutil, "disk_usage", lambda _: SimpleNamespace(free=63))
    checks = []
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(fixture["config"], fixture["launch"], checks)
    assert error.value.code == "OUTPUT_SPACE_INSUFFICIENT"
    assert checks[-1]["name"] == "output_storage" and checks[-1]["ok"] is False
    assert list(Path(fixture["config"]["output_root"]).iterdir()) == []
    monkeypatch.setattr(training_validation, "gpu_inventory", lambda: ([], {"available": False, "reason": "no NVML"}))
    checks = []
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(fixture["config"], fixture["launch"], checks)
    assert error.value.code == "GPU_UNAVAILABLE"
    assert checks[-1]["name"] == "gpu_capacity" and checks[-1]["ok"] is False


def test_unknown_assigned_gpu_does_not_pass_capacity(probe_fixture):
    checks = []
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(probe_fixture["config"], probe_fixture["launch"], checks, assigned=["GPU-other"])
    assert error.value.code == "GPU_UNAVAILABLE"


def test_wrapper_detailed_failure_is_preserved(probe_fixture):
    fixture = probe_fixture
    response = {"schema_version": 1, "scope": "configuration_and_local_assets",
                "checks": [{"name": "norm_contract", "ok": False, "message": "norm dimension mismatch"}],
                "error": {"code": "NORM_CONTRACT", "message": "norm dimension mismatch"}}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\nraise SystemExit(1)\n")
    checks = []
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(fixture["config"], fixture["launch"], checks)
    assert error.value.code == "NORM_CONTRACT"
    assert checks[0]["name"] == "output_isolation" and checks[0]["ok"] is True
    assert checks[1:] == response["checks"]


def fps_warning():
    return {"name": "dataset_fps", "ok": True, "severity": "warning", "code": "DATASET_FPS_MISMATCH",
            "message": "Dataset registration FPS 30 differs from metadata FPS 60"}


def write_probe_response(fixture, checks, plan=None):
    response = {"schema_version": 1, "scope": "configuration_and_local_assets",
                "plan": fixture["plan"] if plan is None else plan, "checks": checks}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\n")


@pytest.mark.parametrize("required_check", ["dataset_timestamps", "dataset_samples"])
def test_wrapper_cannot_claim_success_without_real_dataset_check(probe_fixture, required_check):
    fixture = probe_fixture
    checks = [{"name": name, "ok": True, "message": "verified"} for name in (
        "recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
        "model_checkpoint", "training_entrypoint", "training_cache") if name != required_check]
    write_probe_response(fixture, checks)
    observed = []
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(fixture["config"], fixture["launch"], observed)
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"
    assert observed[0]["name"] == "output_isolation" and observed[0]["ok"] is True
    assert observed[1:] == checks
    assert not Path(fixture["spec"]["output_dir"]).exists()


@pytest.fixture
def composed_probe_fixture(probe_fixture):
    fixture = probe_fixture
    data = fixture["plan"]["datasets"][0]
    data.update(factory="ComposableDataConfig", frames=32,
                composition={"action_horizon": 16, "action_sequence_keys": []})
    data["sample_probe"] = {"schema_version": 1, "status": "verified", "path": data["path"],
                            "checked_samples": 3, "sample_indices": [0, 16, 31],
                            "samples": [{"index": index} for index in (0, 16, 31)],
                            "action_horizon": 16, "action_sequence_keys": []}
    return fixture


def test_wrapper_accepts_task_bound_positive_composed_sample_evidence(composed_probe_fixture):
    fixture = composed_probe_fixture
    checks = [{"name": name, "ok": True, "message": "verified"} for name in (
        "recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
        "model_checkpoint", "training_entrypoint", "training_cache")]
    write_probe_response(fixture, checks)
    result = training_validation.validate_training(fixture["config"], fixture["launch"], [])
    assert result["plan"] == fixture["plan"]


@pytest.mark.parametrize("field,value", [
    ("missing", None), ("schema_version", True), ("schema_version", 2), ("status", "not_applicable"),
    ("path", "/different/dataset"), ("checked_samples", 0), ("checked_samples", True), ("checked_samples", 4),
    ("sample_indices", []), ("sample_indices", [-1, 16, 31]), ("sample_indices", [0, 16, 32]),
    ("sample_indices", [False, 16, 31]), ("sample_indices", [0, 0, 31]),
    ("samples", []), ("samples", [{"index": 0}, {"index": 15}, {"index": 31}]),
    ("action_horizon", 50), ("action_horizon", True), ("action_sequence_keys", ["actions"]),
])
def test_wrapper_rejects_forged_composed_sample_evidence_despite_passing_check(composed_probe_fixture, field, value):
    fixture = composed_probe_fixture
    data = fixture["plan"]["datasets"][0]
    if field == "missing":
        data.pop("sample_probe")
    else:
        data["sample_probe"][field] = value
    checks = [{"name": name, "ok": True, "message": "verified"} for name in (
        "recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
        "model_checkpoint", "training_entrypoint", "training_cache")]
    write_probe_response(fixture, checks)
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(fixture["config"], fixture["launch"], [])
    assert error.value.code == "PREFLIGHT_INVALID_RESPONSE"
    assert not Path(fixture["spec"]["output_dir"]).exists()


@pytest.mark.parametrize("explicit_error", [False, True])
def test_wrapper_real_sample_failure_is_blocking_and_preserves_diagnostics(probe_fixture, explicit_error):
    fixture = probe_fixture
    message = "RGB 图像形状必须与明确 CHW/HWC 布局一致，历史图像须选择时间索引"
    failure = {"name": "dataset_samples", "ok": False, "severity": "error",
               "code": "COMPOSITION_DATA_INVALID", "message": message}
    checks = [fps_warning(), failure]
    response = {"schema_version": 1, "scope": "configuration_and_local_assets",
                "plan": fixture["plan"], "checks": checks}
    if explicit_error:
        response["error"] = {"code": "COMPOSITION_DATA_INVALID", "message": message}
    # Even a zero-exit wrapper cannot authorize a sample that failed its transform.
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\n")
    observed = []
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(fixture["config"], fixture["launch"], observed)
    assert error.value.code == ("COMPOSITION_DATA_INVALID" if explicit_error else "RECIPE_VALIDATION_FAILED")
    assert observed[0]["name"] == "output_isolation" and observed[0]["ok"] is True
    assert observed[1:] == checks
    assert observed[-1]["ok"] is False and observed[-1]["severity"] == "error"
    assert not Path(fixture["spec"]["output_dir"]).exists()


def test_fps_warning_survives_wrapper_validation_and_plan_digest(probe_fixture):
    fixture = probe_fixture
    plan = copy.deepcopy(fixture["plan"])
    warning = fps_warning()
    plan["datasets"][0]["warnings"] = [{"code": warning["code"], "supplied": 30, "metadata": 60,
        "dataset_path": fixture["inputs"]["dataset"]["path"], "message": warning["message"]}]
    wrapper_checks = [{"name": name, "ok": True, "message": "verified"} for name in
                      ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache")]
    write_probe_response(fixture, [*wrapper_checks, warning], plan)
    checks = []
    evidence = training_validation.validate_training(fixture["config"], fixture["launch"], checks)
    assert checks[0]["name"] == "output_isolation" and checks[0]["ok"] is True
    assert checks[1:len(wrapper_checks) + 2] == [*wrapper_checks, warning]
    assert evidence["plan"] == plan
    assert evidence["plan_digest"] == canonical_digest(plan)
    assert evidence["plan_digest"] != canonical_digest(fixture["plan"])


@pytest.mark.parametrize("failure", [
    {"name": "norm_contract", "ok": False, "message": "norm dimension mismatch"},
    {"name": "data_contract", "ok": False, "severity": "warning", "message": "invalid dataset shape"},
])
def test_wrapper_fps_warning_never_overrides_false_check(probe_fixture, failure):
    checks = [fps_warning(), failure]
    write_probe_response(probe_fixture, checks)
    observed = []
    with pytest.raises(AgentError) as error:
        training_validation.validate_training(probe_fixture["config"], probe_fixture["launch"], observed)
    assert error.value.code == "RECIPE_VALIDATION_FAILED"
    assert observed[0]["name"] == "output_isolation" and observed[0]["ok"] is True
    assert observed[1:] == checks


@pytest.mark.parametrize("inner,expected_count", [
    ({"name": "validation", "ok": False, "message": "probe failed"}, 1),
    ({"name": "norm_contract", "ok": False, "message": "probe failed"}, 2),
    ({"name": "validation", "ok": False, "message": "another failure"}, 2),
    ({"name": "validation", "ok": False, "message": "probe failed", "phase": "dataset"}, 2),
])
def test_preflight_only_deduplicates_identical_validation_failure(agent_fixture, monkeypatch, inner, expected_count):
    profile = {**agent_fixture["config"]["profiles"][0], "adapter": "openpi-sft"}
    monkeypatch.setattr(preflight, "validate_local_inputs", lambda *args: (profile, {}))
    monkeypatch.setattr(preflight, "capture_code", lambda *args, **kwargs: dict(agent_fixture["payload"]["code_expectation"]))
    monkeypatch.setattr(preflight, "build_launch", lambda *args, **kwargs: {})
    monkeypatch.setattr(preflight, "prepare_launch", lambda *args, **kwargs: {})

    def fail_probe(config, launch, checks):
        checks.extend([fps_warning(), inner])
        raise AgentError("RECIPE_VALIDATION_FAILED", "probe failed")

    monkeypatch.setattr(preflight, "validate_training", fail_probe)
    result = preflight.run_preflight(agent_fixture["config"], agent_fixture["payload"])
    failures = [check for check in result["checks"] if check["ok"] is False]
    assert len(failures) == expected_count
    assert inner in failures
    assert {"name": "validation", "ok": False, "message": "probe failed"} in failures
    assert fps_warning() in result["checks"]
    assert result["error"]["code"] == "RECIPE_VALIDATION_FAILED"


@pytest.mark.parametrize("mutation", ["missing_check", "missing_evidence", "missing_builder", "different_dataset",
                                    "different_cache_root", "builder_outside_cache", "path_outside_roots",
                                    "relative_cache", "unknown_path_key", "symlink_escape"])
def test_wrapper_requires_complete_cache_evidence_for_every_dataset(probe_fixture, mutation):
    fixture = probe_fixture
    plan = copy.deepcopy(fixture["plan"])
    checks = [{"name": name, "ok": True, "message": "verified"} for name in
              ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache")]
    cache = plan["training_cache"]
    if mutation == "missing_check":
        checks.pop()
    elif mutation == "missing_evidence":
        plan.pop("training_cache")
    elif mutation == "missing_builder":
        cache["builders"] = []
    elif mutation == "different_dataset":
        cache["builders"][0]["path"] = str(fixture["root"] / "different-dataset")
    elif mutation == "different_cache_root":
        cache["paths"]["HF_DATASETS_CACHE"] = str(fixture["root"] / "different-cache")
    elif mutation == "builder_outside_cache":
        cache["builders"][0]["cache_dir"] = str(fixture["root"] / "different-cache")
    elif mutation == "path_outside_roots":
        cache["paths"]["TMPDIR"] = "/var/cache/unmanaged"
    elif mutation == "relative_cache":
        cache["datasets_cache"] = "relative/cache"
    elif mutation == "unknown_path_key":
        cache["paths"]["UNRECOGNIZED_CACHE"] = cache["datasets_cache"]
    else:
        link = fixture["root"] / "cache-link"
        link.symlink_to(fixture["root"].parent, target_is_directory=True)
        cache["paths"]["TMPDIR"] = str(link / "other-directory")
    response = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan, "checks": checks}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\n")
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(fixture["config"], fixture["launch"], [])
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"


def test_wrapper_preserves_specific_training_cache_permission_failure(probe_fixture):
    fixture = probe_fixture
    response = {"schema_version": 1, "scope": "configuration_and_local_assets",
                "checks": [{"name": "training_cache", "ok": False, "message": "existing dataset lock is not writable"}],
                "error": {"code": "TRAINING_CACHE_NOT_WRITABLE", "message": "existing dataset lock is not writable"}}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\nraise SystemExit(1)\n")
    checks = []
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(fixture["config"], fixture["launch"], checks)
    assert failure.value.code == "TRAINING_CACHE_NOT_WRITABLE"
    assert checks[0]["name"] == "output_isolation" and checks[0]["ok"] is True
    assert checks[1:] == response["checks"]


def test_probe_rejects_execution_plan_for_different_requested_entrypoint(probe_fixture):
    fixture = probe_fixture
    launch = fixture["launch"]
    argv = list(launch["argv"])
    index = argv.index("--spec-json")
    spec = json.loads(argv[index + 1])
    spec["entrypoint"] = "scripts/custom_train.py"
    argv[index + 1] = json.dumps(spec)
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(fixture["config"], {**launch, "argv": argv}, [])
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"
    assert "requested training entrypoint" in str(failure.value)


def test_preflight_forwards_requested_entrypoint_to_capture_and_launch(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    payload = {**agent_fixture["payload"], "entrypoint": "scripts/custom_train.py"}
    profile = config["profiles"][0]
    monkeypatch.setattr(preflight, "validate_local_inputs", lambda *_: (profile, {}))
    captured, launched = [], []

    def capture(*args, **kwargs):
        captured.append(kwargs.get("entrypoint"))
        return dict(payload["code_expectation"])

    def launch(*args, **kwargs):
        launched.append(kwargs.get("entrypoint"))
        return {}

    monkeypatch.setattr(preflight, "capture_code", capture)
    monkeypatch.setattr(preflight, "build_launch", launch)
    assert "error" not in preflight.run_preflight(config, payload)
    assert captured == launched == ["scripts/custom_train.py"]


def test_probe_has_bounded_output_and_kills_process_group(tmp_path):
    script = tmp_path / "noisy.py"
    script.write_text("import sys,time\nsys.stdout.write('x'*100000);sys.stdout.flush();time.sleep(30)\n")
    started = time.monotonic()
    with pytest.raises(AgentError) as error:
        training_validation._probe_process([sys.executable, str(script)], str(tmp_path), dict(os.environ), timeout=2, output_limit=1000)
    assert error.value.code == "PREFLIGHT_OUTPUT_LIMIT"
    assert time.monotonic() - started < 5


def test_probe_timeout_kills_descendant_even_if_parent_exits(tmp_path):
    pid_file = tmp_path / "child-pid"
    script = tmp_path / "hang.py"
    script.write_text("import subprocess,sys\nfrom pathlib import Path\n"
                      "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])\n"
                      f"Path({str(pid_file)!r}).write_text(str(child.pid))\n")
    with pytest.raises(AgentError) as error:
        training_validation._probe_process([sys.executable, str(script)], str(tmp_path), dict(os.environ), timeout=0.4)
    assert error.value.code == "PREFLIGHT_TIMEOUT"
    import psutil
    deadline = time.monotonic() + 2
    pid = int(pid_file.read_text())
    while time.monotonic() < deadline:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            # Successful cleanup can reap the process between construction and
            # status lookup; a vanished descendant meets the same assertion.
            break
        time.sleep(0.02)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("preflight left a live descendant")


def test_plan_identity_change_or_tampered_digest_blocks_launch(agent_fixture):
    baseline = agent_fixture["payload"]["code_expectation"]
    old = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {"norm_sha256": "old"}}
    old["plan_digest"] = canonical_digest(old["plan"])
    new = {**old, "plan": {"norm_sha256": "new"}}
    new["plan_digest"] = canonical_digest(new["plan"])
    for changed in (new, {**old, "plan_digest": "forged"}, None):
        with pytest.raises(AgentError) as error:
            compare_code({**baseline, "automatic_preflight": old}, {**baseline, "automatic_preflight": changed})
        assert error.value.code == "ASSETS_CHANGED_SINCE_PREFLIGHT"


def test_preflight_recaptures_code_after_validation(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    profile["adapter"] = "openpi-sft"
    monkeypatch.setattr(preflight, "validate_local_inputs", lambda config, payload: (profile, {}))
    baseline = agent_fixture["payload"]["code_expectation"]
    monkeypatch.setattr(preflight, "build_launch", lambda *args, **kwargs: {})
    monkeypatch.setattr(preflight, "prepare_launch", lambda *args, **kwargs: {})
    monkeypatch.setattr(preflight, "validate_training", lambda *args, **kwargs: {"plan_digest": "digest"})
    snapshots = iter([baseline, {**baseline, "changes_digest": "edited-during-probe"}])
    monkeypatch.setattr(preflight, "capture_code", lambda *args, **kwargs: next(snapshots))
    result = preflight.run_preflight(config, {**agent_fixture["payload"], "gpu_count": 1})
    assert result["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    assert "automatic_preflight" not in result["code_snapshot"]


def test_runner_rechecks_plan_and_never_spawns_changed_assets(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    frozen = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {"norm_sha256": "old"}}
    frozen["plan_digest"] = canonical_digest(frozen["plan"])
    changed = {**frozen, "plan": {"norm_sha256": "new"}}
    changed["plan_digest"] = canonical_digest(changed["plan"])
    baseline = payload["code_expectation"]
    payload["code_expectation"] = {**baseline, "automatic_preflight": frozen}
    bind_config_confirmation(payload, frozen)
    payload["gpu_uuids"] = []
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *args: (profile, {}))
    monkeypatch.setattr(output_isolation, "git", lambda *args: str(agent_fixture["repo"]).encode())
    monkeypatch.setattr(runner, "capture_code", lambda *args, **kwargs: dict(baseline))
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: {"cwd": str(agent_fixture["repo"]), "argv": []})
    monkeypatch.setattr(runner, "prepare_launch", lambda launch, *args: launch)
    monkeypatch.setattr(runner, "validate_training", lambda *args, **kwargs: changed)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("training process started with changed assets"))
    assert runner.execute(config, payload["attempt_id"]) == 1
    terminal = state.pending_events()[-1]
    assert terminal["data"]["error"]["code"] == "ASSETS_CHANGED_SINCE_PREFLIGHT"
    assert terminal["data"]["process_started"] is False


def test_runner_records_automatic_evidence_on_started_and_saved_provenance(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {"norm_sha256": "verified"}}
    automatic["plan_digest"] = canonical_digest(automatic["plan"])
    baseline = payload["code_expectation"]
    payload["code_expectation"] = {**baseline, "automatic_preflight": automatic}
    bind_config_confirmation(payload, automatic)
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *args: (profile, {}))
    monkeypatch.setattr(output_isolation, "git", lambda *args: str(agent_fixture["repo"]).encode())

    def capture(*args, **kwargs):
        if len(args) > 2:
            args[2].mkdir(parents=True, exist_ok=True)
        return dict(baseline)

    monkeypatch.setattr(runner, "capture_code", capture)
    code = ("import json;from pathlib import Path;"
            f"output=Path({str(output)!r});(output/'model').mkdir();"
            "(output/'artifacts.json').write_text(json.dumps([{'kind':'MODEL','name':'test','version':'1','path':str(output/'model')}]))")
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: {
        "cwd": str(agent_fixture["repo"]), "argv": [sys.executable, "-c", code]})
    frozen_digests = []

    def prepare(launch, *args, **kwargs):
        if "expected_plan_digest" in kwargs:
            frozen_digests.append(kwargs["expected_plan_digest"])
        return launch

    monkeypatch.setattr(runner, "prepare_launch", prepare)
    monkeypatch.setattr(runner, "validate_training", lambda *args, **kwargs: automatic)
    assert runner.execute(config, payload["attempt_id"]) == 0
    assert state.attempt(payload["attempt_id"])["state"] == "SUCCEEDED"
    started = next(event for event in state.pending_events(1000) if event["type"] == "STARTED")
    assert started["data"]["code_snapshot"]["automatic_preflight"] == automatic
    assert frozen_digests == [automatic["plan_digest"]]
    for path in (output / "provenance" / "code-snapshot.json", output / "provenance" / "end" / "code-snapshot.json"):
        assert json.loads(path.read_text())["automatic_preflight"] == automatic


@pytest.mark.parametrize("mutation", ["missing", "digest", "syntax", "auto_config"])
def test_wrapper_requires_valid_generated_code_and_exact_automatic_settings(probe_fixture, mutation):
    fixture = probe_fixture
    plan = copy.deepcopy(fixture["plan"])
    if mutation == "missing":
        plan.pop("generated_config")
    elif mutation == "digest":
        plan["generated_config"]["source"] += "# changed without a new digest\n"
    elif mutation == "syntax":
        plan["generated_config"]["source"] = "CONFIG = (\n"
        plan["generated_config"]["sha256"] = hashlib.sha256(plan["generated_config"]["source"].encode()).hexdigest()
    else:
        plan["auto_config"] = {"model_type": "pi05", "finetune_mode": "full", "action_horizon": 50, "action_dim": 32}
    response = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
                "checks": [{"name": name, "ok": True, "message": "verified"} for name in
                           ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache")]}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(response) + "))\n")
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(fixture["config"], fixture["launch"], [])
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"


@pytest.mark.parametrize("mutation,expected", [
    ("missing_code", "CONFIG_CONFIRMATION_REQUIRED"),
    ("invalid_code", "CONFIG_CONFIRMATION_REQUIRED"),
    ("missing_confirmation", "CONFIG_CONFIRMATION_REQUIRED"),
    ("wrong_confirmation", "CONFIG_CONFIRMATION_MISMATCH"),
    ("missing_payload_confirmation", "CONFIG_CONFIRMATION_MISMATCH"),
    ("wrong_payload_confirmation", "CONFIG_CONFIRMATION_MISMATCH"),
])
def test_runner_never_spawns_without_the_exact_user_config_confirmation(agent_fixture, monkeypatch, mutation, expected):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {}}
    bind_config_confirmation(payload, automatic)
    if mutation == "missing_code":
        automatic["plan"].pop("generated_config")
    elif mutation == "invalid_code":
        automatic["plan"]["generated_config"]["source"] += "# tampered\n"
    elif mutation == "missing_confirmation":
        payload["code_expectation"].pop("confirmed_config_sha256")
    elif mutation == "wrong_confirmation":
        payload["code_expectation"]["confirmed_config_sha256"] = "0" * 64
    elif mutation == "missing_payload_confirmation":
        payload.pop("confirmed_config_sha256")
    else:
        payload["confirmed_config_sha256"] = "0" * 64
    state = State(config["state_dir"])
    state.create_attempt(payload, str(Path(config["output_root"]) / payload["attempt_id"]))
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *_: (profile, {}))
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: pytest.fail("unconfirmed config reached launch builder"))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unconfirmed config spawned training"))
    assert runner.execute(config, payload["attempt_id"]) == 1
    final = state.pending_events()[-1]
    assert final["type"] == "FAILED" and final["data"]["process_started"] is False
    assert final["data"]["error"]["code"] == expected
    assert not (Path(config["output_root"]) / payload["attempt_id"]).exists()
