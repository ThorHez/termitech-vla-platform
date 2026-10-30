import copy
import json
from pathlib import Path

import pytest

from vla_agent import preflight, runner, training_validation
from vla_agent.safety import AgentError
from vla_agent.snapshots import capture_code, compare_code
from vla_agent.state import State


def bound_inputs(root):
    inputs = {}
    for role, kind in (("model", "MODEL"), ("dataset_0", "DATASET"), ("norm_0", "NORM"),
                       ("dataset_1", "DATASET"), ("norm_1", "NORM")):
        path = root / role
        path.mkdir(exist_ok=True)
        inputs[role] = {"id": role + "-asset", "path": str(path), "kind": kind}
        if kind == "NORM":
            (path / "norm_stats.json").write_text("{}")
    bindings = [{"dataset_asset_id": f"dataset_{i}-asset", "norm_asset_id": f"norm_{i}-asset",
                 "recipe_index": i, "weight": weight} for i, weight in enumerate((1.0, 20.0))]
    return inputs, bindings


def test_agent_resolves_every_explicit_dataset_and_norm_inside_roots(agent_fixture):
    inputs, bindings = bound_inputs(agent_fixture["root"])
    profile, resolved = preflight.validate_local_inputs(agent_fixture["config"], {
        **agent_fixture["payload"], "inputs": inputs, "dataset_bindings": bindings})
    assert profile["id"] == "smoke-profile"
    assert resolved == inputs


@pytest.mark.parametrize("mutation", ["missing", "extra", "swapped_id", "wrong_kind", "escape"])
def test_agent_rejects_missing_mismatched_or_outside_binding_assets(agent_fixture, tmp_path, mutation):
    inputs, bindings = bound_inputs(agent_fixture["root"])
    if mutation == "missing":
        inputs.pop("norm_1")
    elif mutation == "extra":
        inputs["dataset"] = dict(inputs["dataset_0"])
    elif mutation == "swapped_id":
        inputs["norm_0"]["id"] = inputs["norm_1"]["id"]
    elif mutation == "wrong_kind":
        inputs["norm_1"]["kind"] = "MODEL"
    else:
        inputs["dataset_1"]["path"] = str(tmp_path.parent)
    with pytest.raises(AgentError) as failure:
        preflight.validate_local_inputs(agent_fixture["config"], {
            **agent_fixture["payload"], "inputs": inputs, "dataset_bindings": bindings})
    assert failure.value.code == ("PATH_OUTSIDE_ROOTS" if mutation == "escape" else "DATASET_BINDINGS_INVALID")


@pytest.mark.parametrize("field,value", [("dataset_asset_id", "replacement-data"),
                                          ("norm_asset_id", "replacement-norm"),
                                          ("recipe_index", 3), ("weight", 30.0)])
def test_dataset_pairing_processing_and_weights_are_frozen_in_code_evidence(agent_fixture, field, value):
    _, bindings = bound_inputs(agent_fixture["root"])
    config = agent_fixture["config"]
    profile = config["profiles"][0]
    evidence = agent_fixture["root"] / "mix-evidence"
    frozen = capture_code(profile, config["roots"], evidence, dataset_bindings=bindings)
    assert frozen["dataset_bindings"] == bindings
    assert json.loads((evidence / "code-snapshot.json").read_text())["dataset_bindings"] == bindings
    changed = copy.deepcopy(bindings)
    changed[1][field] = value
    with pytest.raises(AgentError, match="dataset_bindings"):
        compare_code(frozen, capture_code(profile, config["roots"], dataset_bindings=changed))
    with pytest.raises(AgentError, match="dataset_bindings"):
        compare_code(frozen, capture_code(profile, config["roots"]))


def test_preflight_forwards_full_binding_list_and_freezes_it(agent_fixture, monkeypatch):
    inputs, bindings = bound_inputs(agent_fixture["root"])
    seen = []

    def launch(*args, **kwargs):
        seen.append((args[2], kwargs["dataset_bindings"]))
        return {}

    monkeypatch.setattr(preflight, "build_launch", launch)
    result = preflight.run_preflight(agent_fixture["config"], {
        **agent_fixture["payload"], "inputs": inputs, "dataset_bindings": bindings})
    assert "error" not in result, result
    assert seen == [(inputs, bindings)]
    assert result["code_snapshot"]["dataset_bindings"] == bindings


def explicit_probe(fixture):
    inputs, bindings = bound_inputs(fixture["root"])
    argv = list(fixture["launch"]["argv"])
    index = argv.index("--spec-json")
    spec = json.loads(argv[index + 1])
    spec.update(inputs=inputs, requested_inputs={role: value["path"] for role, value in inputs.items()},
                dataset_bindings=bindings)
    argv[index + 1] = json.dumps(spec)
    plan = copy.deepcopy(fixture["plan"])
    plan.update(inputs=spec["requested_inputs"], requested_inputs=spec["requested_inputs"], dataset_bindings=bindings,
                weights=[row["weight"] for row in bindings], datasets=[{
                    "path": inputs[f"dataset_{i}"]["path"],
                    "norm_path": str(Path(inputs[f"norm_{i}"]["path"]) / "norm_stats.json"),
                    "norm_sha256": "a" * 64} for i in range(len(bindings))])
    cache = plan["training_cache"]
    cache["builders"] = [{"path": item["path"], "cache_dir": cache["datasets_cache"] + "/parquet/default"}
                         for item in plan["datasets"]]
    return {**fixture["launch"], "argv": argv}, plan


def write_probe_response(fixture, plan):
    # Synthetic wrapper attestations only; this fixture does not decode data.
    result = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
              "checks": [{"name": name, "ok": True, "message": "verified"} for name in
                         ("recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples",
                          "norm_contract", "model_checkpoint", "training_entrypoint", "training_cache")]}
    fixture["script"].write_text("import json\nprint('VLA_PREFLIGHT:'+json.dumps(" + repr(result) + "))\n")


# Reuse the bounded interpreter fixture rather than importing real training dependencies.
from tests.agent.test_automatic_preflight import probe_fixture  # noqa: E402,F401


def test_probe_accepts_complete_ordered_mix_with_per_row_norms(probe_fixture):
    launch, plan = explicit_probe(probe_fixture)
    write_probe_response(probe_fixture, plan)
    actual = training_validation.validate_training(probe_fixture["config"], launch, [])
    assert actual["plan"] == plan
    assert actual["plan"]["weights"] == [1.0, 20.0]


@pytest.mark.parametrize("mutation", ["missing_bindings", "weight", "truncated", "old_weights", "swapped_norm"])
def test_probe_rejects_stale_or_truncated_dataset_mix(probe_fixture, mutation):
    launch, plan = explicit_probe(probe_fixture)
    if mutation == "missing_bindings":
        plan.pop("dataset_bindings")
    elif mutation == "weight":
        plan["dataset_bindings"] = copy.deepcopy(plan["dataset_bindings"])
        plan["dataset_bindings"][1]["weight"] = 50.0
    elif mutation == "truncated":
        plan["datasets"].pop()
    elif mutation == "old_weights":
        plan["weights"] = [1.0, 50.0]
    else:
        plan["datasets"][1]["norm_path"] = plan["datasets"][0]["norm_path"]
    write_probe_response(probe_fixture, plan)
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(probe_fixture["config"], launch, [])
    assert failure.value.code == "PREFLIGHT_INVALID_RESPONSE"


def test_runner_rejects_changed_mix_before_building_or_spawning_training(agent_fixture, monkeypatch):
    config = agent_fixture["config"]
    inputs, bindings = bound_inputs(agent_fixture["root"])
    payload = {**agent_fixture["payload"], "inputs": inputs, "dataset_bindings": copy.deepcopy(bindings)}
    payload["code_expectation"] = capture_code(config["profiles"][0], config["roots"], dataset_bindings=bindings)
    payload["dataset_bindings"][1]["weight"] = 50.0
    state = State(config["state_dir"])
    state.create_attempt(payload, str(Path(config["output_root"]) / payload["attempt_id"]))
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: pytest.fail("changed mix reached launch builder"))
    assert runner.execute(config, payload["attempt_id"]) == 1
    final = state.pending_events()[-1]
    assert final["data"]["error"]["code"] == "CODE_CHANGED_SINCE_PREFLIGHT"
    assert final["data"]["process_started"] is False
