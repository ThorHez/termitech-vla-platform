"""Read-only path checks shared by preview and actual execution."""
import copy
import json
from pathlib import Path

import pytest

from vla_agent import runner, training_validation
from vla_agent import output_isolation
from vla_agent.output_isolation import validate_output_isolation
from vla_agent.safety import AgentError
from vla_agent.state import State

from tests.agent.test_automatic_preflight import probe_fixture


def check(fixture, output, *, inputs=None, code=None, root=None):
    config = {**fixture["config"], **({"output_root": str(root)} if root is not None else {})}
    return validate_output_isolation(config, str(code or fixture["repo"]), inputs or {}, str(output))


def test_sibling_output_under_shared_parent_is_valid_and_does_not_create_a_directory(agent_fixture):
    fixture = agent_fixture
    output = fixture["root"] / "new-attempt"
    data = fixture["root"] / "dataset"
    data.mkdir()
    result = check(fixture, output, inputs={"dataset": {"path": str(data)}}, root=fixture["root"])
    assert result["repository_root"] == str(fixture["repo"])
    assert result["output_root"] == str(fixture["root"])
    assert not output.exists()


@pytest.mark.parametrize("relative", [".", "checkpoints/new-attempt"])
def test_exact_or_nested_source_outputs_are_rejected_before_creation(agent_fixture, relative):
    output = agent_fixture["repo"] / relative
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, output, root=agent_fixture["repo"])
    assert failure.value.code == "OUTPUT_PATH_CONFLICT"
    assert "Git" in str(failure.value) or "训练项目" in str(failure.value)
    if relative != ".":
        assert not output.exists()


def test_actual_output_cannot_contain_the_source_repository(agent_fixture):
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, agent_fixture["root"], root=agent_fixture["root"])
    assert failure.value.code == "OUTPUT_PATH_CONFLICT"


def test_nested_project_still_excludes_the_entire_git_repository(agent_fixture):
    project = agent_fixture["repo"] / "subproject"
    project.mkdir()
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, agent_fixture["repo"] / "other-output", code=project, root=agent_fixture["repo"])
    assert failure.value.code == "OUTPUT_PATH_CONFLICT"
    assert "Git 仓库" in str(failure.value)


@pytest.mark.parametrize("relation", ["same", "inside", "contains"])
def test_input_assets_cannot_overlap_the_actual_output_in_either_direction(agent_fixture, relation):
    base = Path(agent_fixture["config"]["output_root"])
    output = base / "attempt"
    source = output if relation == "same" else base if relation == "inside" else output / "dataset"
    source.mkdir(parents=True, exist_ok=True)
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, output, inputs={"dataset_0": {"path": str(source)}})
    assert failure.value.code == "OUTPUT_PATH_CONFLICT"
    assert "dataset_0" in str(failure.value)


def test_source_output_root_symlink_is_resolved_to_the_conflicting_repository(agent_fixture):
    link = agent_fixture["root"] / "output-link"
    link.symlink_to(agent_fixture["repo"], target_is_directory=True)
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, link / "attempt", root=link)
    assert failure.value.code == "OUTPUT_PATH_CONFLICT"
    assert not (agent_fixture["repo"] / "attempt").exists()


def test_output_symlink_escape_is_not_a_valid_sibling(agent_fixture):
    base = Path(agent_fixture["config"]["output_root"])
    (base / "project-link").symlink_to(agent_fixture["repo"], target_is_directory=True)
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, base / "project-link" / "attempt")
    assert failure.value.code == "PATH_OUTSIDE_ROOTS"


def test_unreadable_output_prefix_does_not_pass_on_a_permission_guess(agent_fixture, monkeypatch):
    output = Path(agent_fixture["config"]["output_root"]) / "attempt"
    monkeypatch.setattr(output_isolation.os, "access", lambda *args: False)
    with pytest.raises(AgentError) as failure:
        check(agent_fixture, output)
    assert failure.value.code == "OUTPUT_UNAVAILABLE"
    assert not output.exists()


@pytest.mark.parametrize("fault", ["unknown_root", "regular_file", "unknown_asset"])
def test_unavailable_or_non_directory_prefixes_never_pass(agent_fixture, fault):
    fixture = agent_fixture
    base = Path(fixture["config"]["output_root"])
    root, inputs = base, {}
    if fault == "unknown_root":
        root = base / "unknown"
        output = root / "attempt"
    elif fault == "regular_file":
        (base / "file").write_text("data")
        output = base / "file" / "attempt"
    else:
        output = base / "attempt"
        inputs = {"norm": {"path": str(fixture["root"] / "missing-asset")}}
    with pytest.raises(AgentError):
        check(fixture, output, inputs=inputs, root=root)
    assert not output.exists()


def test_preflight_rejects_output_inside_code_before_running_the_native_probe(probe_fixture, monkeypatch):
    fixture = probe_fixture
    config = {**fixture["config"], "output_root": str(fixture["repo"])}
    spec = {**fixture["spec"], "output_dir": str(fixture["repo"] / "new-attempt")}
    argv = list(fixture["launch"]["argv"])
    argv[argv.index("--spec-json") + 1] = json.dumps(spec)
    monkeypatch.setattr(training_validation, "_probe_process", lambda *a, **k: pytest.fail("must reject before executing probe"))
    checks = []
    with pytest.raises(AgentError) as failure:
        training_validation.validate_training(config, {**fixture["launch"], "argv": argv}, checks)
    assert failure.value.code == "OUTPUT_PATH_CONFLICT"
    assert checks == [{"name": "output_isolation", "ok": False, "message": str(failure.value)}]
    assert not Path(spec["output_dir"]).exists()


def test_runner_repeats_the_same_check_even_without_input_assets(agent_fixture, monkeypatch):
    fixture = agent_fixture
    config = {**fixture["config"], "output_root": str(fixture["repo"])}
    payload = copy.deepcopy(fixture["payload"])
    output = fixture["repo"] / payload["attempt_id"]
    state = State(config["state_dir"])
    state.create_attempt(payload, str(output))
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *args: (config["profiles"][0], {}))
    monkeypatch.setattr(output_isolation, "git", lambda *args: str(fixture["repo"]).encode())
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("must not spawn training"))
    assert runner.execute(config, payload["attempt_id"]) == 1
    assert state.attempt(payload["attempt_id"])["state"] == "FAILED"
    assert not output.exists()
    terminal = state.pending_events()[-1]
    assert terminal["data"]["error"]["code"] == "OUTPUT_PATH_CONFLICT"
