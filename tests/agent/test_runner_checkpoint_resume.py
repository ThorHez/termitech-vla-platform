"""The Agent refuses resume evidence that differs from the requested checkpoint."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.agent.conftest import bind_config_confirmation
from tests.agent.test_automatic_preflight import probe_fixture  # noqa: F401
from vla_agent import runner, training_validation
from vla_agent.safety import AgentError
from vla_agent.state import State
from vla_protocol import canonical_digest


def resume_source(root):
    output = Path(root) / "source-attempt"
    return {"asset_id": "resume-asset", "run_id": "source-run", "attempt_id": output.name,
            "output_dir": str(output), "path": str(output / "checkpoints/sft/platform/5"), "step": 5}


@pytest.mark.parametrize("change", [None, "wrong_source", "wrong_config", "no_train_state"])
def test_preflight_validates_resume_checkpoint_and_reviewed_source(probe_fixture, change):
    fixture = probe_fixture
    source = resume_source(fixture["config"]["output_root"])
    Path(source["path"]).mkdir(parents=True)
    launch = copy.deepcopy(fixture["launch"])
    index = launch["argv"].index("--spec-json") + 1
    spec = json.loads(launch["argv"][index])
    spec["resume_source"] = source
    spec["inputs"]["model"] = {"id": source["asset_id"], "path": source["path"], "kind": "MODEL"}
    spec["requested_inputs"]["model"] = source["path"]
    launch["argv"][index] = json.dumps(spec)
    plan = copy.deepcopy(fixture["plan"])
    plan["inputs"]["model"] = plan["requested_inputs"]["model"] = source["path"]
    plan["resume_source"] = {**source, "asset_id": "wrong"} if change == "wrong_source" else source
    proof = {key: "a" * 64 for key in ("checkpoint_metadata_sha256", "params_manifest_digest",
             "train_state_metadata_sha256", "train_state_manifest_digest", "source_effective_config_sha256",
             "source_execution_plan_digest")}
    proof.update(commit_timestamp_nsecs=1, train_state_bytes=8)
    if change == "no_train_state":
        proof.pop("train_state_manifest_digest")
    plan["resume_checkpoint"] = proof
    request = {} if change == "wrong_config" else {"resume_source": source}
    code = f"FROZEN_REQUEST = {request!r}\n"
    plan["generated_config"].update(source=code, sha256=hashlib.sha256(code.encode()).hexdigest())
    checks = ["recipe_configuration", "data_contract", "dataset_timestamps", "dataset_samples", "norm_contract",
              "model_checkpoint", "training_entrypoint", "training_cache", "resume_checkpoint"]
    response = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": plan,
                "checks": [{"name": name, "ok": True, "message": "verified"} for name in checks]}
    fixture["script"].write_text(f"import json\nprint('VLA_PREFLIGHT:' + json.dumps({response!r}))\n")
    if change is None:
        evidence = training_validation.validate_training(fixture["config"], launch, [])
        assert evidence["plan_digest"] == canonical_digest(plan)
    else:
        with pytest.raises(AgentError) as error:
            training_validation.validate_training(fixture["config"], launch, [])
        assert error.value.code == "PREFLIGHT_INVALID_RESPONSE"


def test_runner_rejects_changed_resume_source_before_creating_output(agent_fixture, monkeypatch):
    config, payload = agent_fixture["config"], agent_fixture["payload"]
    source = resume_source(config["output_root"])
    payload.update(template_id="openpi-sft", resume_source={**source, "asset_id": "changed"})
    automatic = {"schema_version": 1, "scope": "configuration_and_local_assets", "plan": {"resume_source": source}}
    automatic["plan_digest"] = canonical_digest(automatic["plan"])
    bind_config_confirmation(payload, automatic)
    state = State(config["state_dir"])
    output = Path(config["output_root"]) / payload["attempt_id"]
    state.create_attempt(payload, str(output))
    profile = {**config["profiles"][0], "adapter": "openpi-sft"}
    monkeypatch.setattr(runner, "validate_local_inputs", lambda *_: (profile, {}))
    monkeypatch.setattr(runner, "build_launch", lambda *args, **kwargs: pytest.fail("mismatched source reached launch"))
    assert runner.execute(config, payload["attempt_id"]) == 1
    assert not output.exists() and state.attempt(payload["attempt_id"])["child_identity"] is None
    assert state.pending_events()[-1]["data"]["error"]["code"] == "RESUME_SOURCE_MISMATCH"
