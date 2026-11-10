from hashlib import sha256

import pytest

from vla_protocol.training_resume import validate_resume_evidence, validate_resume_source


SOURCE = {"asset_id": "model", "run_id": "run", "attempt_id": "attempt",
          "path": "/machine/output/attempt/checkpoints/recipe/platform/5",
          "output_dir": "/machine/output/attempt", "step": 5}


def test_resume_source_keeps_the_exact_frozen_identity():
    assert validate_resume_source(SOURCE) == SOURCE


def test_resume_source_accepts_committed_step_zero_identity():
    source = {**SOURCE, "step": 0, "path": SOURCE["path"].removesuffix("5") + "0"}
    assert validate_resume_source(source) == source


@pytest.mark.parametrize("changes", [
    {"step": True}, {"step": -1}, {"step": 6}, {"path": "/platform/5"},
    {"path": "/machine/output/attempt/checkpoints/recipe/platform/005"},
    {"path": "/machine/output/attempt/checkpoints/recipe/../platform/5"},
    {"path": "/machine/output/other/checkpoints/recipe/platform/5"},
    {"path": "/machine/output/attempt/checkpoints/recipe/platform/5/params"},
    {"output_dir": "/machine/output/attempt/"}, {"attempt_id": "other"}, {"extra": "field"},
])
def test_resume_source_rejects_unbound_or_ambiguous_checkpoints(changes):
    with pytest.raises(ValueError):
        validate_resume_source({**SOURCE, **changes})


def evidence(source=SOURCE):
    code = f"FROZEN_REQUEST = {dict(resume_source=source)!r}\n"
    return {"resume_source": SOURCE, "resume_checkpoint": {
        "checkpoint_metadata_sha256": "a" * 64, "params_manifest_digest": "b" * 64,
        "train_state_metadata_sha256": "c" * 64, "train_state_manifest_digest": "d" * 64,
        "source_effective_config_sha256": "e" * 64, "source_execution_plan_digest": "f" * 64,
        "commit_timestamp_nsecs": 100, "train_state_bytes": 1000,
    }, "generated_config": {"schema_version": 1, "language": "python", "filename": "training_config.py",
                            "source": code, "sha256": sha256(code.encode()).hexdigest()}}


def test_resume_evidence_binds_full_checkpoint_and_confirmable_config():
    plan = evidence()
    assert validate_resume_evidence(plan, SOURCE) == plan["resume_checkpoint"]
    assert validate_resume_evidence({}, None) is None
    with pytest.raises(ValueError, match="fresh run"):
        validate_resume_evidence(plan, None)


def test_resume_evidence_rejects_reviewed_config_with_a_different_source():
    with pytest.raises(ValueError, match="Reviewed config"):
        validate_resume_evidence(evidence({**SOURCE, "run_id": "other"}), SOURCE)


def test_resume_evidence_rejects_checkpoint_without_train_state():
    plan = evidence()
    plan["resume_checkpoint"]["train_state_bytes"] = 0
    with pytest.raises(ValueError, match="train state"):
        validate_resume_evidence(plan, SOURCE)
