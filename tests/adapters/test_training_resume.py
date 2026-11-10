"""SFT resume plans preserve the source attempt and require full checkpoint evidence."""

import ast
import copy
import dataclasses
import json
from pathlib import Path

import pytest

from test_training_plan import Config, training  # noqa: F401
from vla_adapters import build_launch, openpi_entry, training_plan
from vla_protocol.training_resume import validate_resume_evidence


@dataclasses.dataclass
class ResumeConfig(Config):
    name: str = "single_sft"
    resume_checkpoint: str | None = None


@pytest.fixture
def resume(training, tmp_path):
    base, spec, module, loaders = training
    base = ResumeConfig(data=base.data.datasets[0])
    dataset, norm, model = (spec["inputs"][key] for key in ("dataset_0", "norm_0", "model"))
    source_output = tmp_path / "old-attempt"
    old_spec = {**spec, "recipe": base.name, "dataset_bindings": None, "output_dir": str(source_output),
                "inputs": {"dataset": dataset, "norm": norm, "model": model}}
    entry = Path(old_spec["code_dir"]) / "scripts/train.py"
    entry.write_text(entry.read_text() + "\nPLATFORM_CHECKPOINT_RESUME_VERSION = 1\n")
    original = openpi_entry.resolve_config(base, old_spec, module, loaders)
    original_plan = training_plan.inspect_training_config(original, old_spec, checks=[])

    checkpoint = source_output / "checkpoints" / base.name / "platform" / "5"
    for item in ("params", "train_state"):
        path = checkpoint / item
        (path / "d").mkdir(parents=True)
        (path / "_METADATA").write_text('{"tree_metadata": {}}')
        (path / "manifest.ocdbt").write_bytes(b"tensor manifest")
        (path / "d/shard").write_bytes(b"checkpoint shard")
    (checkpoint / "_CHECKPOINT_METADATA").write_text('{"commit_timestamp_nsecs": 1234}')
    effective = {**openpi_entry.configuration_snapshot(original), "execution_plan": original_plan,
                 "execution_plan_digest": training_plan.fingerprint(original_plan)}
    (source_output / "effective_training.json").write_text(json.dumps(effective))

    source = {"asset_id": "resume-model", "run_id": "old-run", "attempt_id": source_output.name,
              "path": str(checkpoint), "output_dir": str(source_output), "step": 5}
    spec = {**old_spec, "template_id": "openpi-sft", "output_dir": str(tmp_path / "new-attempt"),
            "resume_source": source, "parameters": {**old_spec["parameters"], "steps": 30},
            "inputs": {**old_spec["inputs"], "model": {"id": source["asset_id"], "path": source["path"], "kind": "MODEL"}}}
    return base, spec, module, loaders


def test_resume_plan_freezes_original_checkpoint_and_binds_new_output(resume):
    base, spec, module, loaders = resume
    source = spec["resume_source"]
    config = openpi_entry.resolve_config(base, spec, module, loaders)
    assert config.resume_checkpoint == source["path"] and config.resume is config.overwrite is False
    assert config.checkpoint_base_dir == str(Path(spec["output_dir"]) / "checkpoints")
    checks = []
    plan = training_plan.inspect_training_config(config, spec, checks=checks)
    assert plan["resume_source"] == source and plan["resume_checkpoint"]["train_state_bytes"] > 0
    assert validate_resume_evidence(plan, source) == plan["resume_checkpoint"]
    assert any(check["name"] == "resume_checkpoint" and check["ok"] for check in checks)
    tree = ast.parse(plan["generated_config"]["source"])
    frozen = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FROZEN_REQUEST" for target in node.targets))
    assert frozen["resume_source"] == source
    assert not Path(spec["output_dir"]).exists()

    # Reinspection at START includes changes to optimizer shards in its plan digest.
    (Path(source["path"]) / "train_state/d/shard").write_bytes(b"changed optimizer state")
    current = training_plan.inspect_training_config(config, spec, checks=[])
    assert training_plan.fingerprint(current) != training_plan.fingerprint(plan)


@pytest.mark.parametrize("change,code", [
    ("params_only", "RESUME_CHECKPOINT_INVALID"),
    ("uncommitted", "RESUME_CHECKPOINT_INVALID"),
    ("source_config", "RESUME_SOURCE_CONFIG_INVALID"),
    ("learning_rate", "RESUME_CONFIG_MISMATCH"),
    ("dataset", "RESUME_DATA_MISMATCH"),
    ("norm", "RESUME_DATA_MISMATCH"),
    ("old_entrypoint", "RESUME_UNSUPPORTED"),
    ("steps", "RESUME_STEPS_INVALID"),
    ("same_output", "RESUME_SOURCE_INVALID"),
])
def test_resume_rejects_incomplete_or_incompatible_sources(resume, change, code):
    base, spec, module, loaders = resume
    source = spec["resume_source"]
    checkpoint = Path(source["path"])
    if change == "params_only":
        (checkpoint / "train_state/_METADATA").unlink()
    elif change == "uncommitted":
        (checkpoint / "_CHECKPOINT_METADATA").write_text("{}")
    elif change == "source_config":
        (Path(source["output_dir"]) / "effective_training.json").write_text("{}")
    elif change == "learning_rate":
        spec["parameters"]["learning_rate"] *= 2
    elif change == "dataset":
        (Path(spec["inputs"]["dataset"]["path"]) / "data/episode_000000.parquet").write_bytes(b"changed parquet")
    elif change == "norm":
        norm = Path(spec["inputs"]["norm"]["path"])
        norm.write_text(norm.read_text() + "\n")
    elif change == "old_entrypoint":
        entry = Path(spec["code_dir"]) / "scripts/train.py"
        entry.write_text(entry.read_text().replace("PLATFORM_CHECKPOINT_RESUME_VERSION = 1", ""))
    elif change == "steps":
        spec["parameters"]["steps"] = source["step"] + 1
    elif change == "same_output":
        spec["output_dir"] = source["output_dir"]
    with pytest.raises(ValueError, match=code):
        config = openpi_entry.resolve_config(base, spec, module, loaders)
        training_plan.inspect_training_config(config, spec, checks=[])


def test_launch_carries_resume_identity_and_preserves_legacy_inputs(resume):
    _, spec, _, _ = resume
    profile = {"id": "sft", "adapter": "openpi-sft", "code_dir": spec["code_dir"], "python": "/usr/bin/python3"}
    args = (profile, "openpi-sft", spec["inputs"], spec["parameters"], spec["output_dir"], ["GPU-a"])
    launch = build_launch(*args, recipe=spec["recipe"], resume_source=spec["resume_source"])
    frozen = json.loads(launch["argv"][-1])
    assert frozen["resume_source"] == spec["resume_source"] and frozen["template_id"] == "openpi-sft"
    launch = build_launch(*args, recipe=spec["recipe"])
    assert "resume_source" not in json.loads(launch["argv"][-1])
    changed = copy.deepcopy(spec["resume_source"])
    changed["asset_id"] = "a-different-checkpoint"
    with pytest.raises(ValueError, match="RESUME_SOURCE_INVALID"):
        build_launch(*args, recipe=spec["recipe"], resume_source=changed)
    with pytest.raises(ValueError, match="RESUME_UNSUPPORTED"):
        build_launch(*args, recipe=spec["recipe"], resume_source=spec["resume_source"], entrypoint="custom.py")


def test_resume_accepts_committed_step_zero(resume):
    base, spec, module, loaders = resume
    source = spec["resume_source"]
    path = Path(source["path"])
    zero = path.rename(path.with_name("0"))
    source.update(step=0, path=str(zero))
    spec["inputs"]["model"]["path"] = str(zero)
    config = openpi_entry.resolve_config(base, spec, module, loaders)
    plan = training_plan.inspect_training_config(config, spec, checks=[])
    assert validate_resume_evidence(plan, source) == plan["resume_checkpoint"]
