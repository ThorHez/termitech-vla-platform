import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import recap_runtime, training_plan
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.training_plan import fingerprint, inspect_training_config, training_entrypoint
from test_training_plan import Assets, Config, Data, training


@dataclasses.dataclass
class Pi0ValueConfig:
    value_min: float = -1.0
    value_max: float = 0.0


@dataclasses.dataclass
class Pi0Config:
    action_horizon: int = 16


@dataclasses.dataclass
class CheckpointWeightLoaderWithValueHead:
    params_path: str


@dataclasses.dataclass
class CheckpointWeightLoaderWithGripperHead:
    params_path: str


@dataclasses.dataclass
class ValueFactory(Data):
    mapping: dict = dataclasses.field(default_factory=lambda: {
        "actions": "actions", "prompt": "task", "value_target": "value_target",
        "episode_index": "episode_index", "frame_index": "frame_index"})


@dataclasses.dataclass
class ACPFactory(Data):
    mapping: dict = dataclasses.field(default_factory=lambda: {
        "actions": "actions", "prompt": "task", "is_positive": "is_positive"})
    acp_dropout: float = 0.3


@dataclasses.dataclass
class ReCapConfig(Config):
    model: object = dataclasses.field(default_factory=Pi0ValueConfig)
    val_ratio: float = 0.1
    eval_interval: int = 1000
    eval_batches: int = 10


@pytest.fixture
def recap(training, monkeypatch):
    old, spec, module, loaders = training
    base = ReCapConfig(**{field.name: copy.deepcopy(getattr(old, field.name))
                         for field in dataclasses.fields(old)})
    base.data.datasets = [ValueFactory(row.repo_id, row.assets) for row in base.data.datasets]
    base.weight_loader = CheckpointWeightLoaderWithValueHead("old/params")
    spec.update(template_id="recap", training_stage="value", stage_config={"val_ratio": 0.2})
    monkeypatch.setenv("HOME", str(Path(spec["code_dir"]).parent / "home"))
    for entry in ("train_value.py", "train_value_multi_dataset.py", "train_advantage.py"):
        (Path(spec["code_dir"]) / "scripts" / entry).write_text(
            "raise AssertionError('preflight must not execute entry')\n"
            "def main(config): pass\ndef init_wandb(*a, **kw): pass\n")
    monkeypatch.setattr(recap_runtime, "require_cached_supervision", lambda dataset, spec: {
        "parquet_files": 1, "checked_files": [{"path": "data/episode_000000.parquet", "sample_rows": 20,
            "columns": ["actions", "value_target", "is_positive", "episode_index", "frame_index"]}],
        "supervision_summary": {"value_target": {"min": -0.8, "max": 0., "count": 20},
                                "is_positive": {"min": 0, "max": 1, "count": 20, "binary": True}},
        "scope": "existing basic validation unit fixture"})
    return base, spec, module, loaders


def test_value_stage_preserves_value_loader_and_selects_value_training(recap):
    base, spec, module, loaders = recap
    config = resolve_config(base, spec, module, loaders)
    assert type(config.weight_loader).__name__ == "CheckpointWeightLoaderWithValueHead"
    assert config.weight_loader.params_path.endswith("/model/params")
    assert base.weight_loader.params_path == "old/params"
    assert config.val_ratio == 0.2 and base.val_ratio == 0.1
    assert training_entrypoint(config, spec).name == "train_value_multi_dataset.py"
    checks = []
    plan = inspect_training_config(config, spec, checks=checks)
    assert plan["training_stage"] == "value" and plan["stage_config"] == {"val_ratio": 0.2}
    assert plan["recap_stage_contract"]["model_type"] == "Pi0ValueConfig"
    assert plan["recap_stage_contract"]["validation"]["val_ratio"] == 0.2
    assert all("value_target" in row["required_columns"] for row in plan["recap_stage_contract"]["datasets"])
    assert next(row for row in checks if row["name"] == "recap_stage")["ok"] is True


def as_acp(recap):
    base, spec, module, loaders = recap
    base.model = Pi0Config()
    base.data.datasets = [ACPFactory(row.repo_id, row.assets) for row in base.data.datasets]
    base.weight_loader = CheckpointWeightLoaderWithGripperHead("old/params")
    spec.update(training_stage="offline_rl", stage_config={"acp_dropout": 0.15})
    return base, spec, module, loaders


def test_offline_stage_overrides_all_data_factories_without_value_checkpoint_dependency(recap):
    base, spec, module, loaders = as_acp(recap)
    config = resolve_config(base, spec, module, loaders)
    assert type(config.weight_loader).__name__ == "CheckpointWeightLoaderWithGripperHead"
    assert [factory.acp_dropout for factory in config.data.datasets] == [0.15] * 3
    assert [factory.acp_dropout for factory in base.data.datasets] == [0.3] * 3
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["entrypoint"].endswith("/scripts/train_multi_dataset.py")
    assert plan["training_stage"] == "offline_rl"
    assert all("is_positive" in row["required_columns"] for row in plan["recap_stage_contract"]["datasets"])
    assert set(spec["inputs"]) == {"model", *(f"dataset_{i}" for i in range(3)), *(f"norm_{i}" for i in range(3))}


def test_omitted_stage_values_inherit_recipe_and_requested_override_is_frozen(recap):
    base, spec, module, loaders = recap
    spec["stage_config"] = {}
    first = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert first["stage_config"] == {}
    assert first["recap_stage_contract"]["validation"]["val_ratio"] == 0.1
    spec["stage_config"] = {"val_ratio": 0.2}
    second = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert fingerprint(first) != fingerprint(second)


@pytest.mark.parametrize("case", ["policy_for_value", "value_for_policy", "unlabelled_acp", "plain_value_loader"])
def test_mismatched_stage_recipe_is_rejected_before_running_entry(recap, case):
    base, spec, module, loaders = recap
    if case == "policy_for_value":
        base.model = Pi0Config()
    elif case == "value_for_policy":
        spec.update(training_stage="offline_rl", stage_config={})
    elif case == "unlabelled_acp":
        base, spec, module, loaders = as_acp(recap)
        base.data.datasets[0].mapping.pop("is_positive")
    else:
        base.weight_loader = None
    with pytest.raises(ValueError, match="RECAP_STAGE_RECIPE_MISMATCH"):
        resolve_config(base, spec, module, loaders)


@pytest.mark.parametrize("entry", ["train.py", "train_multi_dataset.py", "train_advantage.py", "train_value.py"])
def test_value_stage_rejects_known_wrong_or_single_entry_for_multiple_data(recap, entry):
    base, spec, module, loaders = recap
    spec["entrypoint"] = "scripts/" + entry
    with pytest.raises(ValueError, match="RECAP_STAGE_ENTRYPOINT_MISMATCH"):
        training_entrypoint(resolve_config(base, spec, module, loaders), spec)


@pytest.mark.parametrize("entry", ["train_value.py", "train_value_multi_dataset.py", "train_advantage.py"])
def test_offline_stage_rejects_value_or_joint_entry(recap, entry):
    base, spec, module, loaders = as_acp(recap)
    spec["entrypoint"] = "scripts/" + entry
    with pytest.raises(ValueError, match="RECAP_STAGE_ENTRYPOINT_MISMATCH"):
        training_entrypoint(resolve_config(base, spec, module, loaders), spec)


def install_fake_parquet(monkeypatch, *, names, values):
    class Batch:
        num_rows = len(next(iter(values.values())))
        def to_pydict(self):
            return values
    class Source:
        schema_arrow = SimpleNamespace(names=names)
        def iter_batches(self, batch_size, columns):
            assert batch_size <= 64
            yield Batch()
    parent = ModuleType("pyarrow")
    child = ModuleType("pyarrow.parquet")
    child.ParquetFile = lambda path: Source()
    parent.parquet = child
    monkeypatch.setitem(sys.modules, "pyarrow", parent)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", child)


def test_actual_parquet_columns_are_required_even_if_metadata_claims_they_exist(training, monkeypatch):
    base, spec, *_ = training
    path = base.data.datasets[0].repo_id
    install_fake_parquet(monkeypatch, names=["actions"], values={"value_target": [-0.2]})
    with pytest.raises(ValueError, match="RECAP_DATA_SCHEMA_INVALID.*value_target"):
        recap_runtime._parquet_evidence(path, ["actions", "value_target"], ["value_target"],
                                       stage="value", value_min=-1, value_max=0)


@pytest.mark.parametrize("column,value", [("value_target", float("nan")), ("value_target", -1.1),
    ("value_target", None), ("is_positive", 0.4), ("is_positive", 2), ("is_positive", True)])
def test_actual_supervision_samples_reject_invalid_values(training, monkeypatch, column, value):
    base, spec, *_ = training
    install_fake_parquet(monkeypatch, names=[column], values={column: [value]})
    with pytest.raises(ValueError, match="RECAP_LABEL_INVALID"):
        recap_runtime._parquet_evidence(base.data.datasets[0].repo_id, [column], [column],
                                       stage="value" if column == "value_target" else "offline_rl",
                                       value_min=-1, value_max=0)


def test_actual_scalar_and_singleton_list_targets_are_supported(training, monkeypatch):
    base, spec, *_ = training
    install_fake_parquet(monkeypatch, names=["value_target"], values={"value_target": [[-0.2], 0.0]})
    evidence = recap_runtime._parquet_evidence(base.data.datasets[0].repo_id, ["value_target"], ["value_target"],
                                             stage="value", value_min=-1, value_max=0)
    assert evidence["checked_files"][0]["sample_rows"] == 2


def test_resolved_stage_values_cannot_disagree_with_frozen_request(recap):
    base, spec, module, loaders = recap
    config = resolve_config(base, spec, module, loaders)
    config.val_ratio = 0.6
    with pytest.raises(ValueError, match="RECAP_STAGE_CONFIGURATION_MISMATCH"):
        inspect_training_config(config, spec, checks=[])


def test_value_runtime_cache_checks_real_home_path_and_cleans_probe(recap):
    base, spec, module, loaders = recap
    plan = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    path = Path(plan["recap_stage_contract"]["runtime_cache"]["jax_compilation_cache_dir"])
    assert path == Path(spec["code_dir"]).parent / "home/.cache/jax"
    assert path.is_dir() and not list(path.iterdir())


def test_value_runtime_cache_permission_failure_blocks_ready(recap, monkeypatch):
    base, spec, module, loaders = recap
    def denied(**kwargs):
        raise PermissionError("cache is not writable")
    monkeypatch.setattr(recap_runtime.tempfile, "mkstemp", denied)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNWRITABLE.*cache is not writable"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])


def test_value_runtime_cache_cannot_escape_agent_roots(recap, monkeypatch):
    base, spec, module, loaders = recap
    outside = Path(spec["code_dir"]).parents[1] / "outside-home"
    monkeypatch.setenv("HOME", str(outside))
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNAVAILABLE.*受管目录"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert not outside.exists()


def test_value_home_path_drift_changes_frozen_plan(recap, monkeypatch):
    base, spec, module, loaders = recap
    config = resolve_config(base, spec, module, loaders)
    first = inspect_training_config(config, spec, checks=[])
    monkeypatch.setenv("HOME", str(Path(spec["code_dir"]).parent / "changed-home"))
    second = inspect_training_config(config, spec, checks=[])
    assert fingerprint(first) != fingerprint(second)


@pytest.mark.parametrize("key,value", [("val_ratio", 1.0), ("eval_interval", 0),
                                     ("eval_batches", float("inf"))])
def test_inherited_value_validation_parameters_must_be_valid(recap, key, value):
    base, spec, module, loaders = recap
    spec["stage_config"] = {}
    setattr(base, key, value)
    with pytest.raises(ValueError, match="RECAP_STAGE_CONFIG_INVALID"):
        resolve_config(base, spec, module, loaders)


@pytest.fixture
def supervision_source(training, monkeypatch):
    _, spec, *_ = training
    code = Path(spec["code_dir"])
    for relative in ("src/openpi", "third_party/lerobot/lerobot", "packages/openpi-client"):
        source = code / relative / "implementation.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("VERSION = 1\n")
    path = Path(spec["inputs"]["dataset_0"]["path"])
    info = json.loads((path / "meta/info.json").read_text())
    info["fps"] = 20
    info["features"].update(value_target={"shape": [1], "dtype": "float32"},
                            is_positive={"shape": [1], "dtype": "float32"})
    (path / "meta/info.json").write_text(json.dumps(info))
    spec["validation_cache_dir"] = str(code.parent / "supervision-receipts")
    data = {"path": str(path), "frames": info["total_frames"], "episodes": info["total_episodes"],
            "effective_fps": 20, "metadata_sha256": hashlib.sha256((path / "meta/info.json").read_bytes()).hexdigest(),
            "data_manifest_digest": training_plan.file_manifest(path / "data", spec)[2]}
    install_fake_parquet(monkeypatch, names=["actions", "episode_index", "frame_index", "value_target", "is_positive"],
                         values={"value_target": [-1.25, 0.25], "is_positive": [0, 1]})
    monkeypatch.setenv("HOME", str(code.parent / "home"))
    return data, spec, info


def test_basic_supervision_records_observed_range_and_binary_labels_without_training_clip(supervision_source):
    data, spec, info = supervision_source
    evidence = recap_runtime.prepare_source_supervision(data, spec, info)
    assert evidence["status"] == "verified"
    assert evidence["supervision_summary"]["value_target"] == {"min": -1.25, "max": 0.25, "count": 2}
    assert evidence["supervision_summary"]["is_positive"] == {"min": 0, "max": 1, "count": 2, "binary": True}
    assert evidence["checked_files"][0]["sample_rows"] == 2
    assert "actions" in evidence["checked_files"][0]["columns"]
    assert recap_runtime.require_cached_supervision(data, spec) == evidence


def test_value_submission_reuses_receipt_and_checks_current_range_without_parquet_reads(supervision_source, monkeypatch):
    data, spec, info = supervision_source
    recap_runtime.prepare_source_supervision(data, spec, info)
    monkeypatch.setattr(recap_runtime, "_parquet_evidence", lambda *a, **kw: pytest.fail("Submission must not read Parquet rows"))
    monkeypatch.setattr(sys.modules["pyarrow.parquet"], "ParquetFile", lambda *a, **kw: pytest.fail("No Parquet constructor at submission"))
    factory = ValueFactory(data["path"], Assets(data["path"]))
    config = ReCapConfig(data=factory, model=Pi0ValueConfig(value_min=-2., value_max=1.),
                         weight_loader=CheckpointWeightLoaderWithValueHead("weights"))
    spec.update(template_id="recap", training_stage="value", stage_config={})
    first = recap_runtime.inspect_stage(config, spec, [factory], [data])
    assert first == recap_runtime.inspect_stage(config, spec, [factory], [data])
    assert first["value_range"] == [-2., 1.]
    assert first["datasets"][0]["supervision_summary"]["value_target"]["min"] == -1.25
    config.model = Pi0ValueConfig(value_min=-1., value_max=0.)
    with pytest.raises(ValueError, match="RECAP_LABEL_INVALID.*value_target"):
        recap_runtime.inspect_stage(config, spec, [factory], [data])


def test_offline_submission_reuses_binary_receipt_without_scanning_data(supervision_source, monkeypatch):
    data, spec, info = supervision_source
    recap_runtime.prepare_source_supervision(data, spec, info)
    monkeypatch.setattr(recap_runtime, "_parquet_evidence", lambda *a, **kw: pytest.fail("No late supervision scanning"))
    factory = ACPFactory(data["path"], Assets(data["path"]))
    config = ReCapConfig(data=factory, model=Pi0Config(),
                         weight_loader=CheckpointWeightLoaderWithGripperHead("weights"))
    spec.update(template_id="recap", training_stage="offline_rl", stage_config={})
    result = recap_runtime.inspect_stage(config, spec, [factory], [data])
    assert result["datasets"][0]["supervision_columns"] == ["is_positive"]
    assert result["datasets"][0]["supervision_summary"]["is_positive"]["binary"] is True


@pytest.mark.parametrize("case", ["missing", "changed_data", "changed_metadata", "changed_fps"])
def test_missing_or_stale_supervision_requires_basic_validation_without_fallback(supervision_source, monkeypatch, case):
    data, spec, info = supervision_source
    if case != "missing":
        recap_runtime.prepare_source_supervision(data, spec, info)
    if case == "changed_data":
        (Path(data["path"]) / "data/episode_000000.parquet").write_bytes(b"new source")
    elif case == "changed_metadata":
        (Path(data["path"]) / "meta/info.json").write_text(json.dumps({**info, "fps": 25}))
    elif case == "changed_fps":
        data["effective_fps"] = 25
    monkeypatch.setattr(recap_runtime, "_parquet_evidence", lambda *a, **kw: pytest.fail("No fallback scan on missing/stale receipt"))
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED.*基础校验"):
        recap_runtime.require_cached_supervision(data, spec)


def test_basic_validation_rejects_nonbinary_labels_and_submission_checks_actual_columns(supervision_source, monkeypatch):
    data, spec, info = supervision_source
    install_fake_parquet(monkeypatch, names=["actions", "value_target", "is_positive"],
                         values={"value_target": [-0.5], "is_positive": [0.4]})
    with pytest.raises(ValueError, match="RECAP_LABEL_INVALID.*is_positive"):
        recap_runtime.prepare_source_supervision(data, spec, info)
    assert not list(Path(spec["validation_cache_dir"]).glob("*.json"))
    install_fake_parquet(monkeypatch, names=["value_target", "is_positive"],
                         values={"value_target": [-0.5], "is_positive": [1]})
    recap_runtime.prepare_source_supervision(data, spec, info)
    factory = ACPFactory(data["path"], Assets(data["path"]))
    config = ReCapConfig(data=factory, model=Pi0Config())
    spec.update(template_id="recap", training_stage="offline_rl", stage_config={})
    monkeypatch.setattr(recap_runtime, "_parquet_evidence", lambda *a, **kw: pytest.fail("No fallback for missing stage columns"))
    with pytest.raises(ValueError, match="RECAP_DATA_SCHEMA_INVALID.*actions"):
        recap_runtime.inspect_stage(config, spec, [factory], [data])


def test_basic_supervision_retains_sixteen_file_and_sixty_four_row_coverage(supervision_source, monkeypatch):
    data, spec, info = supervision_source
    root = Path(data["path"])
    for index in range(1, 20):
        (root / "data" / f"episode_{index:06d}.parquet").write_bytes(b"bounded supervision fixture")
    info["total_frames"] = data["frames"] = 20 * 64
    (root / "meta/info.json").write_text(json.dumps(info))
    data["metadata_sha256"] = hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
    data["data_manifest_digest"] = training_plan.file_manifest(root / "data", spec)[2]
    install_fake_parquet(monkeypatch, names=["value_target", "is_positive"],
                         values={"value_target": [-0.5] * 64, "is_positive": [0, 1] * 32})
    evidence = recap_runtime.prepare_source_supervision(data, spec, info)
    assert evidence["parquet_files"] == 20 and len(evidence["checked_files"]) == 16
    assert evidence["checked_files"][0]["path"].endswith("episode_000000.parquet")
    assert evidence["checked_files"][-1]["path"].endswith("episode_000019.parquet")
    assert all(row["sample_rows"] == 64 for row in evidence["checked_files"])
    assert evidence["supervision_summary"]["value_target"]["count"] == 16 * 64
