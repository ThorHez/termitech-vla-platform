import copy
import dataclasses
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
    monkeypatch.setattr(recap_runtime, "_parquet_evidence", lambda path, required, supervised, **kw: {
        "parquet_files": 1, "checked_files": [{"path": "data/episode_000000.parquet", "sample_rows": 20}],
        "scope": "unit fixture"})
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
