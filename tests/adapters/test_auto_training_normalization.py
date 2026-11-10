"""Automatic startup normalization is frozen before execution, not in preflight."""

import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vla_adapters import openpi_entry, training_plan
from vla_adapters.openpi_entry import checkpoint_norm_descriptor, resolve_config
from vla_adapters.training_plan import auto_shared_norm_contract, inspect_training_config

from test_shared_normalization import shared
from test_training_plan import training  # noqa: F401


def test_inherited_same_source_norm_still_requests_current_raw_sample_distribution(training):
    base, spec, module, loaders = training
    reference = copy.deepcopy(spec["inputs"]["norm_0"])
    for i, binding in enumerate(spec["dataset_bindings"]):
        binding["norm_asset_id"] = reference["id"]
        spec["inputs"][f"norm_{i}"] = copy.deepcopy(reference)
    config = resolve_config(base, spec, module, loaders)
    assert config.data.auto_shared_norm is True and config.data.shared_norm_stats_path is None
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["normalization"]["mode"] == "auto_shared"
    assert {row["norm_usage"] for row in plan["datasets"]} == {"reference_only"}
    assert len({row["effective_norm_path"] for row in plan["datasets"]}) == 1
    assert "auto_shared_norm=True" in plan["generated_config"]["source"]


def test_single_child_keeps_bound_norm_and_registered_fps_is_carried_to_numeric_reader(training):
    base, spec, module, loaders = training
    spec["dataset_bindings"] = spec["dataset_bindings"][:1]
    spec["inputs"] = {key: value for key, value in spec["inputs"].items() if key in {"model", "dataset_0", "norm_0"}}
    spec["inputs"]["dataset_0"]["dataset_profile"] = {"fps": 20.}
    config = resolve_config(base, spec, module, loaders)
    assert config.data.auto_shared_norm is False and config.data.use_merged_norm_stats is False
    assert config.data.datasets[0]._norm_effective_fps == 20.
    assert not hasattr(base.data.datasets[0], "_norm_effective_fps")


def test_reference_norm_width_or_formula_does_not_block_new_shared_computation(training):
    base, spec, module, loaders = training
    for key, asset in spec["inputs"].items():
        if key.startswith("norm_"):
            Path(asset["path"]).write_text(json.dumps({"old_statistics": "retained only as an input reference"}))
    config = resolve_config(base, spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["normalization"]["mode"] == "auto_shared"
    assert all(row["norm_usage"] == "reference_only" and len(row["norm_sha256"]) == 64 for row in plan["datasets"])


@pytest.fixture
def planner(training, monkeypatch):
    base, spec, module, loaders = training
    config = resolve_config(base, spec, module, loaders)
    relative = ["training/training_normalization.py", "training/shared_norm.py", "training/numeric_stats.py",
                "training/composed_processing.py", "shared/weighted_stats.py"]
    root = Path(spec["code_dir"]) / "src/openpi"
    for name in relative:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen numeric implementation " + name)
    plan = {"mode": "auto_shared", "method": "weighted_raw_samples", "coverage": "full",
            "fingerprint": "f" * 64, "sources": [{"weight": row["weight"]} for row in spec["dataset_bindings"]],
            "numeric_io": [{"schema_version": 3, "action_sequence": "per_frame", "temporal_columns": []}
                           for _ in spec["dataset_bindings"]],
            "output_path": str(Path(spec["output_dir"]) / "normalization/norm_stats.json")}
    calls = []
    helper = SimpleNamespace(__file__=str(root / relative[0]),
        planned_training_normalization=lambda actual: calls.append(actual) or copy.deepcopy(plan),
        prepare_training_normalization=lambda *_: pytest.fail("Preflight cannot calculate all statistics"))
    imported = training_plan.importlib.import_module
    monkeypatch.setattr(training_plan.importlib, "import_module", lambda name: helper
                        if name == "openpi.training.training_normalization" else imported(name))
    return config, spec, helper, plan, calls


def test_preflight_only_plans_and_freezes_all_numeric_implementations(planner):
    config, spec, _, _, calls = planner
    planned = auto_shared_norm_contract(config, spec)
    assert calls == [config]
    assert len(planned["implementation_sha256"]) == 5
    assert planned["status"] == "pending_training_start" and planned["source_norm_assets"] == "reference_only"
    assert not Path(planned["output_path"]).exists()
    implementation = Path(spec["code_dir"]) / "src/openpi/shared/weighted_stats.py"
    implementation.write_text(implementation.read_text() + "\n# changed formula")
    assert auto_shared_norm_contract(config, spec)["implementation_sha256"] != planned["implementation_sha256"]


def test_preview_and_attempt_outputs_have_identical_frozen_normalization_identity(planner):
    config, spec, _, plan, _ = planner
    preview = auto_shared_norm_contract(config, spec)
    assert preview["output_path"].startswith("<runtime_output_dir>/")
    spec["output_dir"] += "-different-real-attempt"
    plan["output_path"] = str(Path(spec["output_dir"]) / "normalization/norm_stats.json")
    actual = auto_shared_norm_contract(config, spec)
    assert training_plan.fingerprint(preview) == training_plan.fingerprint(actual)
    assert preview == actual


@dataclasses.dataclass
class ComposableDataConfig:
    repo_id: str
    composition: dict


@pytest.mark.parametrize("change, expected", [
    ("composition", ("状态表示（组合向量与关节）", "动作表示（组合向量与关节）")),
    ("metadata", ("源元数据的位置单位（m与mm）",)),
])
def test_shared_layout_error_explains_reference_norms_and_names_the_differing_groups(
    planner, monkeypatch, change, expected,
):
    config, spec, helper, _, _ = planner
    factories = [
        ComposableDataConfig(child.repo_id, {
            "state_representation": "composite", "action_representation": "composite",
        })
        for child in config.data.datasets
    ]
    if change == "composition":
        factories[2].composition.update(state_representation="joint", action_representation="joint")
    else:
        for index, factory in enumerate(factories):
            path = Path(factory.repo_id) / "meta/info.json"
            info = json.loads(path.read_text())
            path.write_text(json.dumps({**info, "position_unit": "mm" if index == 2 else "m"}))
    config.data.datasets = factories
    monkeypatch.setitem(sys.modules, "openpi.training.shared_norm", shared)
    helper.planned_training_normalization = lambda actual: shared.source_descriptors(actual.data.datasets, [1.] * 3)

    with pytest.raises(ValueError) as caught:
        auto_shared_norm_contract(config, spec)

    message = str(caught.value)
    assert message.startswith("SHARED_NORM_INVALID: 本次多数据集训练会自动计算共享归一化参数")
    assert "各组绑定的 norm_stats 仅作参考" in message
    assert f"第 1 组「{Path(factories[0].repo_id).name}」与第 3 组「{Path(factories[2].repo_id).name}」" in message
    assert all(detail in message for detail in expected)
    assert str(Path(factories[2].repo_id).parent) not in message


@pytest.mark.parametrize("case", ["outside_module", "outside_output", "missing_source", "incomplete_plan", "old_reader"])
def test_preflight_rejects_missing_or_wrong_checkout_automatic_statistics(planner, case):
    config, spec, helper, plan, _ = planner
    if case == "outside_module":
        helper.__file__ = str(Path(spec["code_dir"]).parent / "wrong.py")
    elif case == "outside_output":
        plan["output_path"] = str(Path(spec["code_dir"]) / "norm_stats.json")
    elif case == "missing_source":
        Path(helper.__file__).unlink()
    elif case == "old_reader":
        plan["numeric_io"][0].update(schema_version=2, action_sequence="packed_first_step")
    else:
        plan.pop("fingerprint")
    with pytest.raises(ValueError, match="MODULE_OUTSIDE_CODE_DIR|SHARED_NORM_UNSUPPORTED"):
        auto_shared_norm_contract(config, spec)


def test_model_norm_descriptor_uses_only_checkpoint_shared_statistics(tmp_path):
    checkpoint = tmp_path / "checkpoints/100"
    source = checkpoint / "assets/shared/norm_stats.json"
    source.parent.mkdir(parents=True)
    document = {"norm_stats": {key: {"mean": [0.], "std": [1.], "q01": [-2.], "q99": [2.]}
                                for key in ("state", "actions")},
                "shared_normalization": {"method": "weighted_raw_samples"}, "normalization_method": "quantile"}
    source.write_text(json.dumps(document))
    result = checkpoint_norm_descriptor(checkpoint, {"allowed_roots": [str(tmp_path)]})
    assert result == {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                      "normalization_method": "quantile", "shared": True}
    assert checkpoint_norm_descriptor(tmp_path / "old-checkpoint", {"allowed_roots": [str(tmp_path)]}) is None
    external = tmp_path / "source-norm.json"
    external.write_text(source.read_text())
    source.unlink(); source.symlink_to(external)
    with pytest.raises(ValueError, match="OUTPUT_INVALID"):
        checkpoint_norm_descriptor(checkpoint, {"allowed_roots": [str(tmp_path)]})
