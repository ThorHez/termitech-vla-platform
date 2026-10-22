import copy
import dataclasses
import json
from pathlib import Path

import pytest

from vla_adapters import build_launch
from vla_adapters.dataset_bindings import binding_inputs, normalize_dataset_bindings, validate_binding_assets
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.training_plan import fingerprint, inspect_training_config

from test_training_plan import training  # noqa: F401; realistic local dataset/checkpoint fixture


def explicit(training, indices=(0, 1, 2)):
    base, spec, module, loaders = training
    rows = [{"dataset_asset_id": f"data-{i}", "norm_asset_id": f"norm-{i}",
             "recipe_index": source, "weight": float((i + 1) * 10)} for i, source in enumerate(indices)]
    spec["dataset_bindings"] = rows
    spec["inputs"] = {"model": spec["inputs"]["model"]}
    for i in range(len(rows)):
        # Reverse the file paths so the resolver must obey bindings, rather than
        # retain an original dataset by matching its recipe path.
        path = base.data.datasets[2 - i % 3].repo_id
        spec["inputs"][f"dataset_{i}"] = {"id": f"data-{i}", "path": path, "kind": "DATASET"}
        spec["inputs"][f"norm_{i}"] = {"id": f"norm-{i}", "path": path + "/norm_stats.json", "kind": "NORM"}
    return base, spec, module, loaders


def test_all_rows_override_data_norm_and_weights_without_mutating_recipe(training):
    base, spec, module, loaders = explicit(training)
    original = copy.deepcopy(base)
    config = resolve_config(base, spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert [row["path"] for row in plan["datasets"]] == [spec["inputs"][f"dataset_{i}"]["path"] for i in range(3)]
    assert [row["norm_path"] for row in plan["datasets"]] == [spec["inputs"][f"norm_{i}"]["path"] for i in range(3)]
    assert config.data.weights == plan["weights"] == [10., 20., 30.]
    assert plan["dataset_bindings"] == spec["dataset_bindings"]
    assert base == original
    spec["dataset_bindings"][1]["weight"] = 5
    changed = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert fingerprint(changed) != fingerprint(plan)


def test_can_remove_reorder_and_clone_recipe_factories(training):
    @dataclasses.dataclass
    class HorizonOne(type(training[0].data.datasets[0])):
        image_horizon: int = 1

    base, spec, module, loaders = explicit(training, indices=(1, 1))
    source = base.data.datasets[1]
    base.data.datasets[1] = HorizonOne(source.repo_id, source.assets)
    config = resolve_config(base, spec, module, loaders)
    assert len(config.data.datasets) == 2
    assert all(isinstance(row, HorizonOne) and row.image_horizon == 1 for row in config.data.datasets)
    plan = inspect_training_config(config, spec, checks=[])
    assert len(plan["datasets"]) == 2 and plan["weights"] == [10., 20.]
    # The omitted original child must not be inspected or silently included.
    base.data.datasets[0].repo_id = "/missing/omitted/dataset"
    assert len(inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])["datasets"]) == 2


def test_every_resource_uses_registered_assets_even_when_all_recipe_defaults_are_unavailable(training):
    base, spec, module, loaders = explicit(training, indices=(2, 0))
    for i, child in enumerate(base.data.datasets):
        child.repo_id = f"/unregistered/default-dataset-{i}"
        child.assets = type(child.assets)(f"/unregistered/default-norm-{i}", "other")
    base.weight_loader = "/unregistered/default-model"
    config = resolve_config(base, spec, module, loaders)
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["input_authority"] == "registered_assets"
    assert len(plan["datasets"]) == 2
    for i, row in enumerate(plan["datasets"]):
        assert row["path"] == spec["inputs"][f"dataset_{i}"]["path"]
        assert row["norm_path"] == spec["inputs"][f"norm_{i}"]["path"]
        assert row["dataset_asset_id"] == f"data-{i}"
        assert row["norm_asset_id"] == f"norm-{i}"
        assert row["recipe_index"] == (2, 0)[i]
    assert config.weight_loader == plan["model"]["path"]
    assert plan["model"]["asset_id"] == spec["inputs"]["model"]["id"]
    assert "/unregistered/" not in json.dumps(plan)


@pytest.mark.parametrize("field", ["rlds_data_dir", "filter_dict_path"])
@pytest.mark.parametrize("owner", ["factory", "base_config"])
def test_alternative_recipe_data_resources_cannot_bypass_asset_binding(training, field, owner):
    @dataclasses.dataclass
    class ResourceData(type(training[0].data.datasets[0])):
        base_config: object = None
        rlds_data_dir: object = None
        filter_dict_path: object = None

    @dataclasses.dataclass
    class BaseData:
        rlds_data_dir: object = None
        filter_dict_path: object = None

    base, spec, module, loaders = explicit(training, indices=(0,))
    source = base.data.datasets[0]
    factory = ResourceData(source.repo_id, source.assets, BaseData())
    setattr(factory if owner == "factory" else factory.base_config, field, "/unregistered/source")
    base.data.datasets[0] = factory
    with pytest.raises(ValueError, match="RECIPE_INPUT_SOURCE_UNSUPPORTED"):
        resolve_config(base, spec, module, loaders)
    setattr(factory if owner == "factory" else factory.base_config, field, None)
    config = resolve_config(base, spec, module, loaders)
    setattr(config.data.datasets[0] if owner == "factory" else config.data.datasets[0].base_config, field, "/unregistered/source")
    with pytest.raises(ValueError, match="RECIPE_INPUT_SOURCE_UNSUPPORTED"):
        inspect_training_config(config, spec, checks=[])


def test_base_config_resource_defaults_are_rebound_without_changing_processing_options(training):
    @dataclasses.dataclass
    class BaseData:
        repo_id: str = "/unregistered/base-dataset"
        asset_id: str = "unregistered-norm"
        norm_stats: object = dataclasses.field(default_factory=lambda: {"old": "statistics"})
        prompt_from_task: bool = True

    @dataclasses.dataclass
    class Factory(type(training[0].data.datasets[0])):
        base_config: object = dataclasses.field(default_factory=BaseData)

    base, spec, module, loaders = explicit(training, indices=(0,))
    source = base.data.datasets[0]
    base.data.datasets[0] = Factory(source.repo_id, source.assets)
    config = resolve_config(base, spec, module, loaders)
    selected = config.data.datasets[0].base_config
    assert selected.repo_id == spec["inputs"]["dataset_0"]["path"]
    assert selected.asset_id == "." and selected.norm_stats is None and selected.prompt_from_task
    assert base.data.datasets[0].base_config.repo_id == "/unregistered/base-dataset"


def test_plan_rejects_unselected_model_loader_even_with_valid_model_input(training):
    base, spec, module, loaders = explicit(training)
    config = resolve_config(base, spec, module, loaders)
    config.weight_loader = spec["inputs"]["dataset_0"]["path"]
    with pytest.raises(ValueError, match="MODEL_BINDING_MISMATCH"):
        inspect_training_config(config, spec, checks=[])


@pytest.mark.parametrize("field", ["repo_id", "assets"])
def test_legacy_single_dataset_plan_still_requires_selected_assets(training, field):
    base, spec, module, loaders = training
    source = base.data.datasets[0]
    base.data = base.data.datasets[1]
    spec["inputs"] = {"model": spec["inputs"]["model"], "dataset": spec["inputs"]["dataset_1"], "norm": spec["inputs"]["norm_1"]}
    spec.pop("dataset_bindings")
    config = resolve_config(base, spec, module, loaders)
    setattr(config.data, field, getattr(source, field))
    with pytest.raises(ValueError, match="DATASET_BINDINGS_INVALID"):
        inspect_training_config(config, spec, checks=[])


def test_single_recipe_accepts_one_explicit_pair_but_not_multiple(training):
    base, spec, module, loaders = explicit(training, indices=(0,))
    base.data = base.data.datasets[0]
    config = resolve_config(base, spec, module, loaders)
    assert config.data.repo_id == spec["inputs"]["dataset_0"]["path"]
    assert inspect_training_config(config, spec, checks=[])["weights"] == [10.]
    spec["dataset_bindings"][0]["recipe_index"] = 1
    with pytest.raises(ValueError, match="DATASET_BINDINGS_INVALID"):
        resolve_config(base, spec, module, loaders)


@pytest.mark.parametrize("case", ["missing_pair", "wrong_id", "legacy_role", "unknown_index", "no_norm"])
def test_invalid_pair_or_processing_config_cannot_fall_back_to_recipe(training, case):
    base, spec, module, loaders = explicit(training)
    if case == "missing_pair":
        spec["inputs"].pop("norm_1")
    elif case == "wrong_id":
        spec["inputs"]["dataset_1"]["id"] = "unselected-data"
    elif case == "legacy_role":
        spec["inputs"]["dataset"] = spec["inputs"]["dataset_0"]
    elif case == "unknown_index":
        spec["dataset_bindings"][1]["recipe_index"] = 3
    elif case == "no_norm":
        spec["inputs"]["norm_1"]["path"] += ".missing"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve_config(base, spec, module, loaders)


def test_plan_rejects_hidden_child_or_mispaired_norm(training):
    base, spec, module, loaders = explicit(training)
    config = resolve_config(base, spec, module, loaders)
    config.data.datasets[1].assets = config.data.datasets[0].assets
    with pytest.raises(ValueError, match="配对不一致"):
        inspect_training_config(config, spec, checks=[])
    config = resolve_config(base, spec, module, loaders)
    config.data.datasets.append(base.data.datasets[0])
    config.data.weights.append(1)
    with pytest.raises(ValueError, match="数量或权重"):
        inspect_training_config(config, spec, checks=[])


def test_launch_spec_carries_complete_ordered_bindings(training):
    base, spec, module, loaders = explicit(training)
    profile = {"code_dir": spec["code_dir"], "python": "/usr/bin/python3", "adapter": "openpi-sft", "recipe": spec["recipe"]}
    launch = build_launch(profile, "openpi-sft", spec["inputs"], spec["parameters"], spec["output_dir"],
                          ["gpu-0"], dataset_bindings=spec["dataset_bindings"])
    actual = json.loads(launch["argv"][-1])
    assert actual["dataset_bindings"] == spec["dataset_bindings"]
    assert actual["inputs"] == spec["inputs"]


def test_norm_directory_preserves_asset_identity_and_safe_symlink_target(training, tmp_path):
    base, spec, module, loaders = explicit(training)
    target = Path(spec["inputs"]["norm_0"]["path"])
    directory = tmp_path / "norm-directory"
    directory.mkdir()
    (directory / "norm_stats.json").symlink_to(target)
    spec["inputs"]["norm_0"]["path"] = str(directory)
    plan = inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])
    assert plan["datasets"][0]["norm_asset_path"] == str(directory)
    assert plan["datasets"][0]["norm_path"] == str(target)


@pytest.mark.parametrize("invalid", [[], {}, [None],
    [{"dataset_asset_id": "d", "norm_asset_id": "n", "recipe_index": True, "weight": 1}]]
    + [[{"dataset_asset_id": "d", "norm_asset_id": "n", "recipe_index": 0, "weight": weight}]
       for weight in (0, -1, float("inf"), float("nan"), True)]
    + [[{"dataset_asset_id": "d", "norm_asset_id": "n", "recipe_index": 0, "weight": 1e308}] * 2])
def test_binding_contract_rejects_invalid_values(invalid):
    with pytest.raises(ValueError, match="DATASET_BINDINGS_INVALID"):
        normalize_dataset_bindings(invalid)


def test_binding_contract_has_no_aliasing_and_legacy_is_optional():
    rows = [{"dataset_asset_id": "d", "norm_asset_id": "n", "recipe_index": 0, "weight": 1}]
    result = normalize_dataset_bindings(rows)
    result[0]["weight"] = 99
    assert rows[0]["weight"] == 1
    assert binding_inputs(rows) == {"dataset_0": "d", "norm_0": "n"}
    assert normalize_dataset_bindings(None) is None
    assert validate_binding_assets(None, {"dataset": {"path": "/legacy"}}) is None
