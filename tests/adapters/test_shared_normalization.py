"""Shared raw-sample statistics retain explicit assets and reproducible provenance."""

import copy
import dataclasses
from functools import partial
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

from vla_adapters import openpi_entry
from vla_adapters.openpi_entry import resolve_config
from vla_adapters.recipe_definitions import validate_overrides
from vla_adapters.training_plan import fingerprint, inspect_training_config

from test_training_plan import Multi, training  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/openpi-umi/src/openpi/training/shared_norm.py"
SPEC = importlib.util.spec_from_file_location("shared_normalization_contract", SOURCE)
shared = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(shared)


def numeric_scale(value, *, scale=1.):
    return value * scale


@dataclasses.dataclass
class SharedMulti(Multi):
    shared_norm_stats_path: str | None = None


def artifact(factories, weights):
    return {"norm_stats": {
        key: {"mean": [0., 0.], "std": [1., 1.], "q01": [-2., -3.], "q99": [4., 5.],
              "quantile_method": "q01_q99"} for key in ("state", "actions")
    }, "shared_normalization": {
        "schema_version": 1, "method": "weighted_raw_samples", "weight_semantics": "per_sample_repeat",
        "coverage": "full", "quantile_method": "weighted_histogram", "quantile_bins": 5000,
        "sources": [{**source, "rows": 20, "state_vectors": 20, "action_vectors": 320}
                    for source in shared.source_descriptors(factories, weights)],
    }}


def shared_training(training):
    base, spec, module, loaders = training
    base.data = SharedMulti(**{field.name: getattr(base.data, field.name) for field in dataclasses.fields(Multi)})
    module.MultiDataConfigFactory = SharedMulti
    validator = Path(spec["code_dir"]) / "src/openpi/training/shared_norm.py"
    validator.parent.mkdir(parents=True)
    validator.write_bytes(SOURCE.read_bytes())
    norm = Path(spec["output_dir"]).parent / "shared/norm_stats.json"
    norm.parent.mkdir()
    for i, row in enumerate(spec["dataset_bindings"]):
        row["norm_asset_id"] = "shared-norm"
        spec["inputs"][f"norm_{i}"] = {"id": "shared-norm", "path": str(norm), "kind": "NORM"}
    document = artifact(base.data.datasets, [row["weight"] for row in spec["dataset_bindings"]])
    norm.write_text(json.dumps(document))
    return base, spec, module, loaders, norm, document


def test_distinct_norm_assets_disable_old_quantile_merger_and_preserve_recipe(training):
    base, spec, module, loaders = training
    original = copy.deepcopy(base)
    config = resolve_config(base, spec, module, loaders)
    assert config.data.use_merged_norm_stats is False
    assert base == original and base.data.use_merged_norm_stats is True
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["normalization"] == {"mode": "per_dataset"}
    assert len({row["norm_path"] for row in plan["datasets"]}) == 3


def test_same_shared_asset_uses_one_validated_file_and_freezes_provenance(training):
    base, spec, module, loaders, norm, document = shared_training(training)
    config = resolve_config(base, spec, module, loaders)
    assert config.data.use_merged_norm_stats is True
    assert config.data.shared_norm_stats_path == str(norm)
    assert len({child.assets.assets_dir for child in config.data.datasets}) == 1
    plan = inspect_training_config(config, spec, checks=[])
    assert plan["normalization"]["mode"] == "shared"
    assert plan["normalization"]["shared"]["metadata"] == document["shared_normalization"]
    assert all(row["norm_asset_id"] == "shared-norm" for row in plan["datasets"])
    assert len({row["norm_sha256"] for row in plan["datasets"]}) == 1
    assert plan["use_merged_norm_stats"] is True


def test_shared_selection_requires_new_runtime_field_and_validator(training):
    base, spec, module, loaders, _, _ = shared_training(training)
    base.data = Multi(**{field.name: getattr(base.data, field.name) for field in dataclasses.fields(Multi)})
    module.MultiDataConfigFactory = Multi
    with pytest.raises(ValueError, match="SHARED_NORM_UNSUPPORTED.*升级"):
        resolve_config(base, spec, module, loaders)
    base.data = SharedMulti(**{field.name: getattr(base.data, field.name) for field in dataclasses.fields(Multi)})
    module.MultiDataConfigFactory = SharedMulti
    (Path(spec["code_dir"]) / "src/openpi/training/shared_norm.py").unlink()
    with pytest.raises(ValueError, match="SHARED_NORM_UNSUPPORTED.*升级"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])


@pytest.mark.parametrize("change", ["weight", "metadata", "parquet", "source", "processor"])
def test_changed_generation_inputs_cannot_reuse_shared_statistics(training, change):
    base, spec, module, loaders, _, _ = shared_training(training)
    if change == "weight":
        spec["dataset_bindings"][0]["weight"] = 7.
    elif change == "metadata":
        info = Path(base.data.datasets[0].repo_id) / "meta/info.json"
        value = json.loads(info.read_text()); value["extra_source_fact"] = "changed"
        info.write_text(json.dumps(value))
    elif change == "parquet":
        path = Path(base.data.datasets[0].repo_id) / "data/episode_000000.parquet"
        path.write_bytes(path.read_bytes() + b"changed")
    elif change == "source":
        spec["inputs"]["dataset_0"]["path"] = spec["inputs"]["dataset_1"]["path"]
    else:
        @dataclasses.dataclass
        class ChangedFactory(type(base.data.datasets[0])):
            position_unit: str = "mm"
        item = base.data.datasets[0]
        base.data.datasets[0] = ChangedFactory(item.repo_id, item.assets)
    with pytest.raises(ValueError, match="SHARED_NORM_INVALID"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])


def test_shared_file_drift_after_preflight_is_rejected_before_training(training, monkeypatch):
    base, spec, module, loaders, norm, document = shared_training(training)
    config = resolve_config(base, spec, module, loaders)
    frozen = inspect_training_config(config, spec, checks=[])
    spec["expected_plan_digest"] = fingerprint(frozen)
    document["norm_stats"]["state"]["mean"][0] = 0.25
    norm.write_text(json.dumps(document))
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda _: (config, None))
    with pytest.raises(ValueError, match="ASSETS_CHANGED_SINCE_PREFLIGHT"):
        openpi_entry._run_spec(spec)


@pytest.mark.parametrize("change", ["legacy", "partial", "old_quantile", "wrong_bins", "negative_std"])
def test_legacy_or_incomplete_statistics_cannot_claim_full_shared_provenance(training, change):
    base, spec, module, loaders, norm, document = shared_training(training)
    if change == "legacy": del document["shared_normalization"]
    elif change == "partial": document["shared_normalization"]["sources"][0]["rows"] = 19
    elif change == "old_quantile": document["norm_stats"]["actions"].pop("quantile_method")
    elif change == "wrong_bins": document["shared_normalization"]["quantile_bins"] = 100
    else: document["norm_stats"]["actions"]["std"][0] = -1
    norm.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="SHARED_NORM_INVALID|NORM_FORMAT"):
        inspect_training_config(resolve_config(base, spec, module, loaders), spec, checks=[])


def test_same_width_different_layout_cannot_share_statistics(training):
    @dataclasses.dataclass
    class NumericFactory:
        repo_id: str
        layout: tuple
    factories = [NumericFactory(child.repo_id, ("joint", "gripper")) for child in training[0].data.datasets]
    factories[1].layout = ("gripper", "joint")
    with pytest.raises(ValueError, match="layouts, ordering"):
        shared.source_descriptors(factories, [1., 1., 1.])


def test_composable_compatibility_ignores_camera_and_prompt_but_retains_numeric_semantics(training):
    @dataclasses.dataclass
    class ComposableDataConfig:
        repo_id: str
        composition: dict
    composition = {"state_fields": [{"key": "joints", "width": 2, "normalize": True}],
                   "action_layout": [{"kind": "joint", "width": 2, "normalize": True}],
                   "position_unit": "m", "cameras": [{"key": "camera_a"}], "prompt_key": "task"}
    factories = [ComposableDataConfig(child.repo_id, copy.deepcopy(composition)) for child in training[0].data.datasets]
    factories[1].composition.update(cameras=[{"key": "camera_b"}], prompt_key="instruction")
    descriptors = shared.source_descriptors(factories, [1., 2., 3.])
    assert descriptors[0]["processor_sha256"] != descriptors[1]["processor_sha256"]
    assert len({row["compatibility_sha256"] for row in descriptors}) == 1
    factories[1].composition["position_unit"] = "mm"
    with pytest.raises(ValueError, match="SHARED_NORM_INVALID"):
        shared.source_descriptors(factories, [1., 2., 3.])


@pytest.mark.parametrize("value", ["/unregistered/norm_stats.json", {"shared_norm_stats_path": "/unregistered/norm_stats.json"}])
def test_shared_file_cannot_be_injected_through_saved_recipe_overrides(value):
    path = "data.shared_norm_stats_path" if isinstance(value, str) else "data.base_config"
    with pytest.raises(ValueError, match="ASSET|asset|资产"):
        validate_overrides({path: value})


def processor_checkout(tmp_path, name):
    package = tmp_path / name / "src/openpi"
    source = package / "training/processor.py"
    source.parent.mkdir(parents=True)
    source.write_text("from dataclasses import dataclass\n"
                      "@dataclass\nclass Processor:\n    repo_id: str\n    scale: float = 1.0\n")
    for relative in ("training/composed_processing.py", "transforms.py", "policies/umi_policy.py"):
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# numeric processing implementation\n")
    return package


def source_factories(package, training, monkeypatch):
    source = package / "training/processor.py"
    spec = importlib.util.spec_from_file_location("shared_processor_fixture", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    exec(compile(source.read_bytes(), str(source), "exec"), module.__dict__)
    return [module.Processor(child.repo_id) for child in training[0].data.datasets]


@pytest.mark.parametrize("relative", ["training/processor.py", "training/composed_processing.py",
                                      "transforms.py", "policies/umi_policy.py"])
def test_factory_or_delegated_numeric_implementation_changes_require_new_statistics(
        training, tmp_path, monkeypatch, relative):
    package = processor_checkout(tmp_path, "original-code")
    factories = source_factories(package, training, monkeypatch)
    weights = [1., 2., 3.]
    document = artifact(factories, weights)
    assert shared.validate_shared_metadata(document, factories, weights) == document["shared_normalization"]
    source = package / relative
    source.write_text(source.read_text() + "\n# changed numeric implementation\n")
    current = shared.source_descriptors(factories, weights)
    assert current[0]["processor_sha256"] != document["shared_normalization"]["sources"][0]["processor_sha256"]
    with pytest.raises(ValueError, match="processor changed.*recompute"):
        shared.validate_shared_metadata(document, factories, weights)


def test_same_implementation_relocated_to_another_checkout_retains_identity(training, tmp_path, monkeypatch):
    package = processor_checkout(tmp_path, "first-checkout")
    factories = source_factories(package, training, monkeypatch)
    weights = [1., 2., 3.]
    before = shared.source_descriptors(factories, weights)
    copied = tmp_path / "another-checkout/src/openpi"
    shutil.copytree(package, copied)
    relocated = source_factories(copied, training, monkeypatch)
    assert shared.source_descriptors(relocated, weights) == before
    assert all(not Path(name).is_absolute() for name in shared._implementation(relocated[0]))


def test_anonymous_dynamic_factory_without_real_class_source_is_rejected(training):
    anonymous = dataclasses.make_dataclass("UninspectableFactory", [("repo_id", str)])
    factories = [anonymous(child.repo_id) for child in training[0].data.datasets]
    with pytest.raises(ValueError, match="implementation cannot be verified|no inspectable Python source"):
        shared.source_descriptors(factories, [1., 2., 3.])


def test_loaded_norm_metadata_and_base_cache_do_not_change_processor_identity(training):
    @dataclasses.dataclass
    class CachedFactory:
        repo_id: str
        base_config: dict
        norm_stats_metadata: object = None
    factories = [CachedFactory(child.repo_id, {"use_quantile_norm": True, "norm_stats": None,
                                               "norm_stats_metadata": None})
                 for child in training[0].data.datasets]
    before = shared.source_descriptors(factories, [1., 2., 3.])
    for factory in factories:
        factory.norm_stats_metadata = {"shared_normalization": "loaded derived provenance"}
        factory.base_config.update(norm_stats={"state": "loaded cache"}, norm_stats_metadata={"derived": True})
    assert shared.source_descriptors(factories, [1., 2., 3.]) == before


def test_finite_per_source_weights_remain_valid_when_their_naive_sum_overflows(training):
    weights = [1e308, 1e308, 1e308]
    factories = training[0].data.datasets
    document = artifact(factories, weights)
    assert [source["weight"] for source in shared.source_descriptors(factories, weights)] == weights
    assert shared.validate_shared_metadata(document, factories, weights) == document["shared_normalization"]


def test_single_read_document_loader_returns_the_validated_document(training, tmp_path, monkeypatch):
    factories, weights = training[0].data.datasets, [1., 2., 3.]
    document = artifact(factories, weights)
    norm = tmp_path / "one-read/norm_stats.json"
    norm.parent.mkdir()
    norm.write_text(json.dumps(document))
    original = shared._read_document
    reads = []
    def read_once(path):
        if Path(path) == norm:
            reads.append(path)
            assert len(reads) == 1
        return original(path)
    monkeypatch.setattr(shared, "_read_document", read_once)
    assert shared.load_shared_norm_document(norm, factories, weights) == document
    assert reads == [norm]


@pytest.mark.parametrize("parameter", ["args", "keywords"])
def test_partial_numeric_parameters_are_frozen_and_changes_require_new_statistics(training, parameter):
    @dataclasses.dataclass
    class PartialFactory:
        repo_id: str
        transform: object
    factories = [PartialFactory(child.repo_id, partial(numeric_scale, 2., scale=3.))
                 for child in training[0].data.datasets]
    weights = [1., 2., 3.]
    document = artifact(factories, weights)
    assert shared.validate_shared_metadata(document, factories, weights) == document["shared_normalization"]
    for factory in factories:
        factory.transform = partial(numeric_scale, 4. if parameter == "args" else 2.,
                                    scale=5. if parameter == "keywords" else 3.)
    with pytest.raises(ValueError, match="processor changed.*recompute"):
        shared.validate_shared_metadata(document, factories, weights)


@pytest.mark.parametrize("callable_kind", ["stateful_instance", "bound_method", "bound_builtin", "anonymous_partial"])
def test_unproven_callable_state_cannot_enter_shared_processor_provenance(training, callable_kind):
    class StatefulScale:
        def __init__(self):
            self.scale = 2.
        def __call__(self, value):
            return value * self.scale
        def transform(self, value):
            return value * self.scale
    @dataclasses.dataclass
    class CallableFactory:
        repo_id: str
        transform: object
    value = StatefulScale()
    transform = {"stateful_instance": value, "bound_method": value.transform,
                 "bound_builtin": [2.].append, "anonymous_partial": partial(lambda value: value * 2.)}[callable_kind]
    factories = [CallableFactory(child.repo_id, transform) for child in training[0].data.datasets]
    with pytest.raises(ValueError, match="SHARED_NORM_INVALID.*(stateful callable|bound processor|anonymous processor)"):
        shared.source_descriptors(factories, [1., 2., 3.])
