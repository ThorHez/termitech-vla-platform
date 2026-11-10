import ast
import copy
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import build_launch, norm_entry
from vla_adapters.automatic_config import automatic_recipe_name, normalize_auto_config
from vla_adapters.cache_contract import CACHE_ENV_KEYS
from vla_adapters.composition_spec import normalize_composition
from vla_adapters.data_processors import processor_identity
from vla_adapters.norm_datasets import validate_norm_configuration
from vla_protocol import canonical_digest
from vla_protocol.norm_plan import validate_norm_plan

from test_automatic_config import AutomaticTrainConfig, automatic, processor_module, training  # noqa: F401
from test_composed_processing_contract import composed_module, profile as composed_profile  # noqa: F401


@dataclasses.dataclass
class NormTrainConfig(AutomaticTrainConfig):
    num_workers: int = 2


@pytest.fixture
def automatic_norm(automatic, composed_module, monkeypatch):
    training_spec, module, _ = automatic
    composition = normalize_composition(composed_profile()["processing"], norm_stats=True)
    rows = [{"dataset_asset_id": row["dataset_asset_id"], "processor": {
                "name": "ComposableDataConfig", "parameters": {"composition": copy.deepcopy(composition)}}}
            for row in training_spec["dataset_bindings"][:2]]
    spec = {"template_id": "openpi-norm", "recipe": automatic_recipe_name({}),
            "auto_config": normalize_auto_config({}), "norm_datasets": rows,
            "inputs": {f"dataset_{i}": {**training_spec["inputs"][f"dataset_{i}"], "kind": "DATASET"}
                       for i in range(len(rows))},
            "code_dir": training_spec["code_dir"], "output_dir": training_spec["output_dir"],
            "parameters": {"batch_size": 32, "num_workers": 0}, "gpu_count": 0,
            "allowed_roots": training_spec["allowed_roots"]}
    module.TrainConfig = NormTrainConfig
    module.DataConfigFactory = composed_module.DataConfigFactory
    module.ComposableDataConfig = composed_module.ComposableDataConfig
    module.IndependentFactory = composed_module.IndependentFactory
    original_contract = module.ComposableDataConfig.processing_contract

    def processing_contract(self, model, *, norm_stats=False):
        result = original_contract(self, model)
        if norm_stats:
            result.update(image_keys=[], depth_keys=[])
        return result

    monkeypatch.setattr(module.ComposableDataConfig, "processing_contract", processing_contract)
    source = Path(spec["code_dir"]) / "src/openpi/training/composed_processing.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# Registered composition contract fixture\n")
    numeric = ModuleType("openpi.training.numeric_stats")
    numeric.__file__ = str(source.with_name("numeric_stats.py"))
    Path(numeric.__file__).write_text("# Registered numeric projection fixture\n")
    for asset in spec["inputs"].values():
        info_path = Path(asset["path"]) / "meta/info.json"
        info = json.loads(info_path.read_text())
        info["features"] = {field["key"]: {"dtype": "float32", "shape": [field["width"]]}
                            for field in composition["state_fields"]}
        info["features"][composition["action_key"]] = {"dtype": "float32", "shape": [10]}
        info_path.write_text(json.dumps(info))
    module.__file__ = str(Path(spec["code_dir"]) / "config.py")
    module.get_config = lambda _: pytest.fail("Automatic statistics must not look up a code recipe")
    package = ModuleType("openpi.training"); package.config = module; package.numeric_stats = numeric
    root = ModuleType("openpi"); root.training = package
    for name, value in [("openpi", root), ("openpi.training", package), ("openpi.training.config", module),
                        ("openpi.training.numeric_stats", numeric)]:
        monkeypatch.setitem(sys.modules, name, value)
    monkeypatch.setattr(norm_entry, "configure_cache", lambda _: None)
    monkeypatch.setattr(norm_entry, "install_registered_fps", lambda _: None)
    for key in CACHE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return spec, module


def numeric_evidence(config):
    composition = config.data.composition
    states = [field["key"] for field in composition["state_fields"]]
    source = Path(sys.modules["openpi.training.numeric_stats"].__file__)
    return {"schema_version": 3, "reader": "pyarrow_parquet_projection",
            "columns": [*states, composition["action_key"]],
            "state_columns": states, "action_column": composition["action_key"],
            "temporal_columns": [], "action_sequence": "per_frame",
            "image_columns": [], "video_decode": False,
            "source": {"path": "src/openpi/training/numeric_stats.py", "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}}


def test_automatic_norm_builds_without_recipe_weights_norms_or_mixture(automatic_norm):
    spec, _ = automatic_norm
    profile = {"adapter": "openpi-sft", "recipe": "old_profile_recipe",
               "code_dir": spec["code_dir"], "python": sys.executable}
    launch = build_launch(profile, "openpi-norm", spec["inputs"], {}, spec["output_dir"], [],
                          auto_config={"action_horizon": 16}, norm_datasets=spec["norm_datasets"])
    frozen = json.loads(launch["argv"][-1])
    assert frozen["recipe"] == "auto_pi05_full_h16_d32_sft"
    assert frozen["auto_config"] == normalize_auto_config({"action_horizon": 16})
    configs = [norm_entry.prepare_config({**frozen, "norm_dataset_index": i}) for i in range(2)]
    assert all(config.model.pi05 and config.model.action_horizon == 16 for config in configs)
    assert all(config.weight_loader is None and not hasattr(config.data, "datasets") for config in configs)
    assert [config.data.repo_id for config in configs] == [asset["path"] for asset in spec["inputs"].values()]
    assert all(config.data.assets.assets_dir.endswith("/.unbound") for config in configs)
    assert all(config.batch_size == 32 and config.num_workers == 0 for config in configs)


def test_automatic_norm_composition_uses_each_rows_model_layout_independently(automatic_norm, composed_module):
    spec, module = automatic_norm
    module.DataConfigFactory = composed_module.DataConfigFactory
    module.ComposableDataConfig = composed_module.ComposableDataConfig
    first = composed_profile()["processing"]
    second = copy.deepcopy(first)
    first["cameras"][0].update(slot="base_0_rgb", kind="rgb")
    second["cameras"][0].update(slot="base_0_rgb", kind="depth")
    second["state_history"] = 6
    for row, composition in zip(spec["norm_datasets"], [first, second]):
        row["processor"] = {"name": "ComposableDataConfig", "parameters": {"composition": composition}}
    configs = [norm_entry.prepare_config({**spec, "norm_dataset_index": i}) for i in range(2)]
    assert all(config.model.image_keys == () and config.model.depth_keys == () for config in configs)
    assert all(config.data.composition["cameras"] == [] for config in configs)
    assert [config.data.composition["state_history"] for config in configs] == [1, 6]
    assert all(config.data.training_stage == "sft" for config in configs)
    assert all(config.model.action_horizon == spec["auto_config"]["action_horizon"] for config in configs)


@pytest.mark.parametrize("auto,rows,error", [
    ({}, None, "AUTO_CONFIG_PROCESSOR_REQUIRED"),
    ({}, [{"dataset_asset_id": "a", "recipe_index": 0}], "AUTO_CONFIG_PROCESSOR_REQUIRED"),
    ({"finetune_mode": "lora"}, [{"dataset_asset_id": "a", "processor": {"name": "IndependentFactory", "parameters": {}}}], "AUTO_CONFIG_UNSUPPORTED"),
])
def test_automatic_norm_rejects_recipe_slots_missing_processors_and_training_mode(auto, rows, error):
    with pytest.raises(ValueError, match=error):
        validate_norm_configuration(auto, rows)


def test_norm_automatic_configuration_cannot_fall_back_to_named_recipe(automatic_norm):
    spec, _ = automatic_norm
    with pytest.raises(ValueError, match="AUTO_CONFIG_CONFLICT"):
        norm_entry.prepare_config({**spec, "recipe": "old_code_recipe"})


def test_norm_rejects_unproven_legacy_processor_without_visual_fallback(automatic_norm):
    spec, _ = automatic_norm
    legacy = copy.deepcopy(spec)
    legacy["norm_datasets"][0]["processor"] = {"name": "IndependentFactory", "parameters": {}}
    with pytest.raises(ValueError, match="NORM_NUMERIC_ONLY_UNSUPPORTED"):
        norm_entry.prepare_config(legacy)


@pytest.mark.parametrize("packed", [False, True])
def test_norm_preflight_uses_numeric_projection_without_sample_model_or_tokenizer_creation(automatic_norm, monkeypatch, packed):
    spec, _ = automatic_norm
    config = norm_entry.prepare_config(spec)
    action_shape = [config.model.action_horizon, 10] if packed else [10]
    info_path = Path(spec["inputs"]["dataset_0"]["path"]) / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["features"][config.data.composition["action_key"]]["shape"] = action_shape
    info_path.write_text(json.dumps(info))
    calls = []
    selected = SimpleNamespace(repo_id=spec["inputs"]["dataset_0"]["path"])

    def project(factory, model, *, effective_fps):
        calls.append((factory, model.action_horizon, effective_fps))
        return selected

    monkeypatch.setattr(type(config.data), "processing_sample_config", lambda *args: pytest.fail("Do not construct the visual sample pipeline"), raising=False)
    monkeypatch.setattr(type(config.data), "create", lambda *args: pytest.fail("Do not construct a model/tokenizer during statistics preflight"))
    numeric_projection = sys.modules["openpi.training.numeric_stats"]
    monkeypatch.setattr(numeric_projection, "build_numeric_statistics_config", project, raising=False)
    monkeypatch.setattr(numeric_projection, "numeric_io_evidence", lambda data: {
        **numeric_evidence(config), "action_sequence": "packed_all_steps" if packed else "per_frame",
    }, raising=False)

    actions = SimpleNamespace(ndim=1 + len(action_shape), size=10, shape=(1, *action_shape))
    state = SimpleNamespace(ndim=2, size=7, shape=(1, 7))

    numeric = ModuleType("numpy")
    numeric.asarray = lambda value: value
    numeric.isfinite = lambda _: SimpleNamespace(all=lambda: True)
    monkeypatch.setitem(sys.modules, "numpy", numeric)
    probe_config = SimpleNamespace(data=config.data, model=config.model, assets_dirs="unused-assets",
                                   batch_size=config.batch_size)
    statistics = SimpleNamespace(create_torch_dataloader=lambda *args, **kwargs:
                                 (iter([{"state": state, "actions": actions}]), 1))
    result = norm_entry.inspect_dataset(probe_config, spec, statistics, "dataset_0", spec["norm_datasets"][0])
    assert calls == [(config.data, spec["auto_config"]["action_horizon"], None)]
    assert result["dataset_asset_id"] == spec["norm_datasets"][0]["dataset_asset_id"]
    assert result["numeric_io"]["image_columns"] == [] and result["numeric_io"]["video_decode"] is False
    assert result["composition"]["spec"]["cameras"] == []
    assert result["sample_shapes"]["actions"] == action_shape
    if packed:
        actions.ndim, actions.shape = 2, (1, 10)
        with pytest.raises(ValueError, match="NORM_SAMPLE_INVALID.*形状"):
            norm_entry.inspect_dataset(probe_config, spec, statistics, "dataset_0", spec["norm_datasets"][0])
    monkeypatch.setattr(numeric_projection, "numeric_io_evidence", lambda data: {
        **numeric_evidence(config), "schema_version": 2, "action_sequence": "packed_first_step",
    })
    with pytest.raises(ValueError, match="NORM_RUNTIME_UNSUPPORTED.*原始动作形状"):
        norm_entry.inspect_dataset(probe_config, spec, statistics, "dataset_0", spec["norm_datasets"][0])


def test_norm_auto_plan_freezes_all_models_and_generated_config_and_rejects_drift(automatic_norm, monkeypatch):
    spec, _ = automatic_norm
    entry = Path(spec["code_dir"]) / "scripts/compute_norm_stats.py"
    entry.write_text("# registered statistics script\n")

    def inspect(config, selected, statistics, role, row):
        path = Path(selected["inputs"][role]["path"])
        contract = norm_entry.dataset_contract(config.data, selected, model_config=config.model,
                                               binding=row, norm_stats=True)
        return {"path": str(path), "input_role": role, "dataset_asset_id": row["dataset_asset_id"],
                "processor": row["processor"], "processor_identity": processor_identity(config.data, selected["code_dir"]),
                "factory": type(config.data).__name__, "frames": 20,
                "metadata_sha256": hashlib.sha256((path / "meta/info.json").read_bytes()).hexdigest(),
                "manifest_sha256": norm_entry.norm_manifest(path, selected)[2],
                "composition": contract["composition"], "actions_shape": contract["actions_shape"],
                "numeric_io": numeric_evidence(config),
                "sample_shapes": {"state": [7], "actions": [10]},
                "model_configuration": {"model_type": "pi05" if config.model.pi05 else "pi0",
                    "action_horizon": config.model.action_horizon, "action_dim": config.model.action_dim}}

    monkeypatch.setattr(norm_entry, "inspect_dataset", inspect)

    def plan(selected):
        configs = [norm_entry.prepare_config({**selected, "norm_dataset_index": i}) for i in range(2)]
        return norm_entry.inspect_config(configs, selected, object(), entry)

    first = plan(spec)
    validate_norm_plan(first, spec, spec["inputs"])
    assert first["auto_config"] == spec["auto_config"]
    assert len(first["datasets"]) == 2
    assert first["training_cache"]["mode"] == "numeric_projection"
    assert first["training_cache"]["persistent_cache"] is False
    assert first["training_cache"]["datasets"] == [{"path": data["path"], "columns": data["numeric_io"]["columns"]}
                                                  for data in first["datasets"]]
    assert canonical_digest(first) == canonical_digest(plan({**spec, "output_dir": "/other/attempt"}))
    changed = {**spec, "auto_config": normalize_auto_config({"action_horizon": 16})}
    changed["recipe"] = automatic_recipe_name(changed["auto_config"])
    second = plan(changed)
    assert canonical_digest(first) != canonical_digest(second)
    assert first["generated_config"]["sha256"] != second["generated_config"]["sha256"]
    with pytest.raises(ValueError, match="automatic model configuration"):
        validate_norm_plan({**first, "auto_config": changed["auto_config"]}, spec, spec["inputs"])
    tampered = copy.deepcopy(first)
    tampered["datasets"][1]["model_configuration"]["action_horizon"] = 7
    with pytest.raises(ValueError, match="dataset model configuration"):
        validate_norm_plan(tampered, spec, spec["inputs"])


@pytest.mark.parametrize("sample_api", [True, False])
@pytest.mark.parametrize("numeric_only", [True, False])
def test_actual_statistics_script_main_uses_projection_or_preserves_explicit_legacy_cli(tmp_path, monkeypatch, sample_api, numeric_only):
    script = Path(__file__).resolve().parents[2] / "third_party/openpi-umi/scripts/compute_norm_stats.py"
    node = next(node for node in ast.parse(script.read_text()).body if getattr(node, "name", None) == "main")
    calls = []
    data = SimpleNamespace(rlds_data_dir=None)

    class Factory:
        def create(self, assets, model):
            assert not numeric_only, "Numeric-only statistics must not fall back to create"
            assert not sample_api, "Model/tokenizer construction must never run for a sample-capable processor"
            calls.append("legacy_create")
            return data

    factory = Factory()
    if sample_api:
        def processing_sample_config(assets, model):
            assert not numeric_only, "Numeric-only statistics must not construct visual sample transforms"
            calls.append("sample_without_tokenizer")
            return data
        factory.processing_sample_config = processing_sample_config
    config = SimpleNamespace(data=factory, assets_dirs=tmp_path, model=SimpleNamespace(action_horizon=16),
                             batch_size=32, num_workers=0)

    class RunningStats:
        def update(self, value):
            assert value == [1., 2.]

        def get_statistics(self):
            return {"mean": [1.5], "std": [.5], "q01": [1.], "q99": [2.]}

    modules = {name: ModuleType(name) for name in ["openpi", "openpi.shared", "openpi.training",
               "openpi.shared.runtime_paths", "openpi.shared.normalize", "openpi.training.config",
               "openpi.training.numeric_stats", "tqdm"]}
    modules["openpi.shared.runtime_paths"].configure_training_cache = lambda: None
    modules["openpi.shared.normalize"].RunningStats = RunningStats
    modules["openpi.shared.normalize"].serialize_json = lambda stats: json.dumps({"norm_stats": stats})
    modules["openpi.training.config"].MultiDataConfigFactory = type("MultiDataConfigFactory", (), {})
    def project(selected_factory, model, *, effective_fps):
        assert selected_factory is factory and model is config.model and effective_fps is None
        calls.append("numeric_projection")
        return data
    modules["openpi.training.numeric_stats"].build_numeric_statistics_config = project
    modules["tqdm"].tqdm = lambda loader, **kwargs: loader
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    output = tmp_path / "output"

    def publish(path, document, *, overwrite):
        path.mkdir()
        (path / "norm_stats.json").write_text(json.dumps(document))

    namespace = {"np": SimpleNamespace(asarray=lambda value: value), "json": json,
        "_load_file_config": lambda *args: config, "_local_path": Path,
        "_check_output": lambda *args, **kwargs: None, "_publish": publish,
        "create_torch_dataloader": lambda *args: ([{"state": [1., 2.], "actions": [1., 2.]}], 1)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(script), "exec"), namespace)
    namespace["main"](config_file=str(tmp_path / "training_config.py"), output_dir=str(output), numeric_only=numeric_only)
    assert calls == ["numeric_projection" if numeric_only else "sample_without_tokenizer" if sample_api else "legacy_create"]
    assert json.loads((output / "norm_stats.json").read_text())["norm_stats"]["state"]["mean"] == [1.5]
