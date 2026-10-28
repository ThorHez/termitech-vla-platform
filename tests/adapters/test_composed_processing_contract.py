"""A shared data contract connects UI assets, frozen choices and training code."""

from copy import deepcopy
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vla_adapters.composition_spec import (composition_choice, composition_digest, normalize_composition,
                                           validate_composition_profile_input, validate_profile_composition,
                                           verify_composition_metadata)
from vla_adapters.data_processors import (catalog_data_processors, instantiate_data_processor, normalize_processor,
                                         validate_processor_instance)
from vla_adapters.dataset_contracts import verify_registered_dataset_profile
from vla_adapters import training_plan, recap_runtime
from vla_adapters.recipe_definitions import validate_overrides
from test_data_processors import SOURCE, processor_module  # noqa: F401
from test_training_plan import training  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
CORPUS = json.loads((ROOT / "tests/fixtures/composed-processing-v1.json").read_text())


def profile():
    spec = normalize_composition(CORPUS["valid"][0])
    return {"schema_version": 1, "format": "lerobot", "robot_type": spec["robot_model"],
            "arm_mode": spec["arm_mode"], "action_representation": spec["action_representation"],
            "cameras": [{"name": "wrist", "key": c["key"], "kind": c["kind"]} for c in spec["cameras"]],
            "task_description": "Pick up the cup", "processing": spec, "action_dim": 10, "state_dim": 7}


def metadata():
    return {"features": {"observation.joints": {"shape": [6]}, "observation.gripper": {"shape": [1]},
                         "observation.images.wrist": {"shape": [3, 224, 224], "dtype": "image"},
                         "actions": {"shape": [16, 10]}, "task": {"dtype": "string", "shape": [1]}}}


@pytest.fixture(scope="module")
def core():
    path = ROOT / "third_party/openpi-umi/src/openpi/training/composed_processing.py"
    loader = importlib.util.spec_from_file_location("composed_contract_core", path)
    module = importlib.util.module_from_spec(loader)
    sys.modules[loader.name] = module
    loader.loader.exec_module(module)
    yield module
    sys.modules.pop(loader.name, None)


@pytest.mark.parametrize("value", CORPUS["valid"])
def test_control_and_training_normalizers_agree_without_model_dependencies(core, value):
    original = deepcopy(value)
    assert core.normalize_composition(value) == normalize_composition(value)
    assert value == original
    assert composition_digest(value) == composition_digest(normalize_composition(value))


@pytest.mark.parametrize("value", CORPUS["invalid"])
def test_both_normalizers_reject_incomplete_or_ambiguous_compositions(core, value):
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(value)


def test_declarations_have_independent_state_and_action_encodings_and_no_dimension_guessing():
    spec = normalize_composition(CORPUS["valid"][0])
    assert spec["state_representation"] == "joint" and spec["action_representation"] == "eef"
    assert sum(x["width"] for x in spec["state_fields"]) == 7
    assert sum(x["width"] for x in spec["action_layout"]) == 10
    value = deepcopy(spec); value["gripper_range"][1] = 10 ** 10000
    with pytest.raises(ValueError): normalize_composition(value)


@pytest.mark.parametrize("declaration", [None, "", " \t\ufeff\u00a0"])
def test_unused_declarations_may_be_blank_without_guessing_values(core, declaration):
    spec = deepcopy(CORPUS["valid"][0])
    for key in ("position_unit", "joint_unit", "coordinate_frame"):
        spec[key] = declaration
    spec["gripper_range"] = None
    for normalize in (normalize_composition, core.normalize_composition):
        result = normalize(spec)
        assert all(result[key] == "" for key in ("position_unit", "joint_unit", "coordinate_frame"))
        assert result["gripper_range"] is None
        assert result["action_layout"] == spec["action_layout"]


def test_omitted_optional_declarations_and_bounds_have_a_stable_canonical_form(core):
    spec = deepcopy(CORPUS["valid"][0])
    for key in ("position_unit", "joint_unit", "coordinate_frame", "gripper_range"):
        spec.pop(key)
    result = normalize_composition(spec)
    assert core.normalize_composition(spec) == result
    assert result["position_unit"] == result["joint_unit"] == result["coordinate_frame"] == ""
    assert result["gripper_range"] is None
    assert composition_digest(spec) == composition_digest(result)
    assert any(block["kind"] == "gripper" for block in result["action_layout"])


@pytest.mark.parametrize("key,value", [
    ("position_unit", "cm"), ("position_unit", " m "), ("position_unit", False),
    ("joint_unit", "radians"), ("joint_unit", 0),
    ("coordinate_frame", "base\x00"), ("coordinate_frame", "x" * 101), ("coordinate_frame", 0),
    ("gripper_range", []), ("gripper_range", [0]), ("gripper_range", [1, 0]),
    ("gripper_range", [float("nan"), 1]), ("gripper_range", [0, float("inf")]),
])
def test_optional_declarations_still_reject_invalid_values_when_supplied(core, key, value):
    spec = deepcopy(CORPUS["valid"][0]); spec[key] = value
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(spec)


@pytest.mark.parametrize("key", ["state_history", "normalization", "state_fields", "action_layout"])
def test_actual_processing_requirements_cannot_be_omitted(core, key):
    spec = deepcopy(CORPUS["valid"][0]); spec.pop(key)
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(spec)


@pytest.mark.parametrize("value", [None, "", " \t\ufeff\u00a0"])
def test_encoding_and_prompt_declarations_are_optional_without_guessing(core, value):
    spec = deepcopy(CORPUS["valid"][0])
    for key in ("action_mode", "rotation_representation", "prompt_key"):
        spec[key] = value
    for normalize in (normalize_composition, core.normalize_composition):
        result = normalize(spec)
        assert all(result[key] == "" for key in ("action_mode", "rotation_representation", "prompt_key"))
        assert result["action_layout"] == spec["action_layout"]


@pytest.mark.parametrize("key,value", [("action_mode", "unknown"), ("action_mode", False),
    ("rotation_representation", "euler"), ("rotation_representation", 0),
    ("prompt_key", "/tmp/task"), ("prompt_key", "task()"), ("prompt_key", 0)])
def test_supplied_optional_encoding_and_prompt_declarations_remain_validated(core, key, value):
    spec = deepcopy(CORPUS["valid"][0]); spec[key] = value
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(spec)


@pytest.mark.parametrize("width", [3, 4, 6])
def test_blank_rotation_keeps_generic_eef_layout_checks(core, width):
    spec = deepcopy(CORPUS["valid"][0]); spec["rotation_representation"] = ""
    next(block for block in spec["action_layout"] if block["kind"] == "rotation")["width"] = width
    assert core.normalize_composition(spec) == normalize_composition(spec)
    spec["action_layout"][0]["width"] = 2
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(spec)


def test_blank_rotation_accepts_joint_layout_and_rejects_unknown_eef_width(core):
    spec = deepcopy(CORPUS["valid"][0]); spec["rotation_representation"] = ""
    next(block for block in spec["action_layout"] if block["kind"] == "rotation")["width"] = 5
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(spec)
    spec["action_representation"] = "joint"
    spec["action_layout"] = [{"arm": "left", "kind": "joint", "width": 6, "normalize": True}]
    assert core.normalize_composition(spec) == normalize_composition(spec)
    spec["rotation_representation"] = "quaternion"
    for normalize in (normalize_composition, core.normalize_composition):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize(spec)


@pytest.mark.parametrize("declared", ["", "absolute", "delta"])
@pytest.mark.parametrize("source", ["unknown", "absolute", "delta"])
def test_action_mode_only_compares_explicit_declaration_with_known_source(declared, source):
    row = profile(); spec = row.pop("processing"); spec["action_mode"] = declared
    row["field_annotations"] = [
        {"key": field["key"], "role": "state", "representation": "joint", "value_mode": "absolute",
         "component": field["kind"], "arm": field["arm"]} for field in spec["state_fields"]]
    row["field_annotations"].append({"key": spec["action_key"], "role": "action", "representation": "eef",
                                     "value_mode": source, "component": "other", "arm": "unknown"})
    if declared and source not in {"unknown", declared}:
        with pytest.raises(ValueError, match="COMPOSITION_SOURCE_MISMATCH"):
            validate_composition_profile_input(row, spec)
    else:
        assert validate_composition_profile_input(row, spec) == normalize_composition(spec)
    row["field_annotations"][-1]["representation"] = "unknown"
    with pytest.raises(ValueError, match="COMPOSITION_SOURCE_UNCONFIRMED"):
        validate_composition_profile_input(row, spec)


def test_blank_legacy_encoding_declarations_do_not_conflict_with_asset_metadata():
    row = profile(); row.update(action_mode="delta", rotation_representation="quaternion")
    row["processing"].update(action_mode="", rotation_representation="")
    validate_profile_composition(row)
    assert validate_composition_profile_input(row, row["processing"])["action_mode"] == ""


def test_auto_prompt_does_not_require_a_metadata_feature_but_aux_stays_required():
    spec = profile()["processing"]; spec["prompt_key"] = ""
    info = metadata(); info["features"].pop("task")
    assert verify_composition_metadata(spec, info)["prompt_key"] == ""
    spec["aux_keys"] = ["is_positive"]
    with pytest.raises(ValueError, match="is_positive"):
        verify_composition_metadata(spec, info)


@pytest.mark.parametrize("field,value", [("robot_type", "other"), ("arm_mode", "bimanual"),
    ("action_representation", "joint"), ("action_dim", 7), ("state_dim", 10),
    ("rotation_representation", "quaternion"), ("action_mode", "delta"), ("cameras", [])])
def test_asset_description_cannot_disagree_with_its_processing(field, value):
    row = profile(); row[field] = value
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"): validate_profile_composition(row)


def test_actual_segmented_metadata_proves_state_dim_and_custom_action_key():
    row, info = profile(), metadata()
    assert verify_registered_dataset_profile(row, info) == row
    info["features"]["command"] = info["features"].pop("actions")
    row["processing"]["action_key"] = "command"
    assert verify_registered_dataset_profile(row, info) == row
    assert verify_composition_metadata(row["processing"], info, action_horizon=16)["action_key"] == "command"
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
        verify_composition_metadata(row["processing"], info, action_horizon=32)


@pytest.mark.parametrize("mutation", ["state", "history", "camera", "prompt", "action"])
def test_real_metadata_must_match_composition_before_training(mutation):
    row, info = profile(), metadata()
    if mutation == "state": info["features"]["observation.joints"]["shape"] = [7]
    if mutation == "history": row["processing"]["state_history"] = 2
    if mutation == "camera": info["features"]["observation.images.wrist"]["shape"] = [224, 224, 3]
    if mutation == "prompt": info["features"].pop("task")
    if mutation == "action": info["features"]["actions"]["shape"] = [16, 9]
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
        verify_composition_metadata(row["processing"], info)


@pytest.fixture
def composed_module(processor_module, monkeypatch):
    path = Path(processor_module.__file__)
    path.write_text(SOURCE + '''
@dataclasses.dataclass
class ComposableDataConfig(DataConfigFactory):
    COMPOSITION_SCHEMA_VERSION = 1
    composition: dict | None = None
    training_stage: str = 'sft'
    acp_dropout: float = .1
    def processing_contract(self, model):
        import json, hashlib
        from pathlib import Path
        content=(Path(self.repo_id)/'meta/info.json').read_bytes()
        shape=json.loads(content)['features'][self.composition['action_key']]['shape']
        return dict(training_stage=self.training_stage,model_type=type(model).__name__,pi05=model.pi05,
            action_dim=model.action_dim,action_horizon=model.action_horizon,
            image_keys=['base_0_rgb','left_wrist_0_rgb','right_wrist_0_rgb'],depth_keys=[],
            raw_state_dim=sum(x['width'] for x in self.composition['state_fields']),
            raw_action_dim=sum(x['width'] for x in self.composition['action_layout']),
            state_history=self.composition['state_history'],normalization=self.composition['normalization'],
            action_loss_mask=[1.]*sum(x['width'] for x in self.composition['action_layout'])+[0.]*(model.action_dim-sum(x['width'] for x in self.composition['action_layout'])),
            acp_dropout=float(self.acp_dropout),action_sequence_keys=[self.composition['action_key']] if len(shape)==1 else [],
            action_metadata_sha256=hashlib.sha256(content).hexdigest())
    def create(self): raise AssertionError('catalog must not execute create')
''')
    loader = importlib.util.spec_from_file_location("registered_composed_fixture", path)
    module = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, loader.name, module); loader.loader.exec_module(module)
    return module


def test_catalog_and_constructor_require_explicit_capability_and_preserve_asset_authority(composed_module):
    module = composed_module; root = Path(module.__file__).parent
    entry = next(x for x in catalog_data_processors(module, root) if x["name"] == "ComposableDataConfig")
    assert entry["composition_schema_version"] == 1 and entry["parameters"] == [] and entry["available"]
    choice = composition_choice(profile())
    factory = instantiate_data_processor(choice, module, root, dataset_path="/selected/dataset", norm_dir="/selected/norm",
                                         training_stage="offline_rl", stage_config={"acp_dropout": .3})
    assert factory.repo_id == "/selected/dataset" and factory.assets.assets_dir == "/selected/norm"
    assert factory.training_stage == "offline_rl" and factory.acp_dropout == .3
    assert validate_processor_instance(factory, choice) == choice
    changed = dataclasses.replace(factory, composition={**factory.composition, "coordinate_frame": "other"})
    with pytest.raises(ValueError, match="DATA_PROCESSOR_BINDING_MISMATCH"): validate_processor_instance(changed, choice)
    for extra in ({"repo_id": "/hidden"}, {"training_stage": "offline_rl"}, {"acp_dropout": .5}):
        with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
            normalize_processor({**choice, "parameters": {**choice["parameters"], **extra}})
    module.ComposableDataConfig.COMPOSITION_SCHEMA_VERSION = 2
    with pytest.raises(ValueError, match="DATA_PROCESSOR_UNSUPPORTED"):
        instantiate_data_processor(choice, module, root)


def test_norm_dimensions_and_quantiles_match_raw_concatenated_layout(tmp_path, composed_module):
    choice = composition_choice(profile())
    factory = instantiate_data_processor(choice, composed_module, Path(composed_module.__file__).parent,
                                         dataset_path=str(tmp_path), norm_dir=str(tmp_path))
    path = tmp_path / "norm_stats.json"
    stats = {key: {name: [0.] * width for name in ("mean", "std", "q01", "q99")}
             for key, width in (("state", 7), ("actions", 10))}
    path.write_text(json.dumps({"norm_stats": stats}))
    assert training_plan.norm_contract(factory, {"allowed_roots": [str(tmp_path)]})["norm_path"] == str(path)
    stats["state"] = {name: [0.] * 8 for name in ("mean", "std", "q01", "q99")}
    path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError, match="COMPOSITION_NORM_MISMATCH"):
        training_plan.norm_contract(factory, {"allowed_roots": [str(tmp_path)]})
    stats["state"] = {name: [0.] * 7 for name in ("mean", "std")}
    path.write_text(json.dumps({"norm_stats": stats}))
    with pytest.raises(ValueError, match="NORM_FORMAT"):
        training_plan.norm_contract(factory, {"allowed_roots": [str(tmp_path)]})


def test_recap_uses_explicit_aux_fields_without_legacy_mapping(composed_module):
    spec = profile()["processing"]; spec["aux_keys"] = ["is_positive"]
    factory = composed_module.ComposableDataConfig(composition=spec, training_stage="offline_rl")
    assert recap_runtime._factory_mapping(factory)["is_positive"] == "is_positive"
    assert "actions" in recap_runtime._factory_mapping(factory)
    config = SimpleNamespace(data=factory, model=SimpleNamespace())
    assert recap_runtime.apply_stage(config, {"template_id": "recap", "training_stage": "offline_rl", "stage_config": {}}) is config
    factory.composition = {**spec, "aux_keys": []}
    with pytest.raises(ValueError, match="RECAP_STAGE_RECIPE_MISMATCH"):
        recap_runtime.apply_stage(config, {"template_id": "recap", "training_stage": "offline_rl", "stage_config": {}})


def test_recipe_overrides_cannot_replace_asset_owned_composition():
    for path, value in (("data.composition", profile()["processing"]), ("data", {"composition": profile()["processing"]}),
                        ("data.datasets[0].composition.state_history", 2)):
        with pytest.raises(ValueError, match="RECIPE_(ASSET_FIELDS_MANAGED|OVERRIDES_INVALID)"):
            validate_overrides({path: value})


def test_composed_plan_records_verified_shape_stage_and_helper_source(training, composed_module, core, monkeypatch):
    _, spec, _, _ = training
    path = Path(spec["inputs"]["dataset_0"]["path"])
    info = {**metadata(), "total_frames": 20, "total_episodes": 2}
    (path / "meta/info.json").write_text(json.dumps(info))
    source = Path(spec["code_dir"]) / "src/openpi/training/composed_processing.py"
    source.parent.mkdir(parents=True, exist_ok=True); source.write_text("# frozen processing source\n")
    factory = instantiate_data_processor(composition_choice(profile()), composed_module, spec["code_dir"],
                                         dataset_path=str(path), norm_dir=str(path))
    model = type("Pi0Config", (), {"action_dim": 32, "action_horizon": 16, "pi05": True})()
    # Exercise the actual lightweight training helper through the adapter. Model
    # classes are identity-only doubles; no JAX, tokenizer or model weights load.
    from types import ModuleType
    models = ModuleType("openpi.models")
    models.pi0_config = SimpleNamespace(Pi0Config=type(model))
    models.pi0_gripper = SimpleNamespace(Pi0GripperConfig=type("Pi0GripperConfig", (), {}))
    models.pi0_value = SimpleNamespace(Pi0ValueConfig=type("Pi0ValueConfig", (), {}))
    monkeypatch.setitem(sys.modules, "openpi.models", models)
    monkeypatch.setattr(composed_module.ComposableDataConfig, "processing_contract", core.composition_runtime_contract)
    result = training_plan.dataset_contract(factory, spec, profile(), model_config=model)
    assert result["composition"]["digest"] == composition_digest(profile()["processing"])
    assert result["composition"]["source"]["path"] == "src/openpi/training/composed_processing.py"
    assert result["actions_shape"] == [16, 10]
    # New runs own the selected plan; a source asset need not save processing.
    source_profile = profile(); source_profile.pop("processing")
    source_profile["state_dim"] = 123  # Old aggregate summaries do not fix the run's selected state layout.
    selected = composition_choice(profile())
    explicit = training_plan.dataset_contract(factory, spec, source_profile, model_config=model,
                                              binding={"processor": selected})
    assert explicit["composition"]["digest"] == composition_digest(selected["parameters"]["composition"])
    selected = deepcopy(selected)
    selected["parameters"]["composition"]["state_fields"].reverse()
    selected["parameters"]["composition"]["normalization"] = "zscore"
    selected_factory = dataclasses.replace(factory, composition=selected["parameters"]["composition"])
    independent = training_plan.dataset_contract(selected_factory, spec, profile(), model_config=model,
                                                 binding={"processor": selected})
    assert independent["composition"]["spec"] == selected["parameters"]["composition"]
    with pytest.raises(ValueError, match="COMPOSITION_BINDING_MISMATCH"):
        training_plan.dataset_contract(factory, spec, profile(), model_config=model,
                                       binding={"processor": selected})
    changed = profile(); changed["processing"]["coordinate_frame"] = "another"
    with pytest.raises(ValueError, match="COMPOSITION_BINDING_MISMATCH"):
        training_plan.dataset_contract(factory, spec, changed, model_config=model)
    with pytest.raises(ValueError, match="COMPOSITION_MODEL_UNSUPPORTED"):
        training_plan.dataset_contract(factory, spec, profile(), model_config=type("Pi0MemConfig", (), {})())
    model.action_horizon = 32
    # A valid fps disagreement remains advisory; a real tensor horizon mismatch
    # still reaches and fails the composition contract after that comparison.
    fps_profile = profile();fps_profile['fps'] = 60
    info['fps'] = 30
    (path / 'meta/info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match="COMPOSITION_INVALID"):
        training_plan.dataset_contract(factory, spec, fps_profile, model_config=model)
    model.action_horizon = 16
    fps_checked = training_plan.dataset_contract(factory, spec, fps_profile, model_config=model)
    assert fps_checked['warnings'][0]['code'] == 'DATASET_FPS_MISMATCH'
    assert fps_checked['composition']['action_horizon'] == 16
    checks = []
    info['fps'] = 20
    info['features']['observation.images.wrist']['shape'] = [1, 3, 224, 224]
    (path / 'meta/info.json').write_text(json.dumps(info))
    with pytest.raises(ValueError, match='COMPOSITION_INVALID.*CHW/HWC'):
        training_plan.dataset_contract(factory, spec, fps_profile, model_config=model, checks=checks)
    assert len(checks) == 1
    assert checks[0]['name'] == 'dataset_fps' and checks[0]['ok'] is True
    assert checks[0]['severity'] == 'warning' and checks[0]['code'] == 'DATASET_FPS_MISMATCH'
    assert str(path) in checks[0]['message'] and '60 Hz' in checks[0]['message'] and '20 Hz' in checks[0]['message']
    info['features']['observation.images.wrist']['shape'] = [3, 224, 224]
    (path / 'meta/info.json').write_text(json.dumps(info))
    model.pi05 = False; model.action_dim = 6
    with pytest.raises(ValueError, match="COMPOSITION_MODEL_UNSUPPORTED"):
        training_plan.dataset_contract(factory, spec, profile(), model_config=model)
    model.pi05 = True; model.action_dim = 32
    source.write_text("# changed processing implementation\n")
    changed_source = training_plan.dataset_contract(factory, spec, profile(), model_config=model)
    assert changed_source["composition"]["source"]["sha256"] != result["composition"]["source"]["sha256"]


@pytest.mark.parametrize("with_profile", [True, False])
@pytest.mark.parametrize("prompt_key", ["task", ""])
def test_run_owned_prompt_proves_derived_task_without_an_asset_plan(training, composed_module, with_profile, prompt_key):
    _, spec, _, _ = training
    path = Path(spec["inputs"]["dataset_0"]["path"])
    info = {**metadata(), "total_frames": 20, "total_episodes": 2}
    info["features"].pop("task")
    info["features"]["task_index"] = {"shape": [1], "dtype": "int64"}
    (path / "meta/info.json").write_text(json.dumps(info))
    tasks = path / "meta/tasks.jsonl"
    task_bytes = b'{"task_index":0,"task":"Pick up cup"}\n'
    tasks.write_bytes(task_bytes)
    source = Path(spec["code_dir"]) / "src/openpi/training/composed_processing.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# selected processing source\n")
    selected = composition_choice(profile())
    selected["parameters"]["composition"]["prompt_key"] = prompt_key
    factory = instantiate_data_processor(selected, composed_module, spec["code_dir"],
                                         dataset_path=str(path), norm_dir=str(path))
    model = type("Pi0Config", (), {"action_dim": 32, "action_horizon": 16, "pi05": True})()
    declared = profile() if with_profile else None
    if declared is not None:
        declared.pop("processing")
        declared["feature_keys"] = ["task_index"]
    checked = training_plan.dataset_contract(factory, spec, declared, model_config=model,
                                             binding={"processor": selected})
    assert checked["derived_features"] == ["task"]
    assert checked["task_metadata_sha256"] == hashlib.sha256(task_bytes).hexdigest()
    assert checked.get("registered_profile") == declared
    tasks.unlink()
    if prompt_key:
        with pytest.raises(ValueError, match="DATASET_PROFILE_UNVERIFIABLE"):
            training_plan.dataset_contract(factory, spec, declared, model_config=model,
                                           binding={"processor": selected})
    else:
        checked = training_plan.dataset_contract(factory, spec, declared, model_config=model,
                                                 binding={"processor": selected})
        assert "derived_features" not in checked and "task_metadata_sha256" not in checked


def test_run_prompt_never_invents_task_without_a_real_task_index_mapping(tmp_path):
    spec = {"allowed_roots": [str(tmp_path)]}
    path = tmp_path / "data"; (path / "meta").mkdir(parents=True)
    tasks = path / "meta/tasks.jsonl"
    tasks.write_text('{"task_index":0,"task":"Pick up cup"}\n')
    selected = profile()["processing"]
    assert training_plan.registered_task_features(path, {"features": {}}, spec, None,
                                                 selected_composition=selected) == {}
    info = {"features": {"task_index": {"dtype": "int64", "shape": [1]}}}
    tasks.write_text('{"task_index":0,"task":"Pick up cup"}\n{"task_index":0,"task":"Duplicate"}\n')
    with pytest.raises(ValueError, match="DATASET_PROFILE_UNVERIFIABLE"):
        training_plan.registered_task_features(path, info, spec, None, selected_composition=selected)
    # A historical plan must not trigger task derivation for a different prompt.
    different = {**selected, "prompt_key": "instruction"}
    assert training_plan.registered_task_features(path, info, spec, profile(),
                                                 selected_composition=different) == {}


@pytest.mark.parametrize("claimed_key", ["task", "task/joints"])
def test_automatic_prompt_task_evidence_is_optional_but_available_mapping_must_be_valid(tmp_path, claimed_key):
    spec = {"allowed_roots": [str(tmp_path)]}
    path = tmp_path / "data"; (path / "meta").mkdir(parents=True)
    selected = profile()["processing"]; selected["prompt_key"] = ""
    info = {"features": {"task_index": {"dtype": "int64", "shape": [1]}}}
    assert training_plan.registered_task_features(path, info, spec, None, selected_composition=selected) == {}
    tasks = path / "meta/tasks.jsonl"; tasks.write_text('{"task_index":0,"task":"Pick up cup"}\n')
    result = training_plan.registered_task_features(path, info, spec, None, selected_composition=selected)
    assert result["derived_features"] == ["task"]
    assert result["task_metadata_sha256"] == hashlib.sha256(tasks.read_bytes()).hexdigest()
    tasks.write_text('{"task_index":0,"task":""}\n')
    with pytest.raises(ValueError, match="DATASET_PROFILE_UNVERIFIABLE"):
        training_plan.registered_task_features(path, info, spec, None, selected_composition=selected)
    # Reserved source fields are never selected as automatic language inputs.
    selected["action_key"] = claimed_key
    assert training_plan.registered_task_features(path, info, spec, None, selected_composition=selected) == {}


def test_mixture_uses_independent_bound_stats_and_refuses_incompatible_final_state_width(training, composed_module):
    from vla_adapters.openpi_entry import resolve_config
    base, spec, module, loaders = training
    module.DataConfigFactory = composed_module.DataConfigFactory
    module.ComposableDataConfig = composed_module.ComposableDataConfig
    spec["dataset_bindings"] = [spec["dataset_bindings"][0]]
    spec["dataset_bindings"][0].pop("recipe_index")
    spec["dataset_bindings"][0]["processor"] = composition_choice(profile())
    spec["inputs"] = {key: value for key, value in spec["inputs"].items() if key in {"dataset_0", "norm_0", "model"}}
    resolved = resolve_config(base, spec, module, loaders)
    assert resolved.data.use_merged_norm_stats is False and base.data.use_merged_norm_stats is True
    composed = {"pi05": True, "state_history": 2, "raw_state_dim": 56, "action_dim": 32}
    with pytest.raises(ValueError, match="COMPOSITION_STATE_PADDING_INVALID"):
        training_plan.check_composed_mixture(resolved.data, [{"composition": composed}])
    resolved.data.state_pad_dim = 128
    training_plan.check_composed_mixture(resolved.data, [{"composition": composed}])
    composed["pi05"] = False
    with pytest.raises(ValueError, match="COMPOSITION_STATE_PADDING_INVALID"):
        training_plan.check_composed_mixture(resolved.data, [{"composition": composed}])
    resolved.data.state_pad_dim = 32
    training_plan.check_composed_mixture(resolved.data, [{"composition": composed}])
    resolved.data.use_merged_norm_stats = True
    with pytest.raises(ValueError, match="COMPOSITION_MIXTURE_UNSUPPORTED"):
        training_plan.check_composed_mixture(resolved.data, [{"composition": composed}])
