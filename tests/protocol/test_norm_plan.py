import copy
import hashlib
import json
from pathlib import Path

import pytest

from vla_protocol import canonical_digest
from vla_protocol.norm_plan import validate_norm_cache, validate_norm_plan


def norm_plan(*, composed=False, per_frame=False):
    spec = json.loads((Path(__file__).resolve().parents[1] / "fixtures/composed-processing-v1.json").read_text())["valid"][0]
    spec["cameras"] = []
    states = [row["key"] for row in spec["state_fields"]] if composed else ["observation.state"]
    action = spec["action_key"] if composed else "action"
    temporal = ["episode_index", "frame_index", "timestamp"] if per_frame else []
    columns = [*states, action, *temporal]
    width = sum(row["width"] for row in spec["action_layout"]) if composed else 14
    row = {"dataset_asset_id": "data", **({"processor": {"name": "ComposableDataConfig",
        "parameters": {"composition": spec}}} if composed else {"recipe_index": 0})}
    inputs = {"dataset_0": {"id": "data", "kind": "DATASET", "path": "/machine/data"}}
    request = {"recipe": "recipe", "parameters": {"batch_size": 32, "num_workers": 0}, "norm_datasets": [row]}
    dataset = {**row, "path": "/machine/data", "input_role": "dataset_0", "frames": 32,
        "metadata_sha256": "a" * 64, "manifest_sha256": "b" * 64,
        "sample_shapes": {"state": [7], "actions": [16, width]},
        "actions_shape": [width] if per_frame else [16, width],
        "numeric_io": {"schema_version": 1, "reader": "pyarrow_parquet_projection", "columns": columns,
            "state_columns": states, "action_column": action, "temporal_columns": temporal,
            "action_sequence": "episode_clamp" if per_frame else "packed", "image_columns": [], "video_decode": False,
            "source": {"path": "src/openpi/training/numeric_stats.py", "sha256": "c" * 64}}}
    if composed:
        dataset.update(factory="ComposableDataConfig", processor_identity={"name": "ComposableDataConfig",
            "module": "openpi.training.config", "source": {"path": "src/openpi/training/config.py", "sha256": "d" * 64}},
            composition={"schema_version": 1, "spec": copy.deepcopy(spec), "digest": canonical_digest(spec),
                "training_stage": "sft", "model_type": "Pi0Config", "pi05": True, "action_horizon": 16, "action_dim": 32,
                "raw_state_dim": 7, "raw_action_dim": width, "state_history": spec["state_history"],
                "normalization": spec["normalization"], "action_metadata_sha256": "a" * 64,
                "action_sequence_keys": [action] if per_frame else [], "action_loss_mask": [1.] * width + [0.] * (32 - width),
                "image_keys": [], "depth_keys": [],
                "source": {"path": "src/openpi/training/composed_processing.py", "sha256": "e" * 64}})
    source = "CONFIG = {}\n"
    plan = {"template_id": "openpi-norm", "mode": "per_dataset_batch", **request,
        "gpu_count": 0, "recipe_definition_digest": None, "input_authority": "registered_assets",
        "requested_inputs": {"dataset_0": "/machine/data"}, "inputs": {"dataset_0": "/machine/data"},
        "entrypoint": "/machine/code/scripts/compute_norm_stats.py", "entrypoint_sha256": "f" * 64, "adapter_sha256": "0" * 64,
        "generated_config": {"schema_version": 1, "language": "python", "filename": "training_config.py",
            "source": source, "sha256": hashlib.sha256(source.encode()).hexdigest()}, "datasets": [dataset],
        "training_cache": {"mode": "numeric_projection", "persistent_cache": False, "paths": {},
            "datasets": [{"path": "/machine/data", "columns": list(columns)}]}}
    return plan, request, inputs


@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("per_frame", [False, True])
def test_numeric_source_plan_proves_packed_or_episode_bound_sampling_without_visuals(composed, per_frame):
    plan, request, inputs = norm_plan(composed=composed, per_frame=per_frame)
    assert validate_norm_plan(plan, request, inputs) is plan
    assert validate_norm_cache(plan["training_cache"], plan["datasets"], ["/machine"]) is plan["training_cache"]


def test_temporal_projection_deduplicates_an_explicit_numeric_state_column():
    plan, request, inputs = norm_plan(per_frame=True)
    projection = plan["datasets"][0]["numeric_io"]
    projection["state_columns"] = ["timestamp"]
    projection["columns"] = ["timestamp", "action", "episode_index", "frame_index"]
    plan["training_cache"]["datasets"][0]["columns"] = list(projection["columns"])
    assert validate_norm_plan(plan, request, inputs) is plan


def test_numeric_plan_accepts_explicit_composite_representation_and_freezes_it():
    plan, request, inputs = norm_plan(composed=True)
    spec = request["norm_datasets"][0]["processor"]["parameters"]["composition"]
    spec.update(state_representation="composite", action_representation="composite")
    spec["action_layout"] = [{"arm": "shared", "kind": "shared", "width": 10, "normalize": True}]
    dataset = plan["datasets"][0]
    dataset["composition"].update(spec=copy.deepcopy(spec), digest=canonical_digest(spec))
    assert validate_norm_plan(plan, request, inputs) is plan
    # A composite label cannot claim or invent a per-arm decomposition.
    spec["action_layout"] = [{"arm": "left", "kind": "joint", "width": 10, "normalize": True}]
    dataset["composition"].update(spec=copy.deepcopy(spec), digest=canonical_digest(spec))
    with pytest.raises(ValueError, match="one complete shared vector"):
        validate_norm_plan(plan, request, inputs)


@pytest.mark.parametrize("spelling", ["trailing_slash", "dot", "parent", "symlink"])
def test_agent_plan_keeps_exact_registered_path_spelling_and_canonical_source(tmp_path, spelling):
    source = tmp_path / "dataset"; source.mkdir()
    (tmp_path / "component").mkdir()
    alias = tmp_path / "dataset-alias"; alias.symlink_to(source, target_is_directory=True)
    registered = {"trailing_slash": str(source) + "/", "dot": str(tmp_path) + "/./dataset",
                  "parent": str(tmp_path) + "/component/../dataset", "symlink": str(alias)}[spelling]
    canonical = str(Path(registered).resolve(strict=True))
    plan, request, inputs = norm_plan()
    inputs["dataset_0"]["path"] = canonical
    request["inputs"] = copy.deepcopy(inputs)
    request["requested_inputs"] = {"dataset_0": registered}
    plan["requested_inputs"] = {"dataset_0": registered}
    plan["inputs"]["dataset_0"] = canonical
    plan["datasets"][0]["path"] = canonical
    plan["training_cache"]["datasets"][0]["path"] = canonical
    assert validate_norm_plan(plan, request, inputs) is plan
    assert plan["requested_inputs"]["dataset_0"] == registered
    assert plan["inputs"]["dataset_0"] == str(source.resolve())
    # Even an internally consistent plan may never swap the selected source.
    other = tmp_path / "unselected"; other.mkdir()
    changed = copy.deepcopy(plan)
    changed["inputs"]["dataset_0"] = str(other.resolve())
    changed["datasets"][0]["path"] = str(other.resolve())
    changed["training_cache"]["datasets"][0]["path"] = str(other.resolve())
    with pytest.raises(ValueError, match="canonical selected assets"):
        validate_norm_plan(changed, request, inputs)
    changed = copy.deepcopy(plan)
    changed["requested_inputs"]["dataset_0"] = canonical
    with pytest.raises(ValueError, match="requested recipe, assets or parameters"):
        validate_norm_plan(changed, request, inputs)


def test_controller_plan_without_private_paths_still_freezes_original_asset_path_exactly():
    plan, request, inputs = norm_plan()
    inputs["dataset_0"]["path"] += "/"
    plan["requested_inputs"]["dataset_0"] += "/"
    assert "requested_inputs" not in request
    assert validate_norm_plan(plan, request, inputs) is plan
    plan["requested_inputs"]["dataset_0"] = plan["requested_inputs"]["dataset_0"].rstrip("/")
    with pytest.raises(ValueError, match="requested recipe, assets or parameters"):
        validate_norm_plan(plan, request, inputs)


@pytest.mark.parametrize("requested", [None, {}, {"wrong_role": "/machine/data"},
    {"dataset_0": "relative/data"}, {"dataset_0": "/machine/data", "extra": "/machine/extra"}])
def test_private_original_paths_cannot_change_asset_role_set_or_be_relative(requested):
    plan, request, inputs = norm_plan()
    request["requested_inputs"] = requested
    plan["requested_inputs"] = requested
    with pytest.raises(ValueError, match="selected asset roles"):
        validate_norm_plan(plan, request, inputs)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("reader", "lerobot"), ("video_decode", True),
    ("video_decode", 0), ("image_columns", ["observation.images.wrist"]), ("columns", ["action"]),
    ("columns", ["observation.state", "action", "video"]), ("state_columns", []),
    ("columns", ["observation.state", "action", "action"]), ("action_column", []),
    ("source", {"path": "../numeric_stats.py", "sha256": "a" * 64}),
    ("source", {"path": [], "sha256": "a" * 64}), ("source", {"path": "src/openpi/training/numeric_stats.py", "sha256": "unknown"}),
    ("action_sequence", "episode_clamp"), ("temporal_columns", ["timestamp"])])
def test_statistics_never_accept_visual_loading_unknown_columns_or_unproven_numeric_reader(field, value):
    plan, request, inputs = norm_plan()
    plan["datasets"][0]["numeric_io"][field] = value
    with pytest.raises(ValueError):
        validate_norm_plan(plan, request, inputs)


def test_statistics_cannot_use_old_evidence_without_numeric_source_projection():
    plan, request, inputs = norm_plan()
    plan["datasets"][0].pop("numeric_io")
    with pytest.raises(ValueError, match="numeric-only source projection"):
        validate_norm_plan(plan, request, inputs)


@pytest.mark.parametrize("change", ["visual_spec", "image_keys", "depth_keys", "source_fields", "raw_shape", "temporal_sampling"])
def test_composed_statistics_bind_actual_numeric_columns_and_exclude_model_visual_inputs(change):
    plan, request, inputs = norm_plan(composed=True)
    dataset = plan["datasets"][0]
    if change == "visual_spec":
        spec = request["norm_datasets"][0]["processor"]["parameters"]["composition"]
        spec["cameras"] = [{"key": "observation.images.wrist"}]
        dataset["composition"]["spec"] = copy.deepcopy(spec)
        dataset["composition"]["digest"] = canonical_digest(spec)
    elif change in {"image_keys", "depth_keys"}:
        dataset["composition"][change] = ["base_0_rgb"]
    elif change == "source_fields":
        dataset["numeric_io"]["state_columns"][0] = "unselected.state"
        dataset["numeric_io"]["columns"][0] = "unselected.state"
    elif change == "raw_shape":
        dataset["actions_shape"] = [10]
    else:
        dataset["numeric_io"]["action_sequence"] = "episode_clamp"
        dataset["numeric_io"]["temporal_columns"] = ["episode_index", "frame_index", "timestamp"]
        dataset["numeric_io"]["columns"].extend(dataset["numeric_io"]["temporal_columns"])
    with pytest.raises(ValueError):
        validate_norm_plan(plan, request, inputs)


@pytest.mark.parametrize("change", ["old_builder", "persistent_cache", "missing_source", "different_source", "extra_column",
    "unknown_env", "outside_root", "relative_path", "traversal", "env_type"])
def test_numeric_environment_and_each_source_projection_stay_within_managed_roots(change):
    plan, _, _ = norm_plan()
    cache = plan["training_cache"]
    if change == "old_builder":
        cache["builders"] = []
    elif change == "persistent_cache":
        cache["persistent_cache"] = True
    elif change == "missing_source":
        cache["datasets"].clear()
    elif change == "different_source":
        cache["datasets"][0]["path"] = "/machine/other-data"
    elif change == "extra_column":
        cache["datasets"][0]["columns"].append("video")
    elif change == "unknown_env":
        cache["paths"]["UNRELATED"] = "/machine/cache"
    elif change == "outside_root":
        cache["paths"]["TMPDIR"] = "/outside/cache"
    elif change == "relative_path":
        cache["paths"]["TMPDIR"] = "relative/cache"
    elif change == "traversal":
        cache["paths"]["TMPDIR"] = "/machine/../outside/cache"
    else:
        cache["paths"]["TMPDIR"] = []
    with pytest.raises(ValueError):
        validate_norm_cache(cache, plan["datasets"], ["/machine"])
