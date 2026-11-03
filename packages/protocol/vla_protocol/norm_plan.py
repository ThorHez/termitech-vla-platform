"""Shared evidence requirements for a CPU normalization task."""

from pathlib import PurePosixPath
import re

from . import canonical_digest
from .config_preview import validate_generated_config


NORM_CHECKS = {"recipe_configuration", "data_contract", "dataset_samples", "norm_entrypoint", "training_cache",
               "output_isolation", "output_storage", "gpu_capacity", "runtime", "local_paths", "code_version", "adapter"}
NORM_CACHE_PATH_KEYS = {"HF_DATASETS_CACHE", "HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
                        "TRANSFORMERS_CACHE", "TMPDIR", "TEMP", "TMP", "XDG_CACHE_HOME",
                        "JAX_COMPILATION_CACHE_DIR", "OPENPI_DATA_HOME"}


def validate_norm_plan(plan, request, inputs):
    paths = {role: value["path"] for role, value in inputs.items()}
    # Agent inputs have already been resolved inside its managed roots; keep
    # the original asset spelling separately, including aliases/trailing '/'.
    # Controller requests have no private requested_inputs and use Asset.path.
    requested_paths = request.get("requested_inputs", paths)
    if (not isinstance(requested_paths, dict) or set(requested_paths) != set(paths)
            or any(not isinstance(path, str) or not PurePosixPath(path).is_absolute()
                   for path in requested_paths.values())):
        raise ValueError("Statistics requested input paths differ from the selected asset roles")
    expected = {"template_id": "openpi-norm", "recipe": request["recipe"], "parameters": request["parameters"],
                "gpu_count": 0, "requested_inputs": requested_paths, "recipe_definition_digest": None,
                "input_authority": "registered_assets"}
    if not isinstance(plan, dict) or any(canonical_digest(plan.get(key)) != canonical_digest(value)
                                         for key, value in expected.items()):
        raise ValueError("Statistics plan differs from the requested recipe, assets or parameters")
    auto_config = request.get("auto_config")
    if canonical_digest(plan.get("auto_config")) != canonical_digest(auto_config):
        raise ValueError("Statistics plan differs from the requested automatic model configuration")
    rows = request.get("norm_datasets")
    if rows is None:
        roles = ["dataset"]
        if plan.get("norm_datasets") is not None:
            raise ValueError("Statistics plan contains unrequested batch datasets")
    else:
        if not isinstance(rows, list) or not 1 <= len(rows) <= 64 or plan.get("mode") != "per_dataset_batch":
            raise ValueError("Statistics plan has invalid independent batch inputs")
        if canonical_digest(plan.get("norm_datasets")) != canonical_digest(rows):
            raise ValueError("Statistics plan differs from the independent dataset processing choices")
        roles = [f"dataset_{index}" for index in range(len(rows))]
        ids = []
        for index, row in enumerate(rows):
            if (not isinstance(row, dict) or set(row) not in ({"dataset_asset_id", "recipe_index"}, {"dataset_asset_id", "processor"})
                    or not isinstance(row["dataset_asset_id"], str) or not row["dataset_asset_id"]
                    or inputs.get(roles[index], {}).get("id") != row["dataset_asset_id"]):
                raise ValueError("Statistics batch input asset identity differs from its request")
            ids.append(row["dataset_asset_id"])
        if len(set(ids)) != len(ids):
            raise ValueError("Statistics batch contains duplicate datasets")
    if auto_config is not None:
        if (not isinstance(auto_config, dict) or set(auto_config) != {"model_type", "finetune_mode", "action_horizon", "action_dim"}
                or not isinstance(auto_config["model_type"], str) or auto_config["model_type"] not in {"pi0", "pi05"}
                or auto_config["finetune_mode"] != "full"
                or type(auto_config["action_horizon"]) is not int or not 1 <= auto_config["action_horizon"] <= 1000
                or type(auto_config["action_dim"]) is not int or not 1 <= auto_config["action_dim"] <= 4096
                or rows is None or any("processor" not in row for row in rows)):
            raise ValueError("Statistics automatic configuration requires explicit processors and valid model choices")
        name = (f"auto_{auto_config['model_type']}_full_h{auto_config['action_horizon']}"
                f"_d{auto_config['action_dim']}_sft")
        if request.get("recipe") != name:
            raise ValueError("Statistics automatic configuration identity differs from its model choices")
    if set(paths) != set(roles) or not isinstance(plan.get("inputs"), dict) or set(plan["inputs"]) != set(roles):
        raise ValueError("Statistics plan requires the exact ordered dataset inputs")
    if "requested_inputs" in request and plan["inputs"] != paths:
        raise ValueError("Statistics effective dataset paths differ from the canonical selected assets")
    local = lambda value: isinstance(value, str) and PurePosixPath(value).is_absolute()
    digest = lambda value: isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None
    datasets = plan.get("datasets")
    if (not local(plan.get("entrypoint")) or not digest(plan.get("entrypoint_sha256"))
            or not digest(plan.get("adapter_sha256")) or any(not local(plan["inputs"][role]) for role in roles)
            or not isinstance(datasets, list) or len(datasets) != len(roles)):
        raise ValueError("Missing statistics source or entrypoint evidence")
    for index, (role, dataset) in enumerate(zip(roles, datasets)):
        if (not isinstance(dataset, dict) or dataset.get("path") != plan["inputs"][role]
                or type(dataset.get("frames")) is not int or dataset["frames"] < 1
                or not digest(dataset.get("metadata_sha256")) or not digest(dataset.get("manifest_sha256"))):
            raise ValueError("Missing dataset identity or coverage evidence")
        if rows is not None:
            row = rows[index]
            choice = "processor" if "processor" in row else "recipe_index"
            if (dataset.get("input_role") != role or dataset.get("dataset_asset_id") != row["dataset_asset_id"]
                    or canonical_digest(dataset.get(choice)) != canonical_digest(row[choice])):
                raise ValueError("Statistics plan has a different dataset processing order or choice")
            if choice == "processor":
                processor, identity = row["processor"], dataset.get("processor_identity")
                source = identity.get("source") if isinstance(identity, dict) else None
                if (not isinstance(processor, dict) or dataset.get("factory") != processor.get("name")
                        or not isinstance(identity, dict) or identity.get("name") != processor.get("name")
                        or not isinstance(identity.get("module"), str) or not identity["module"]
                        or not isinstance(source, dict) or not isinstance(source.get("path"), str)
                        or PurePosixPath(source["path"]).is_absolute() or ".." in PurePosixPath(source["path"]).parts
                        or not digest(source.get("sha256"))):
                    raise ValueError("Missing actual statistics processor source identity")
                if processor.get("name") == "ComposableDataConfig":
                    _validate_composition(dataset, processor, auto_config)
                elif dataset.get("composition") is not None:
                    raise ValueError("Statistics plan contains an unrequested field composition")
        shapes = dataset.get("sample_shapes")
        if not isinstance(shapes, dict) or set(shapes) != {"state", "actions"} or any(
            not isinstance(shape, list) or not shape or any(type(size) is not int or size < 1 for size in shape)
            for shape in shapes.values()
        ):
            raise ValueError("Missing decoded state/action sample evidence")
        if auto_config is not None and canonical_digest(dataset.get("model_configuration")) != canonical_digest(
            {key: auto_config[key] for key in ("model_type", "action_horizon", "action_dim")}
        ):
            raise ValueError("Statistics dataset model configuration differs from the requested automatic choices")
        composition = dataset.get("composition")
        processor = rows[index].get("processor") if rows is not None else None
        if composition is not None and (not isinstance(processor, dict) or processor.get("name") != "ComposableDataConfig"):
            if not isinstance(composition, dict):
                raise ValueError("Statistics field composition must be an object")
            _validate_composition(dataset, {"parameters": {"composition": composition.get("spec")}}, auto_config)
        _validate_numeric_io(dataset, composition.get("spec") if isinstance(composition, dict) else None)
    if len(set(plan["inputs"].values())) != len(roles):
        raise ValueError("Statistics batch contains duplicate dataset paths")
    validate_norm_cache(plan.get("training_cache"), datasets)
    validate_generated_config(plan.get("generated_config"))
    return plan


def _validate_composition(dataset, processor, auto_config):
    """Bind the actual SFT data contract without requiring any training asset."""
    spec = processor.get("parameters", {}).get("composition")
    evidence = dataset.get("composition")
    source = evidence.get("source") if isinstance(evidence, dict) else None
    if (not isinstance(spec, dict) or spec.get("cameras") != [] or not isinstance(evidence, dict)
            or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1
            or canonical_digest(evidence.get("spec")) != canonical_digest(spec)
            or evidence.get("digest") != canonical_digest(spec) or evidence.get("training_stage") != "sft"
            or not isinstance(evidence.get("model_type"), str) or evidence["model_type"] not in {"Pi0Config", "Pi0GripperConfig"}
            or not isinstance(source, dict)
            or not isinstance(source.get("path"), str)
            or source["path"] not in {"src/openpi/training/composed_processing.py", "openpi/training/composed_processing.py"}
            or not isinstance(source.get("sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", source["sha256"]) is None):
        raise ValueError("Missing actual statistics field composition or source identity")
    try:
        representations = {"eef", "joint", "composite"}
        if (not isinstance(spec.get("state_representation"), str) or spec["state_representation"] not in representations
                or not isinstance(spec.get("action_representation"), str) or spec["action_representation"] not in representations):
            raise ValueError("Statistics field composition representation is unsupported")
        if spec["action_representation"] == "composite" and (len(spec["action_layout"]) != 1
                or spec["action_layout"][0].get("arm") != "shared" or spec["action_layout"][0].get("kind") != "shared"):
            raise ValueError("Statistics composite actions require one complete shared vector")
        raw_state = sum(field["width"] for field in spec["state_fields"])
        raw_action = sum(field["width"] for field in spec["action_layout"])
        expected = {"raw_state_dim": raw_state, "raw_action_dim": raw_action,
                    "state_history": spec["state_history"], "normalization": spec["normalization"]}
        horizon, dimension = evidence.get("action_horizon"), evidence.get("action_dim")
        shape = dataset.get("actions_shape")
        if (type(horizon) is not int or horizon < 1 or type(dimension) is not int or not raw_action <= dimension <= 4096
                or type(evidence.get("pi05")) is not bool
                or evidence.get("action_metadata_sha256") != dataset["metadata_sha256"]
                or shape not in ([raw_action], [horizon, raw_action])
                or any(canonical_digest(evidence.get(key)) != canonical_digest(value) for key, value in expected.items())
                or evidence.get("action_sequence_keys") != ([spec["action_key"]] if shape == [raw_action] else [])
                or evidence.get("action_loss_mask") != [1.] * raw_action + [0.] * (dimension - raw_action)):
            raise ValueError("Statistics field composition numeric contract differs from its request")
        if auto_config is not None and (evidence["model_type"] != "Pi0Config"
                or evidence["pi05"] != (auto_config["model_type"] == "pi05")
                or horizon != auto_config["action_horizon"] or dimension != auto_config["action_dim"]):
            raise ValueError("Statistics field composition model differs from automatic choices")
        if evidence.get("image_keys") != [] or evidence.get("depth_keys") != []:
            raise ValueError("Statistics field composition must not include any visual inputs")
    except (KeyError, TypeError) as exc:
        raise ValueError("Missing statistics field composition numeric contract") from exc


def _validate_numeric_io(dataset, spec):
    evidence = dataset.get("numeric_io")
    if (not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int
            or evidence["schema_version"] != 1 or evidence.get("reader") != "pyarrow_parquet_projection"
            or evidence.get("image_columns") != [] or evidence.get("video_decode") is not False):
        raise ValueError("Statistics need numeric-only source projection without image or video decoding")

    def key(value):
        return isinstance(value, str) and 1 <= len(value) <= 256 and not any(ord(char) < 32 for char in value)

    columns, states = evidence.get("columns"), evidence.get("state_columns")
    action, temporal = evidence.get("action_column"), evidence.get("temporal_columns")
    if (not isinstance(columns, list) or not 1 <= len(columns) <= 36 or not all(key(item) for item in columns)
            or len(set(columns)) != len(columns) or not isinstance(states, list) or not 1 <= len(states) <= 32
            or not all(key(item) for item in states) or len(set(states)) != len(states) or not key(action)
            or action in states or not isinstance(temporal, list)
            or temporal not in ([], ["episode_index", "frame_index", "timestamp"])
            or columns != list(dict.fromkeys([*states, action, *temporal]))):
        raise ValueError("Statistics numeric projection must contain only ordered state, action and temporal columns")
    source = evidence.get("source")
    if (not isinstance(source, dict) or not isinstance(source.get("path"), str) or source["path"] not in {
            "src/openpi/training/numeric_stats.py", "openpi/training/numeric_stats.py"}
            or not isinstance(source.get("sha256"), str) or re.fullmatch(r"[a-f0-9]{64}", source["sha256"]) is None):
        raise ValueError("Statistics numeric reader needs traceable source identity")
    shape = dataset.get("actions_shape")
    if (not isinstance(shape, list) or len(shape) not in {1, 2}
            or any(type(size) is not int or size < 1 for size in shape)
            or (len(shape) == 2 and (evidence.get("action_sequence") != "packed" or temporal != []))
            or (len(shape) == 1 and (evidence.get("action_sequence") != "episode_clamp"
                                    or temporal != ["episode_index", "frame_index", "timestamp"]))):
        raise ValueError("Statistics numeric projection must prove its packed or episode-bound action sampling")
    if spec is not None:
        if (not isinstance(spec, dict) or not isinstance(spec.get("state_fields"), list)
                or any(not isinstance(row, dict) or not isinstance(row.get("key"), str) for row in spec["state_fields"])
                or states != [row["key"] for row in spec["state_fields"]] or action != spec.get("action_key")):
            raise ValueError("Statistics numeric projection differs from its selected state/action fields")


def validate_norm_cache(cache, datasets, roots=None):
    """Validate the numeric reader's environment without requiring HF builders."""
    def local(value):
        if not isinstance(value, str) or not 1 <= len(value) <= 4096:
            return False
        path = PurePosixPath(value)
        if not path.is_absolute() or str(path) != value or ".." in path.parts:
            return False
        return roots is None or any(path.is_relative_to(PurePosixPath(root)) for root in roots)

    if (roots is not None and (not isinstance(roots, list) or not roots
            or any(not isinstance(root, str) or not PurePosixPath(root).is_absolute() or ".." in PurePosixPath(root).parts
                   for root in roots))):
        raise ValueError("Statistics need managed roots for their numeric source and environment paths")
    if (not isinstance(cache, dict) or set(cache) != {"mode", "persistent_cache", "paths", "datasets"}
            or cache["mode"] != "numeric_projection" or cache["persistent_cache"] is not False
            or not isinstance(datasets, list) or not 1 <= len(datasets) <= 64):
        raise ValueError("Statistics must use numeric projection without a persistent dataset cache")
    paths, projections = cache["paths"], cache["datasets"]
    if (not isinstance(paths, dict) or not set(paths) <= NORM_CACHE_PATH_KEYS
            or any(not local(value) for value in paths.values())
            or not isinstance(projections, list) or len(projections) != len(datasets)):
        raise ValueError("Statistics numeric environment or ordered dataset evidence is invalid")
    for dataset, projection in zip(datasets, projections):
        if (not isinstance(dataset, dict) or not isinstance(projection, dict) or set(projection) != {"path", "columns"}
                or not local(projection["path"]) or projection["path"] != dataset.get("path")
                or not isinstance(dataset.get("numeric_io"), dict)
                or not isinstance(projection["columns"], list) or not projection["columns"]
                or canonical_digest(projection["columns"]) != canonical_digest(dataset["numeric_io"].get("columns"))):
            raise ValueError("Statistics cache evidence differs from each ordered numeric source projection")
    return cache
