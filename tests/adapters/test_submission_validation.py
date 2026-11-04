"""Training submission reuses source proof without claiming decoded transforms."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from vla_adapters import dataset_timestamps, dataset_validation_cache as cache
from vla_adapters import submission_validation as submission
from vla_adapters.sample_evidence import validate_sample_evidence


def source_summary(data):
    indices = sorted({0, data["frames"] // 2, data["frames"] - 1})
    fields = {"observation.state": {"shape": [14], "dtype": "float32"},
              "action": {"shape": [14], "dtype": "float32"},
              "is_positive": {"shape": [], "dtype": "bool"},
              "task_index": {"shape": [], "dtype": "int64"},
              **{key: {"shape": [3, 224, 224], "dtype": "float32"} for key in
                 ("observation.images.top", "observation.images.left", "observation.images.right")}}
    return {"schema_version": 1, "status": "verified", "path": data["path"],
            "raw_samples": {"selection": "first_middle_last", "checked_samples": len(indices),
                            "sample_indices": indices,
                            "samples": [{"index": index, "fields": deepcopy(fields)} for index in indices]}}


@pytest.fixture
def dataset():
    composition = {"state_fields": [{"key": "observation.state", "width": 14}],
                   "action_key": "action", "aux_keys": ["is_positive"],
                   "cameras": [{"key": key, "slot": slot, "kind": "rgb", "layout": "HWC"}
                               for key, slot in zip(
                                   ("observation.images.top", "observation.images.left", "observation.images.right"),
                                   ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))]}
    return {"path": "/registered/dataset", "frames": 5, "episodes": 1, "effective_fps": 20,
            "metadata_sha256": "a" * 64, "data_manifest_digest": "b" * 64,
            "composition": {"spec": composition, "action_horizon": 50, "action_sequence_keys": ["action"]},
            "processor_identity": {"name": "ComposableDataConfig", "source_sha256": "c" * 64},
            "timestamp_contract": {"schema_version": 1, "status": "verified", "effective_fps": 20,
                                   "checked_rows": 5, "checked_pairs": 4, "checked_episodes": 1,
                                   "metadata_sha256": "a" * 64, "data_manifest_digest": "b" * 64}}


def bound_dataset(data):
    result = deepcopy(data)
    result["submission_validation"] = submission.submission_evidence(result, source_summary(result))
    return result


@pytest.mark.parametrize("frames", [1, 2, 3, 32])
def test_submission_accepts_complete_registered_first_middle_last_shape_evidence(dataset, frames):
    dataset["frames"] = dataset["timestamp_contract"]["checked_rows"] = frames
    dataset["timestamp_contract"]["checked_pairs"] = frames - 1
    source = source_summary(dataset)
    evidence = submission.submission_evidence(dataset, source)
    dataset["submission_validation"] = evidence
    submission.validate_submission_evidence(dataset)
    validate_sample_evidence({"data_validation_mode": submission.MODE, "datasets": [dataset]})
    assert evidence["scope"] == "registered_source_and_configuration"
    assert evidence["source_evidence"] == source
    assert "sample_probe" not in dataset
    assert "output" not in json.dumps(evidence) and "model_views" not in json.dumps(evidence)


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("schema_version", 2), ("status", "not_applicable"),
    ("scope", "decoded_samples"), ("path", "/another/dataset"),
    ("metadata_sha256", "0" * 64), ("data_manifest_digest", "0" * 64),
    ("timestamp_contract_sha256", "0" * 64), ("source_evidence_sha256", "0" * 64),
    ("composition_sha256", "0" * 64), ("processor_identity_sha256", "0" * 64),
    ("unexpected", True),
])
def test_submission_rejects_modified_envelope_even_when_dataset_is_unchanged(dataset, field, value):
    data = bound_dataset(dataset)
    data["submission_validation"][field] = value
    with pytest.raises(ValueError):
        submission.validate_submission_evidence(data)


@pytest.mark.parametrize("field,value", [
    ("path", "/another/dataset"), ("metadata_sha256", "0" * 64),
    ("data_manifest_digest", "0" * 64),
    ("timestamp_contract", {"schema_version": 1, "status": "verified", "checked_rows": 99}),
    ("processor_identity", {"name": "another-processor"}),
    ("composition", {"spec": {"state_fields": [], "cameras": [], "action_key": "another-action", "aux_keys": []}}),
])
def test_submission_rejects_dataset_identity_or_configuration_changes(dataset, field, value):
    data = bound_dataset(dataset)
    data[field] = value
    with pytest.raises(ValueError):
        submission.validate_submission_evidence(data)


@pytest.mark.parametrize("change", [
    "schema-version-bool", "status", "path", "selection", "checked-count", "checked-count-bool",
    "indices-empty", "indices-missing-middle", "indices-bool", "rows-empty", "row-index", "row-index-bool",
    "fields-empty", "missing-state", "missing-action", "missing-camera", "missing-aux", "shape-negative",
    "shape-bool", "shape-string", "shape-rank", "dtype-string", "dtype-object", "field-key", "field-list",
])
def test_submission_rejects_incomplete_or_invalid_registered_source_summaries(dataset, change):
    source = source_summary(dataset)
    raw = source["raw_samples"]
    first = raw["samples"][0]
    if change == "schema-version-bool":
        source["schema_version"] = True
    elif change == "status":
        source["status"] = "failed"
    elif change == "path":
        source["path"] += "-other"
    elif change == "selection":
        raw["selection"] = "first_only"
    elif change.startswith("checked-count"):
        raw["checked_samples"] = True if change.endswith("bool") else 0
    elif change.startswith("indices-"):
        raw["sample_indices"] = ([] if change == "indices-empty" else
                                 [0, 4] if change == "indices-missing-middle" else [False, 2, 4])
    elif change == "rows-empty":
        raw["samples"] = []
    elif change.startswith("row-index"):
        first["index"] = False if change.endswith("bool") else 1
    elif change == "fields-empty":
        first["fields"] = {}
    elif change.startswith("missing-"):
        key = {"missing-state": "observation.state", "missing-action": "action",
               "missing-camera": "observation.images.right", "missing-aux": "is_positive"}[change]
        first["fields"].pop(key)
    elif change.startswith("shape-"):
        first["fields"]["action"]["shape"] = {
            "shape-negative": [-1], "shape-bool": [True], "shape-string": "14", "shape-rank": [1] * 6}[change]
    elif change.startswith("dtype-"):
        first["fields"]["action"]["dtype"] = "str" if change == "dtype-string" else "object"
    elif change == "field-key":
        first["fields"][""] = {"shape": [], "dtype": "int64"}
    else:
        first["fields"]["action"] = []
    with pytest.raises(ValueError):
        submission.submission_evidence(dataset, source)


def test_submission_detects_nested_source_edit_without_recomputed_digest(dataset):
    data = bound_dataset(dataset)
    data["submission_validation"]["source_evidence"]["raw_samples"]["samples"][0]["fields"]["action"]["shape"] = [15]
    with pytest.raises(ValueError):
        submission.validate_submission_evidence(data)


@pytest.mark.parametrize("field,value", [
    ("timestamp_contract", None), ("timestamp_contract", {}),
    ("schema_version", True), ("schema_version", 2), ("status", "failed"),
    ("checked_rows", 0), ("checked_rows", True), ("checked_rows", 6),
    ("checked_episodes", 0), ("checked_episodes", True), ("checked_episodes", 2),
    ("effective_fps", None), ("effective_fps", True), ("effective_fps", 30),
    ("metadata_sha256", "0" * 64), ("data_manifest_digest", "0" * 64),
])
def test_new_envelope_requires_successful_timestamp_proof_matching_current_dataset(dataset, field, value):
    if field == "timestamp_contract":
        dataset[field] = value
    else:
        dataset["timestamp_contract"][field] = value
    with pytest.raises(ValueError):
        submission.submission_evidence(dataset, source_summary(dataset))


@pytest.mark.parametrize("field", ["metadata_sha256", "data_manifest_digest"])
@pytest.mark.parametrize("invalid", [None, "", 123, True, "z" * 64, "a" * 63])
def test_new_envelope_rejects_non_hash_source_identity_even_when_timestamp_matches(dataset, field, invalid):
    dataset[field] = dataset["timestamp_contract"][field] = invalid
    with pytest.raises(ValueError):
        submission.submission_evidence(dataset, source_summary(dataset))


@pytest.mark.parametrize("composition", [["not-a-mapping"], {"spec": []}, {"spec": {}}, {"spec": {"state_fields": []}}])
def test_submission_malformed_composition_is_a_validation_error(dataset, composition):
    data = bound_dataset(dataset)
    data["composition"] = composition
    with pytest.raises(ValueError):
        submission.validate_submission_evidence(data)


@pytest.mark.parametrize("change", ["missing-mode", "unknown-mode", "mixed-old-probe", "missing-one-dataset", "legacy-only", "empty"])
def test_registered_mode_cannot_be_downgraded_or_mixed_with_historical_probe_evidence(dataset, change):
    data = bound_dataset(dataset)
    plan = {"data_validation_mode": submission.MODE, "datasets": [data]}
    if change == "missing-mode":
        plan.pop("data_validation_mode")
    elif change == "unknown-mode":
        plan["data_validation_mode"] = "decoded_samples"
    elif change == "mixed-old-probe":
        data["sample_probe"] = {"schema_version": 1, "status": "verified"}
    elif change == "missing-one-dataset":
        plan["datasets"].append(deepcopy(dataset))
    elif change == "legacy-only":
        data.pop("submission_validation")
        data["sample_probe"] = {"schema_version": 1, "status": "verified"}
    else:
        plan["datasets"] = []
    with pytest.raises(ValueError):
        validate_sample_evidence(plan)


def test_historical_v1_decoded_sample_plan_remains_supported(dataset):
    data = deepcopy(dataset)
    data["sample_probe"] = {"schema_version": 1, "status": "verified", "path": data["path"],
                            "checked_samples": 3, "sample_indices": [0, 2, 4],
                            "samples": [{"index": index} for index in (0, 2, 4)],
                            "action_horizon": 50, "action_sequence_keys": ["action"]}
    validate_sample_evidence({"datasets": [data]})
    validate_sample_evidence({"datasets": [{"path": "/legacy", "factory": "LegacyDataConfig"}]})


@pytest.fixture
def cached_dataset(tmp_path, monkeypatch, dataset):
    root = tmp_path / "dataset"
    for name in ("meta", "data", "videos"):
        (root / name).mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_frames":5,"total_episodes":1}')
    (root / "meta/episodes.jsonl").write_text('{"episode_index":0,"length":5}\n')
    (root / "data/episode.parquet").write_bytes(b"not-opened-as-parquet")
    (root / "videos/camera.mp4").write_bytes(b"not-opened-by-any-decoder")
    code = tmp_path / "openpi"
    for name in ("src/openpi", "third_party/lerobot/lerobot", "packages/openpi-client", "adapters"):
        directory = code / name
        directory.mkdir(parents=True)
        (directory / "implementation.py").write_text("VERSION = 1\n")
    data = deepcopy(dataset)
    data["path"] = str(root)
    data["metadata_sha256"] = hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest()
    data["timestamp_contract"]["metadata_sha256"] = data["metadata_sha256"]
    spec = {"allowed_roots": [str(tmp_path)], "code_dir": str(code),
            "validation_cache_dir": str(tmp_path / "private-receipts"),
            "inputs": {"dataset": {"path": str(root), "dataset_profile": {"fps": 20}}}}
    monkeypatch.setattr(cache, "_environment", lambda: {"python": "test"})
    original_tree = cache._tree
    adapters = Path(cache.__file__).resolve().parent
    monkeypatch.setattr(cache, "_tree", lambda path, budget, **options: original_tree(
        code / "adapters" if path == adapters else path, budget, **options))
    return data, spec


def prohibit_dataset_readers(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("submission must not load, scan or decode a dataset")

    monkeypatch.setattr(dataset_timestamps, "_inspect_dataset_timestamps", forbidden)
    monkeypatch.setattr(cache, "get_or_validate", forbidden)
    for name, attributes in {
            "openpi.training.data_loader": ("create_torch_dataset",),
            "lerobot.common.datasets.lerobot_dataset": ("LeRobotDataset",),
            "lerobot.common.datasets.video_utils": ("decode_video_frames",),
            "pyarrow.parquet": ("ParquetFile",),
    }.items():
        module = ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, forbidden)
        monkeypatch.setitem(sys.modules, name, module)


def test_required_source_reuses_registered_receipt_without_any_data_reader(cached_dataset, monkeypatch):
    data, spec = cached_dataset
    source = source_summary(data)
    cache.get_or_validate("source", data, spec, lambda: source, context={"scope": "dataset_source"})
    prohibit_dataset_readers(monkeypatch)
    assert submission.require_cached_dataset_source(data, spec) == source


@pytest.mark.parametrize("malformed", ["count", "fields", "missing-camera", "wrong-path"])
def test_matching_receipt_still_requires_complete_source_sample_evidence(cached_dataset, monkeypatch, malformed):
    data, spec = cached_dataset
    source = source_summary(data)
    if malformed == "count":
        source["raw_samples"]["checked_samples"] = 0
    elif malformed == "fields":
        source["raw_samples"]["samples"][0]["fields"] = []
    elif malformed == "missing-camera":
        source["raw_samples"]["samples"][0]["fields"].pop("observation.images.right")
    else:
        source["path"] += "-other"
    cache.get_or_validate("source", data, spec, lambda: source, context={"scope": "dataset_source"})
    prohibit_dataset_readers(monkeypatch)
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED"):
        submission.require_cached_dataset_source(data, spec)


@pytest.mark.parametrize("missing", ["directory", "receipt", "source-changed", "profile-changed", "code-changed"])
def test_required_source_miss_requires_base_validation_without_inline_loading(cached_dataset, monkeypatch, missing):
    data, spec = cached_dataset
    if missing != "directory":
        cache.get_or_validate("source", data, spec, lambda: source_summary(data), context={"scope": "dataset_source"})
    if missing == "receipt":
        next(Path(spec["validation_cache_dir"]).glob("*.json")).unlink()
    elif missing == "source-changed":
        (Path(data["path"]) / "videos/camera.mp4").write_bytes(b"changed")
    elif missing == "profile-changed":
        spec["inputs"]["dataset"]["dataset_profile"]["fps"] = 30
    elif missing == "code-changed":
        (Path(spec["code_dir"]) / "src/openpi/implementation.py").write_text("VERSION = 2\n")
    prohibit_dataset_readers(monkeypatch)
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED.*请先.*基础校验"):
        submission.require_cached_dataset_source(data, spec)
    if missing == "directory":
        assert not Path(spec["validation_cache_dir"]).exists()


def test_source_change_after_receipt_read_never_authorizes_submission(cached_dataset, monkeypatch):
    data, spec = cached_dataset
    cache.get_or_validate("source", data, spec, lambda: source_summary(data), context={"scope": "dataset_source"})
    read = cache._read

    def changed(*args):
        result = read(*args)
        (Path(data["path"]) / "data/episode.parquet").write_bytes(b"changed-after-receipt-read")
        return result

    monkeypatch.setattr(cache, "_read", changed)
    prohibit_dataset_readers(monkeypatch)
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED"):
        submission.require_cached_dataset_source(data, spec)
