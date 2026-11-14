"""Submission reuses real private receipts and never falls back to data loading."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from vla_adapters import dataset_samples, dataset_source_validation, dataset_timestamps, openpi_entry
from vla_adapters import dataset_validation_cache as cache
from vla_adapters import submission_validation, training_plan
from vla_adapters.sample_evidence import validate_sample_evidence
from test_training_plan import training  # noqa: F401


@pytest.fixture
def submitted(training, monkeypatch):
    base, spec, module, loaders = training
    # Registration's expensive source readers have their own tests. Here its
    # completed summaries are persisted through the actual secure receipt code;
    # none of the submission receipt readers is mocked.
    registered_timestamps = training_plan.require_cached_dataset_timestamps
    registered_source = training_plan.require_cached_dataset_source
    code = Path(spec["code_dir"])
    for relative in ("src/openpi", "third_party/lerobot/lerobot", "packages/openpi-client"):
        source = code / relative / "implementation.py"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("VERSION = 1\n")
    for factory in base.data.datasets:
        info = Path(factory.repo_id) / "meta/info.json"
        info.write_text(json.dumps({**json.loads(info.read_text()), "fps": 20}))
    spec["validation_cache_dir"] = str(code.parent / "private-source-receipts")
    runtime = {"validator_path": str(Path(__file__).resolve()),
               "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    tolerance = 1e-4
    monkeypatch.setattr(dataset_timestamps, "_runtime", lambda: (runtime, tolerance))
    config = openpi_entry.resolve_config(base, spec, module, loaders)
    data = []
    for factory in config.data.datasets:
        dataset = training_plan.dataset_contract(factory, spec)
        data.append(dataset)
        cache.get_or_validate("timestamps", dataset, spec,
            lambda dataset=dataset: {**registered_timestamps(dataset, spec), "runtime": runtime,
                                     "tolerance_s": tolerance},
            context={"runtime": runtime, "tolerance_s": tolerance})
        cache.get_or_validate("source", dataset, spec,
            lambda dataset=dataset: registered_source(dataset, spec),
            context={"scope": "dataset_source"})
    monkeypatch.setattr(training_plan, "require_cached_dataset_timestamps", dataset_timestamps.require_cached_dataset_timestamps)
    monkeypatch.setattr(training_plan, "require_cached_dataset_source", submission_validation.require_cached_dataset_source)

    def forbidden(*args, **kwargs):
        pytest.fail("Submission must not reload datasets, decode images, scan timestamps or refresh missing source evidence")

    monkeypatch.setattr(dataset_samples, "inspect_composed_samples_isolated", forbidden)
    monkeypatch.setattr(dataset_samples, "inspect_composed_samples", forbidden)
    monkeypatch.setattr(dataset_source_validation, "_raw_samples", forbidden)
    monkeypatch.setattr(dataset_timestamps, "_inspect_dataset_timestamps", forbidden)
    monkeypatch.setattr(training_plan, "inspect_composed_samples_isolated", forbidden, raising=False)
    loader = ModuleType("openpi.training.data_loader")
    loader.create_torch_dataset = forbidden
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    native = ModuleType("lerobot.common.datasets.lerobot_dataset")
    native.LeRobotDataset = forbidden
    monkeypatch.setitem(sys.modules, native.__name__, native)
    return config, spec, data


def receipt_files(spec):
    return {path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in Path(spec["validation_cache_dir"]).iterdir() if path.is_file()}


def test_multi_dataset_submission_and_repeated_start_checks_reuse_receipts_without_native_reads(submitted):
    config, spec, _ = submitted
    before = receipt_files(spec)
    plan = training_plan.inspect_training_config(config, spec, checks=[])
    for _ in range(2):
        again = training_plan.inspect_training_config(config, spec, checks=[])
        assert again == plan
        validate_sample_evidence(again)
    assert len(plan["datasets"]) == 3
    for dataset in plan["datasets"]:
        assert "sample_probe" not in dataset
        evidence = dataset["submission_validation"]
        assert evidence["scope"] == "registered_source_and_configuration"
        assert evidence["source_evidence"]["raw_samples"]["checked_samples"] == 3
    assert receipt_files(spec) == before, "Submission may read receipts, never rewrite them"


@pytest.mark.parametrize("kind", ["source", "timestamps"])
def test_missing_basic_evidence_blocks_submission_without_revalidation_fallback(submitted, kind):
    config, spec, datasets = submitted
    context = {"scope": "dataset_source"}
    if kind == "timestamps":
        runtime, tolerance = dataset_timestamps._runtime()
        context = {"runtime": runtime, "tolerance_s": tolerance}
    key = cache.cache_key(kind, datasets[1], spec, context=context)
    assert key is not None
    (Path(spec["validation_cache_dir"]) / f"{kind}-{key}.json").unlink()
    before = receipt_files(spec)
    checks = []
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED.*基础校验"):
        training_plan.inspect_training_config(config, spec, checks=checks)
    assert checks[-1]["dataset_asset_id"] == spec["inputs"]["dataset_1"]["id"]
    assert receipt_files(spec) == before


def test_preflight_lists_all_affected_assets_with_distinct_missing_or_stale_receipts(submitted, monkeypatch, capsys):
    config, spec, datasets = submitted
    runtime, tolerance = dataset_timestamps._runtime()
    source_context = {"scope": "dataset_source"}
    timestamp_context = {"runtime": runtime, "tolerance_s": tolerance}
    for kind, dataset, context in (
        ("timestamps", datasets[0], timestamp_context),
        ("source", datasets[1], source_context),
    ):
        key = cache.cache_key(kind, dataset, spec, context=context)
        assert key is not None
        (Path(spec["validation_cache_dir"]) / f"{kind}-{key}.json").unlink()
    (Path(datasets[2]["path"]) / "data/episode_000000.parquet").write_bytes(b"changed source rows")
    before = receipt_files(spec)
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda request: (config, None))
    with pytest.raises(SystemExit) as stopped:
        openpi_entry._run_spec(spec, validate_only=True)
    assert stopped.value.code == 2
    response = json.loads(capsys.readouterr().out.split("VLA_PREFLIGHT:")[-1])
    assert "plan" not in response
    assert response["error"]["code"] == "DATASET_BASE_VALIDATION_REQUIRED"
    assert "3 个数据集" in response["error"]["message"]
    failures = [check for check in response["checks"] if "dataset_asset_id" in check]
    assert [(check["dataset_asset_id"], check["name"]) for check in failures] == [
        (spec["inputs"]["dataset_0"]["id"], "dataset_timestamps"),
        (spec["inputs"]["dataset_1"]["id"], "dataset_samples"),
        (spec["inputs"]["dataset_2"]["id"], "dataset_timestamps"),
    ]
    assert all(check["ok"] is False and check["code"] == response["error"]["code"] for check in failures)
    assert receipt_files(spec) == before


@pytest.mark.parametrize("change", ["parquet", "metadata", "missing_cache"])
def test_source_drift_or_missing_cache_never_triggers_a_new_dataset_read(submitted, change):
    config, spec, data = submitted
    if change == "parquet":
        (Path(data[1]["path"]) / "data/episode_000000.parquet").write_bytes(b"changed source rows")
    elif change == "metadata":
        info = Path(data[1]["path"]) / "meta/info.json"
        info.write_text(json.dumps({**json.loads(info.read_text()), "fps": 25}))
    else:
        spec["validation_cache_dir"] = str(Path(spec["validation_cache_dir"]).parent / "never-validated")
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED"):
        training_plan.inspect_training_config(config, spec, checks=[])


def test_cached_source_does_not_allow_invalid_current_training_configuration(submitted):
    config, spec, _ = submitted
    config.batch_size = 71
    with pytest.raises(ValueError, match="INVALID_BATCH_SIZE"):
        training_plan.inspect_training_config(config, spec, checks=[])


def test_cli_preflight_and_final_entry_recheck_use_metadata_receipts_only(submitted, monkeypatch, capsys):
    config, spec, _ = submitted
    before = receipt_files(spec)
    monkeypatch.setattr(openpi_entry, "prepare_config", lambda request: (config, None))
    openpi_entry._run_spec(spec, validate_only=True)
    response = json.loads(capsys.readouterr().out.split("VLA_PREFLIGHT:")[-1])
    assert "error" not in response
    assert not Path(spec["output_dir"]).exists()
    # The real training child performs this final plan/digest check before any
    # training entry or optimizer is invoked. Deliberately refuse the final
    # digest so the test exercises that path without starting training.
    changed = {**deepcopy(spec), "expected_plan_digest": "f" * 64}
    with pytest.raises(ValueError, match="ASSETS_CHANGED_SINCE_PREFLIGHT"):
        openpi_entry._run_spec(changed)
    assert receipt_files(spec) == before
    assert not Path(spec["output_dir"]).exists()
