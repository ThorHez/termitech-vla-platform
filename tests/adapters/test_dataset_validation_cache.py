"""Persisted data receipts cannot outlive their source/configuration identity."""

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vla_adapters import dataset_validation_cache as cache
from vla_adapters import dataset_timestamps, training_plan
from vla_adapters.training_plan import fingerprint


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    for name in ("meta", "data", "videos"):
        (root / name).mkdir(parents=True)
    (root / "meta/info.json").write_text('{"total_frames":3,"total_episodes":1}')
    (root / "meta/episodes.jsonl").write_text('{"episode_index":0,"length":3}\n')
    (root / "meta/tasks.jsonl").write_text('{"task":"pick"}\n')
    (root / "data/episode.parquet").write_bytes(b"parquet-input")
    (root / "videos/camera.mp4").write_bytes(b"video-input")
    norm = tmp_path / "norm_stats.json"
    norm.write_text('{"norm_stats":{"state":{"mean":[0]}}}')
    code = tmp_path / "openpi"
    for name in ("src/openpi", "third_party/lerobot/lerobot", "packages/openpi-client"):
        path = code / name / "implementation.py"
        path.parent.mkdir(parents=True)
        path.write_text("VERSION = 1\n")
    data = {"path": str(root), "frames": 3, "episodes": 1, "effective_fps": 20,
            "metadata_sha256": hashlib.sha256((root / "meta/info.json").read_bytes()).hexdigest(),
            "data_manifest_digest": "manifest", "norm_path": str(norm)}
    spec = {"allowed_roots": [str(tmp_path)], "code_dir": str(code),
            "validation_cache_dir": str(tmp_path / "private-receipts"),
            "inputs": {"dataset": {"path": str(root), "dataset_profile": {"fps": 20}}}}
    monkeypatch.setattr(cache, "_environment", lambda: {"python": "test", "libraries": {"numpy": "1"}})
    return data, spec


def success():
    return {"schema_version": 1, "status": "verified", "checked_samples": 3,
            "sample_indices": [0, 1, 2], "scope": "source samples"}


def call_counter():
    calls = []

    def validate():
        calls.append(True)
        return success()

    return calls, validate


def context(data):
    return {"composition": {"state_history": 1}, "binding": {"processor": "composed"},
            "model": {"action_horizon": 16}, "norm_path": data["norm_path"]}


@pytest.mark.parametrize("kind", ["timestamps", "samples", "source"])
def test_cache_returns_original_stable_evidence_without_mutating_plan_inputs(source, kind):
    data, spec = source
    original = deepcopy((data, spec))
    calls, validate = call_counter()
    options = {"context": context(data)} if kind == "samples" else {}
    cold = cache.get_or_validate(kind, data, spec, validate, **options)
    warm = cache.get_or_validate(kind, data, spec, validate, **options)
    assert len(calls) == 1 and cold == warm == success()
    assert fingerprint({"datasets": [{"evidence": cold}]}) == fingerprint({"datasets": [{"evidence": warm}]})
    assert (data, spec) == original
    directory = Path(spec["validation_cache_dir"])
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in directory.iterdir())


def test_registration_and_training_share_source_identity_without_run_parameters(source):
    data, spec = source
    registered = deepcopy(spec)
    registered["registered_profile"] = registered["inputs"]["dataset"].pop("dataset_profile")
    trained = deepcopy(spec)
    trained.update(recipe="different", output_dir="/different/output", gpu_uuids=["GPU-1"],
                   parameters={"steps": 1, "learning_rate": 9e-5}, run_id="another", gpu_count=8)
    trained["inputs"]["dataset"]["id"] = "training-asset-id"
    assert cache.cache_key("timestamps", data, registered) == cache.cache_key("timestamps", data, trained)
    assert cache.cache_key("source", data, registered, context={"scope": "dataset_source"}) == cache.cache_key(
        "source", data, trained, context={"scope": "dataset_source"})


def test_same_size_write_preserving_mtime_invalidates_receipt_by_ctime(source):
    data, spec = source
    calls, validate = call_counter()
    cache.get_or_validate("timestamps", data, spec, validate)
    path = Path(data["path"]) / "data/episode.parquet"
    before = path.stat()
    path.write_bytes(b"changed-input")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size and path.stat().st_mtime_ns == before.st_mtime_ns
    assert path.stat().st_ctime_ns != before.st_ctime_ns
    cache.get_or_validate("timestamps", data, spec, validate)
    assert len(calls) == 2


@pytest.mark.parametrize("relative", ["meta/episodes.jsonl", "meta/tasks.jsonl", "data/episode.parquet", "videos/camera.mp4"])
def test_each_source_tree_change_invalidates_cache(source, relative):
    data, spec = source
    first = cache.cache_key("source", data, spec)
    path = Path(data["path"]) / relative
    path.write_bytes(path.read_bytes() + b" ")
    assert first is not None and cache.cache_key("source", data, spec) != first


@pytest.mark.parametrize("relative", ["src/openpi", "third_party/lerobot/lerobot", "packages/openpi-client"])
def test_all_native_python_source_roots_invalidate_cache(source, relative):
    data, spec = source
    first = cache.cache_key("timestamps", data, spec)
    (Path(spec["code_dir"]) / relative / "implementation.py").write_text("VERSION = 2\n")
    assert first is not None and cache.cache_key("timestamps", data, spec) != first


def test_model_composition_binding_norm_profile_fps_and_environment_are_bound(source, monkeypatch):
    data, spec = source
    original = context(data)
    first = cache.cache_key("samples", data, spec, context=original)
    assert first is not None
    for field, value in (("composition", {"state_history": 2}), ("binding", {"processor": "another"}),
                         ("model", {"action_horizon": 32}), ("stage", "value")):
        changed = {**original, field: value}
        assert cache.cache_key("samples", data, spec, context=changed) != first
    assert cache.cache_key("samples", {**data, "effective_fps": 60}, spec, context=original) != first
    changed_spec = deepcopy(spec)
    changed_spec["inputs"]["dataset"]["dataset_profile"]["extra"] = "changed"
    assert cache.cache_key("samples", data, changed_spec, context=original) != first
    Path(data["norm_path"]).write_text("different norm")
    changed_norm = cache.cache_key("samples", data, spec, context=original)
    assert changed_norm != first
    monkeypatch.setattr(cache, "_environment", lambda: {"python": "different"})
    assert cache.cache_key("samples", data, spec, context=original) != changed_norm


def test_runtime_context_is_fresh_on_each_key_and_validator_mutation_is_rejected(source):
    data, spec = source
    runtime = {"tolerance": 1e-4}
    first = cache.cache_key("timestamps", data, spec, context=lambda: runtime)
    runtime["tolerance"] = 1e-2
    assert cache.cache_key("timestamps", data, spec, context=lambda: runtime) != first

    def changed():
        runtime["tolerance"] = 1
        return success()

    with pytest.raises(ValueError, match="DATASET_VALIDATION_SOURCE_CHANGED"):
        cache.get_or_validate("timestamps", data, spec, changed, context=lambda: runtime)
    assert not list(Path(spec["validation_cache_dir"]).glob("*.json"))


@pytest.mark.parametrize("bad", [b"not JSON", b"[]", b'{"evidence":{}}'])
def test_corrupt_receipt_is_a_miss_and_real_check_repairs_it(source, bad):
    data, spec = source
    calls, validate = call_counter()
    cache.get_or_validate("source", data, spec, validate)
    receipt = next(Path(spec["validation_cache_dir"]).glob("*.json"))
    receipt.write_bytes(bad)
    assert cache.get_or_validate("source", data, spec, validate) == success()
    assert len(calls) == 2
    cache.get_or_validate("source", data, spec, validate)
    assert len(calls) == 2


def test_modified_evidence_digest_is_not_trusted(source):
    data, spec = source
    calls, validate = call_counter()
    cache.get_or_validate("timestamps", data, spec, validate)
    receipt = next(Path(spec["validation_cache_dir"]).glob("*.json"))
    document = json.loads(receipt.read_text())
    document["evidence"]["checked_samples"] = 0
    receipt.write_text(json.dumps(document))
    cache.get_or_validate("timestamps", data, spec, validate)
    assert len(calls) == 2


@pytest.mark.parametrize("target", ["directory", "receipt", "lock", "source", "source_parent"])
def test_links_cannot_authorize_reuse_or_touch_targets(source, tmp_path, target):
    data, spec = source
    calls, validate = call_counter()
    outside = tmp_path / "outside"
    outside.write_text("untouched")
    if target in {"receipt", "lock"}:
        cache.get_or_validate("source", data, spec, validate)
        path = next(Path(spec["validation_cache_dir"]).glob("*.json" if target == "receipt" else "*.lock"))
        path.unlink()
        path.symlink_to(outside)
    elif target == "directory":
        Path(spec["validation_cache_dir"]).symlink_to(tmp_path)
    elif target == "source":
        path = Path(data["path"]) / "videos/camera.mp4"
        path.unlink()
        path.symlink_to(outside)
    else:
        real = Path(data["path"]) / "videos"
        moved = tmp_path / "moved-videos"
        real.rename(moved)
        real.symlink_to(moved)
    count = len(calls)
    assert cache.get_or_validate("source", data, spec, validate) == success()
    assert len(calls) == count + 1 and outside.read_text() == "untouched"


@pytest.mark.parametrize("mode", [0o755, 0o777])
def test_nonprivate_cache_directory_is_never_used(source, mode):
    data, spec = source
    directory = Path(spec["validation_cache_dir"])
    directory.mkdir(mode=mode)
    directory.chmod(mode)
    calls, validate = call_counter()
    assert cache.get_or_validate("source", data, spec, validate) == success()
    assert cache.get_or_validate("source", data, spec, validate) == success()
    assert len(calls) == 2 and not list(directory.iterdir())


def test_symlink_loop_in_cache_location_is_a_full_validation_miss(source, tmp_path):
    data, spec = source
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    changed = {**spec, "validation_cache_dir": str(loop / "receipts")}
    calls, validate = call_counter()
    assert cache.cache_key("source", data, changed) is None
    assert cache.get_or_validate("source", data, changed, validate) == success()
    assert len(calls) == 1


def test_missing_or_outside_cache_uses_full_validator(source):
    data, spec = source
    calls, validate = call_counter()
    for changed in ({**spec, "validation_cache_dir": None}, {**spec, "validation_cache_dir": "/outside/receipts"},
                    {**spec, "validation_cache_dir": str(Path(data["path"]) / "receipts")}):
        assert cache.cache_key("source", data, changed) is None
        assert cache.get_or_validate("source", data, changed, validate) == success()
    assert len(calls) == 3


def test_failed_and_not_applicable_checks_are_not_persisted(source):
    data, spec = source
    for status in ("failed", "not_applicable"):
        calls = []

        def validate():
            calls.append(1)
            return {"schema_version": 1, "status": status}

        cache.get_or_validate("source", data, spec, validate)
        cache.get_or_validate("source", data, spec, validate)
        assert len(calls) == 2
    assert not list(Path(spec["validation_cache_dir"]).glob("*.json"))


def test_validator_errors_are_propagated_even_when_cache_io_also_uses_oserror(source):
    data, spec = source

    def validate():
        raise PermissionError("dataset unreadable")

    with pytest.raises(PermissionError, match="dataset unreadable"):
        cache.get_or_validate("source", data, spec, validate)
    assert not list(Path(spec["validation_cache_dir"]).glob("*.json"))


def test_source_change_during_real_validation_does_not_write_receipt(source):
    data, spec = source

    def validate():
        (Path(data["path"]) / "data/episode.parquet").write_bytes(b"changed")
        return success()

    with pytest.raises(ValueError, match="DATASET_VALIDATION_SOURCE_CHANGED"):
        cache.get_or_validate("source", data, spec, validate)
    assert not list(Path(spec["validation_cache_dir"]).glob("*.json"))


def test_change_after_reading_a_hit_cannot_return_old_evidence(source, monkeypatch):
    data, spec = source
    calls, validate = call_counter()
    cache.get_or_validate("source", data, spec, validate)
    original = cache._read

    def read(*args):
        result = original(*args)
        (Path(data["path"]) / "data/episode.parquet").write_bytes(b"changed")
        return result

    monkeypatch.setattr(cache, "_read", read)
    with pytest.raises(ValueError, match="DATASET_VALIDATION_SOURCE_CHANGED"):
        cache.get_or_validate("source", data, spec, validate)
    assert len(calls) == 2


@pytest.mark.parametrize("kind", ["timestamps", "samples", "source"])
def test_lookup_reuses_matching_receipt_with_only_readonly_opens(source, monkeypatch, kind):
    data, spec = source
    options = {"context": context(data)} if kind == "samples" else {}
    cache.get_or_validate(kind, data, spec, success, **options)
    directory = Path(spec["validation_cache_dir"])
    before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ctime_ns)
              for path in directory.iterdir()}
    original_open = cache.os.open

    def readonly_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        return original_open(path, flags, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("lookup must not validate, create directories, write receipts or acquire locks")

    monkeypatch.setattr(cache.os, "open", readonly_open)
    for name in ("_cache_directory", "_lock", "_write", "get_or_validate"):
        monkeypatch.setattr(cache, name, forbidden)
    monkeypatch.setattr(cache.os, "mkdir", forbidden)
    monkeypatch.setattr(cache.fcntl, "flock", forbidden)
    assert cache.lookup_cached(kind, data, spec, **options) == success()
    after = {path.name: (path.read_bytes(), path.stat().st_mtime_ns, path.stat().st_ctime_ns)
             for path in directory.iterdir()}
    assert after == before


def test_lookup_missing_cache_never_creates_directory_or_falls_back(source, monkeypatch):
    data, spec = source

    def forbidden(*args, **kwargs):
        raise AssertionError("a cache miss must remain a read-only miss")

    monkeypatch.setattr(cache.os, "mkdir", forbidden)
    monkeypatch.setattr(cache, "get_or_validate", forbidden)
    assert cache.lookup_cached("source", data, spec) is None
    assert not Path(spec["validation_cache_dir"]).exists()
    assert cache.lookup_cached("source", data, {**spec, "validation_cache_dir": None}) is None


@pytest.mark.parametrize("bad", [b"not JSON", b"[]", b'{"evidence":{}}'])
def test_lookup_corrupt_receipt_is_a_miss_without_repair(source, bad):
    data, spec = source
    cache.get_or_validate("source", data, spec, success)
    receipt = next(Path(spec["validation_cache_dir"]).glob("*.json"))
    receipt.write_bytes(bad)
    assert cache.lookup_cached("source", data, spec) is None
    assert receipt.read_bytes() == bad


@pytest.mark.parametrize("unsafe", ["directory-mode", "receipt-mode", "directory-link", "receipt-link", "receipt-hardlink"])
def test_lookup_unsafe_cache_cannot_authorize_reuse_or_change_files(source, tmp_path, unsafe):
    data, spec = source
    cache.get_or_validate("source", data, spec, success)
    directory = Path(spec["validation_cache_dir"])
    receipt = next(directory.glob("*.json"))
    original = receipt.read_bytes()
    if unsafe == "directory-mode":
        directory.chmod(0o755)
    elif unsafe == "receipt-mode":
        receipt.chmod(0o644)
    elif unsafe == "directory-link":
        real = tmp_path / "moved-receipts"
        directory.rename(real)
        directory.symlink_to(real, target_is_directory=True)
    elif unsafe == "receipt-link":
        real = tmp_path / "moved-receipt.json"
        receipt.rename(real)
        receipt.symlink_to(real)
    else:
        os.link(receipt, tmp_path / "another-receipt.json")
    assert cache.lookup_cached("source", data, spec) is None
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("relative", ["meta/episodes.jsonl", "data/episode.parquet", "videos/camera.mp4"])
def test_lookup_changed_source_requires_new_evidence_without_validation(source, relative):
    data, spec = source
    cache.get_or_validate("source", data, spec, success)
    path = Path(data["path"]) / relative
    path.write_bytes(path.read_bytes() + b"changed")
    assert cache.lookup_cached("source", data, spec) is None


def test_lookup_rechecks_fresh_source_after_reading_receipt(source, monkeypatch):
    data, spec = source
    cache.get_or_validate("source", data, spec, success)
    original = cache._read

    def read(*args):
        result = original(*args)
        (Path(data["path"]) / "data/episode.parquet").write_bytes(b"changed-during-lookup")
        return result

    monkeypatch.setattr(cache, "_read", read)
    assert cache.lookup_cached("source", data, spec) is None


def test_lookup_rechecks_callable_context_after_reading_receipt(source, monkeypatch):
    data, spec = source
    runtime = {"tolerance_s": 1e-4}
    cache.get_or_validate("timestamps", data, spec, success, context=lambda: runtime)
    original = cache._read

    def read(*args):
        result = original(*args)
        runtime["tolerance_s"] = 1e-3
        return result

    monkeypatch.setattr(cache, "_read", read)
    assert cache.lookup_cached("timestamps", data, spec, context=lambda: runtime) is None


def test_required_timestamps_reuses_registration_context_without_scanning_rows(source, monkeypatch):
    data, spec = source
    runtime = {"source_sha256": "actual-runtime"}
    tolerance = 1e-4
    monkeypatch.setattr(dataset_timestamps, "_runtime", lambda: (runtime, tolerance))
    registered = deepcopy(spec)
    registered["registered_profile"] = registered["inputs"]["dataset"].pop("dataset_profile")
    cache.get_or_validate("timestamps", data, registered, success,
                          context={"runtime": runtime, "tolerance_s": tolerance})

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not scan timestamps or refresh registration evidence")

    monkeypatch.setattr(dataset_timestamps, "_inspect_dataset_timestamps", forbidden)
    monkeypatch.setattr(cache, "get_or_validate", forbidden)
    assert dataset_timestamps.require_cached_dataset_timestamps(data, spec) == success()


@pytest.mark.parametrize("missing", ["directory", "receipt", "cache-disabled", "source-changed", "tolerance-changed"])
def test_required_timestamps_missing_or_stale_evidence_directs_to_base_validation(source, monkeypatch, missing):
    data, spec = source
    runtime = {"tolerance_s": 1e-4}
    monkeypatch.setattr(dataset_timestamps, "_runtime", lambda: ({"source_sha256": "actual-runtime"}, runtime["tolerance_s"]))
    if missing != "directory":
        cache.get_or_validate("timestamps", data, spec, success,
                              context={"runtime": {"source_sha256": "actual-runtime"}, "tolerance_s": runtime["tolerance_s"]})
    if missing == "receipt":
        next(Path(spec["validation_cache_dir"]).glob("*.json")).unlink()
    elif missing == "cache-disabled":
        spec["validation_cache_dir"] = None
    elif missing == "source-changed":
        (Path(data["path"]) / "videos/camera.mp4").write_bytes(b"changed")
    elif missing == "tolerance-changed":
        runtime["tolerance_s"] = 1e-3

    def forbidden(*args, **kwargs):
        raise AssertionError("training must not scan timestamps or refresh registration evidence")

    monkeypatch.setattr(dataset_timestamps, "_inspect_dataset_timestamps", forbidden)
    monkeypatch.setattr(cache, "get_or_validate", forbidden)
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED.*请先.*基础校验"):
        dataset_timestamps.require_cached_dataset_timestamps(data, spec)


def test_cross_process_requests_share_one_full_validation(source, tmp_path):
    data, spec = source
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"data": data, "spec": spec}))
    counter = tmp_path / "validator-calls.txt"
    program = '''
import json, os, sys, time
from pathlib import Path
from vla_adapters.dataset_validation_cache import get_or_validate
request = json.loads(Path(sys.argv[1]).read_text())
def validate():
    fd = os.open(sys.argv[2], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.write(fd, b'validated\\n')
    os.close(fd)
    time.sleep(.25)
    return {"schema_version": 1, "status": "verified", "scope": "source"}
print(json.dumps(get_or_validate("source", request["data"], request["spec"], validate)))
'''
    processes = [subprocess.Popen([sys.executable, "-c", program, str(request), str(counter)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    outputs = [process.communicate(timeout=20) for process in processes]
    assert all(process.returncode == 0 for process in processes), outputs
    assert counter.read_text().splitlines() == ["validated"]
    assert all(json.loads(stdout)["status"] == "verified" for stdout, _ in outputs)


def test_public_timestamp_check_shares_registration_receipt_and_binds_actual_tolerance(source, monkeypatch):
    data, spec = source
    calls, validate = call_counter()
    monkeypatch.setattr(dataset_timestamps, "_inspect_dataset_timestamps", lambda data, spec: validate())
    runtime = {"tolerance": 1e-4}
    monkeypatch.setattr(dataset_timestamps, "_runtime", lambda: ({"source_sha256": "actual-runtime"}, runtime["tolerance"]))
    cold = dataset_timestamps.inspect_dataset_timestamps(data, spec)
    registered = deepcopy(spec)
    registered["registered_profile"] = registered["inputs"]["dataset"].pop("dataset_profile")
    assert dataset_timestamps.inspect_dataset_timestamps(data, registered) == cold
    assert len(calls) == 1
    runtime["tolerance"] = 1e-3
    dataset_timestamps.inspect_dataset_timestamps(data, registered)
    assert len(calls) == 2


def test_inspect_training_config_requires_registration_then_reuses_without_loading(source, tmp_path, monkeypatch):
    data, cache_spec = source
    norm_path = data["norm_path"]
    code = Path(cache_spec["code_dir"])
    implementation = code / "src/openpi/implementation.py"
    base_data = {**data, "data_manifest_digest": training_plan.file_manifest(Path(data["path"]) / "data", cache_spec)[2],
                 "composition": {"source": {"path": "src/openpi/implementation.py",
                                             "sha256": hashlib.sha256(implementation.read_bytes()).hexdigest()}}}
    entry = code / "train.py"
    entry.write_text("def main(config): pass\n")
    model_path = tmp_path / "model"
    model_path.mkdir()
    factory = SimpleNamespace(repo_id=data["path"])
    config = SimpleNamespace(data=factory, batch_size=2, model=SimpleNamespace(action_horizon=16, action_dim=32),
                             weight_loader=SimpleNamespace(params_path=str(model_path)))
    spec = {**cache_spec, "recipe": "test", "gpu_count": 1, "parameters": {"steps": 1},
            "inputs": {"dataset": {"id": "dataset", "path": data["path"], "dataset_profile": {"fps": 20}},
                       "norm": {"id": "norm", "path": norm_path},
                       "model": {"id": "model", "path": str(model_path)}}}
    from vla_adapters.submission_validation import require_cached_dataset_source
    timestamp_receipt = {"schema_version": 1, "status": "verified", "checked_rows": 3,
                         "checked_episodes": base_data["episodes"], "effective_fps": base_data["effective_fps"],
                         "metadata_sha256": base_data["metadata_sha256"], "data_manifest_digest": base_data["data_manifest_digest"]}
    source_receipt = {"schema_version": 1, "status": "verified", "path": data["path"],
                      "raw_samples": {"selection": "first_middle_last", "checked_samples": 3,
                                      "sample_indices": [0, 1, 2], "samples": [
                                          {"index": i, "fields": {"state": {"shape": [2], "dtype": "float32"}}}
                                          for i in (0, 1, 2)]}}
    def timestamps(data, spec):
        receipt = cache.lookup_cached("timestamps", data, spec)
        if receipt is None:
            raise ValueError("DATASET_BASE_VALIDATION_REQUIRED")
        return receipt

    monkeypatch.setattr(training_plan, "dataset_contract", lambda *a, **kw: deepcopy(base_data))
    monkeypatch.setattr(training_plan, "require_cached_dataset_timestamps", timestamps)
    monkeypatch.setattr(training_plan, "require_cached_dataset_source", require_cached_dataset_source)
    monkeypatch.setattr(training_plan, "norm_contract", lambda *a: {
        "norm_path": norm_path, "norm_sha256": hashlib.sha256(Path(norm_path).read_bytes()).hexdigest()})
    monkeypatch.setattr(training_plan, "inspect_stage", lambda *a: None)
    monkeypatch.setattr(training_plan, "model_contract", lambda *a: ({"path": str(model_path)}, 0))
    monkeypatch.setattr(training_plan, "training_entrypoint", lambda *a: entry)
    monkeypatch.setattr(training_plan, "generated_training_config", lambda *a, **kw: {"source": "same-config"})
    monkeypatch.setattr(training_plan, "storage_estimate", lambda *a: {"required_bytes": 1})
    monkeypatch.setattr(training_plan, "inspect_training_cache", lambda *a: {
        "datasets_cache": "/same-cache", "builders": [{"path": data["path"]}]})
    monkeypatch.setattr(training_plan, "resolve_registered_fps", lambda *a: {})
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED"):
        training_plan.inspect_training_config(config, spec, checks=[])
    # Only registration can execute the validators and persist their receipts.
    cache.get_or_validate("timestamps", base_data, spec, lambda: timestamp_receipt)
    cache.get_or_validate("source", base_data, spec, lambda: source_receipt, context={"scope": "dataset_source"})
    before = {p.name: (p.stat().st_mtime_ns, p.read_bytes())
              for p in Path(spec["validation_cache_dir"]).iterdir()}
    monkeypatch.setattr(cache, "get_or_validate", lambda *a, **kw: pytest.fail("submission cannot validate or repair caches"))
    first = training_plan.inspect_training_config(config, spec, checks=[])
    again = training_plan.inspect_training_config(config, spec, checks=[])
    assert first == again and fingerprint(first) == fingerprint(again)
    assert first["data_validation_mode"] == "registered_source"
    assert "sample_probe" not in first["datasets"][0]
    after = {p.name: (p.stat().st_mtime_ns, p.read_bytes())
             for p in Path(spec["validation_cache_dir"]).iterdir()}
    assert before == after
    (Path(data["path"]) / "videos/camera.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="DATASET_BASE_VALIDATION_REQUIRED"):
        training_plan.inspect_training_config(config, spec, checks=[])
