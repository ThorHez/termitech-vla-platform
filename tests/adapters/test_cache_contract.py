import fcntl
import hashlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from vla_adapters import cache_contract
from vla_adapters.cache_contract import configure_cache, inspect_training_cache
from vla_adapters.training_plan import fingerprint


@pytest.fixture
def cache_runtime(tmp_path, monkeypatch):
    for key in cache_contract.CACHE_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    code = tmp_path / "code"
    code.mkdir()
    root = tmp_path / "cache"
    root.mkdir()
    rows = []
    for name in ("dataset-one", "dataset-two"):
        path = tmp_path / name
        (path / "data").mkdir(parents=True)
        rows.append({"path": str(path)})
    locks, builders = [], []

    class LocalFileLock:
        """Actual POSIX locks, supplied as the registered filelock dependency."""
        def __init__(self, path, timeout):
            self.path = path
            self.timeout = timeout

        def __enter__(self):
            locks.append(self.path)
            self.descriptor = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self

        def __exit__(self, *_):
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)

    def builder(name, *, data_dir):
        builders.append((name, data_dir))
        assert name == "parquet"
        suffix = hashlib.sha256(data_dir.encode()).hexdigest()
        with LocalFileLock(str(root / (suffix + ".lock")), timeout=1):
            directory = root / "parquet" / suffix
            directory.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(cache_dir=str(directory))

    datasets = ModuleType("datasets")
    datasets.config = SimpleNamespace(HF_DATASETS_CACHE=str(root))
    datasets.load_dataset_builder = builder
    datasets.load_dataset = lambda *a, **kw: pytest.fail("preflight must not load dataset examples")
    filelock = ModuleType("filelock")
    filelock.FileLock = LocalFileLock
    monkeypatch.setitem(sys.modules, "datasets", datasets)
    monkeypatch.setitem(sys.modules, "filelock", filelock)
    monkeypatch.setenv("HF_DATASETS_CACHE", str(root))
    return {"root": root, "rows": rows, "code": code, "datasets": datasets,
            "locks": locks, "builders": builders,
            "spec": {"code_dir": str(code), "allowed_roots": [str(tmp_path)]}}


def test_checks_actual_builder_locks_without_loading_or_modifying_existing_locks(cache_runtime):
    item = cache_runtime
    existing = item["root"] / "existing-unused.lock"
    existing.write_bytes(b"preserve previous lock contents")
    before = existing.stat().st_mtime_ns
    evidence = inspect_training_cache(item["rows"], item["spec"])
    assert evidence["datasets_cache"] == str(item["root"])
    assert evidence["paths"] == {"HF_DATASETS_CACHE": str(item["root"])}
    assert [row["path"] for row in evidence["builders"]] == [row["path"] for row in item["rows"]]
    assert item["builders"] == [("parquet", row["path"] + "/data") for row in item["rows"]]
    assert len(item["locks"]) == 3  # One new lock probe and two real builder locks.
    assert existing.read_bytes() == b"preserve previous lock contents"
    assert existing.stat().st_mtime_ns == before
    assert not list(item["root"].glob(".vla-cache-probe-*"))


def test_existing_readonly_lock_fails_before_truncation_or_builder(cache_runtime, monkeypatch):
    item = cache_runtime
    existing = item["root"] / "root-owned.lock"
    existing.write_bytes(b"previous run")
    original_open = os.open

    def denied(path, flags, *args, **kwargs):
        if str(path) == str(existing):
            assert flags & os.O_RDWR
            assert not flags & (os.O_TRUNC | os.O_CREAT)
            raise PermissionError(13, "Permission denied", str(path))
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(cache_contract.os, "open", denied)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNWRITABLE.*root-owned.lock"):
        inspect_training_cache(item["rows"], item["spec"])
    assert item["builders"] == item["locks"] == []
    assert existing.read_bytes() == b"previous run"


def test_new_cache_lock_probe_uses_real_lock_and_cleans_failure(cache_runtime, monkeypatch):
    item = cache_runtime

    class FailedLock:
        def __init__(self, path, timeout):
            self.path = path
            assert timeout == 1

        def __enter__(self):
            raise PermissionError(13, "lock denied", self.path)

        def __exit__(self, *_):
            pass

    monkeypatch.setattr(sys.modules["filelock"], "FileLock", FailedLock)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNWRITABLE.*lock denied"):
        inspect_training_cache(item["rows"], item["spec"])
    assert not list(item["root"].glob(".vla-cache-probe-*"))
    assert item["builders"] == []


def test_builder_lock_denial_is_actionable(cache_runtime, monkeypatch):
    item = cache_runtime

    def denied(*args, **kwargs):
        raise PermissionError(13, "builder lock denied", str(item["root"] / "builder.lock"))

    monkeypatch.setattr(item["datasets"], "load_dataset_builder", denied)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNWRITABLE.*builder.lock"):
        inspect_training_cache(item["rows"], item["spec"])


def test_symlink_lock_is_rejected_without_touching_target(cache_runtime, tmp_path):
    item = cache_runtime
    target = tmp_path / "untouched"
    target.write_bytes(b"untouched")
    (item["root"] / "link.lock").symlink_to(target)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNWRITABLE"):
        inspect_training_cache(item["rows"], item["spec"])
    assert target.read_bytes() == b"untouched"
    assert item["builders"] == []


def test_cache_scan_is_bounded(cache_runtime, monkeypatch):
    item = cache_runtime
    (item["root"] / "one").touch()
    (item["root"] / "two").touch()
    monkeypatch.setattr(cache_contract, "MAX_CACHE_ENTRIES", 1)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNAVAILABLE.*上限"):
        inspect_training_cache(item["rows"], item["spec"])
    assert item["builders"] == []


def test_environment_and_imported_datasets_cache_must_match(cache_runtime, monkeypatch):
    item = cache_runtime
    monkeypatch.setenv("HF_DATASETS_CACHE", str(item["root"].parent / "another"))
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNAVAILABLE.*导入训练库之前"):
        inspect_training_cache(item["rows"], item["spec"])
    assert item["builders"] == []


@pytest.mark.parametrize("source", ["datasets", "environment", "builder"])
def test_cache_paths_cannot_escape_managed_roots(cache_runtime, monkeypatch, source):
    item = cache_runtime
    if source == "datasets":
        item["datasets"].config.HF_DATASETS_CACHE = "/outside/datasets"
    elif source == "environment":
        monkeypatch.setenv("TMPDIR", "/outside/tmp")
    else:
        monkeypatch.setattr(item["datasets"], "load_dataset_builder", lambda *a, **kw: SimpleNamespace(cache_dir="/outside/builder"))
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNAVAILABLE.*受管目录"):
        inspect_training_cache(item["rows"], item["spec"])


def test_builder_cache_must_use_the_checked_datasets_cache(cache_runtime, monkeypatch):
    item = cache_runtime
    other = item["root"].parent / "another-cache"
    monkeypatch.setattr(item["datasets"], "load_dataset_builder", lambda *a, **kw: SimpleNamespace(cache_dir=str(other)))
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNAVAILABLE.*另一缓存目录"):
        inspect_training_cache(item["rows"], item["spec"])


def test_environment_path_and_helper_source_drift_change_frozen_contract(cache_runtime, monkeypatch):
    item = cache_runtime
    first = inspect_training_cache(item["rows"], item["spec"])
    monkeypatch.setenv("TMPDIR", str(item["root"].parent / "tmp"))
    second = inspect_training_cache(item["rows"], item["spec"])
    assert fingerprint(first) != fingerprint(second)
    helper = item["code"] / "src/openpi/shared/runtime_paths.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("def configure_training_cache(): pass\n")
    third = inspect_training_cache(item["rows"], item["spec"])
    assert third["runtime_paths_sha256"] == hashlib.sha256(helper.read_bytes()).hexdigest()
    helper.write_text(helper.read_text() + "# changed helper\n")
    fourth = inspect_training_cache(item["rows"], item["spec"])
    assert fingerprint(third) != fingerprint(fourth)


def test_configures_only_verified_project_helper_before_library_imports(cache_runtime, monkeypatch):
    item = cache_runtime
    marker = item["root"] / "configured-first"
    helper = item["code"] / "src/openpi/shared/runtime_paths.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("from pathlib import Path\nimport os\n"
                      "def configure_training_cache():\n"
                      "    assert 'openpi.training.config' not in __import__('sys').modules\n"
                      f"    Path({str(marker)!r}).write_text(__file__)\n"
                      f"    os.environ.setdefault('HF_DATASETS_CACHE', {str(item['root'])!r})\n")
    entry = item["code"] / "scripts/train.py"
    entry.parent.mkdir()
    entry.write_text("raise AssertionError('must not import training entrypoint')\n")
    monkeypatch.delenv("HF_DATASETS_CACHE")
    configure_cache(item["spec"])
    assert marker.read_text() == str(helper)
    assert os.environ["HF_DATASETS_CACHE"] == str(item["root"])
    assert "vla_registered_openpi_runtime_paths" not in sys.modules
    assert not list(helper.parent.rglob("*.pyc"))


def test_missing_project_helper_preserves_existing_environment(cache_runtime, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", "preserve-existing-setting")
    configure_cache(cache_runtime["spec"])
    assert os.environ["HF_DATASETS_CACHE"] == "preserve-existing-setting"


def test_project_helper_cannot_follow_symlink_outside_checkout(cache_runtime, tmp_path):
    item = cache_runtime
    outside = tmp_path / "outside-helper.py"
    outside.write_text("raise AssertionError('outside source must not run')\n")
    helper = item["code"] / "src/openpi/shared/runtime_paths.py"
    helper.parent.mkdir(parents=True)
    helper.symlink_to(outside)
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNAVAILABLE.*训练项目内"):
        configure_cache(item["spec"])


def test_cache_initialization_permission_failure_is_reported(cache_runtime):
    item = cache_runtime
    helper = item["code"] / "src/openpi/shared/runtime_paths.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("def configure_training_cache():\n    raise PermissionError(13, 'cannot create project cache')\n")
    with pytest.raises(ValueError, match="TRAINING_CACHE_UNWRITABLE.*cannot create project cache"):
        configure_cache(item["spec"])
